"""Hardened Zero-Leakage Quarantine Firewall, Inode Guard & SHA-256 Checkpoint Lock.

Enforces three non-negotiable holdout & infrastructure invariants for FerroFlux-FM:
1. ZERO bytes from PI Michelle S. Bradbury, Michael Overholtzer, or Ulrich Wiesner datasets
   (7 omics/imaging datasets incl. GSE310954 + 15 PMCIDs + author/contributor regex)
   may enter any pretraining, fine-tuning, feature-selection, or vocabulary-building stage.
2. Processed data directories on Purdue Anvil (/anvil/projects/x-cis261485) must remain
   strictly under the <500-file guard (<1M project inode cap).
3. Pretrained checkpoints must be SHA-256 hashed and cryptographically verified prior to
   any Stage-0 blind holdout evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
CONFIGS_DIR = ROOT_DIR / "configs"
DEFAULT_ACCESSIONS_FILE = CONFIGS_DIR / "exclude_accessions.txt"
DEFAULT_PMCIDS_FILE = CONFIGS_DIR / "exclude_pmcids.txt"
DEFAULT_FROZEN_MANIFEST = CONFIGS_DIR / "frozen_checkpoints_sha256.json"

# Hardcoded fallback sets guarantee protection even if config files are moved
_BUILTIN_ACCESSIONS: frozenset[str] = frozenset(
    {
        "GSE319397",
        "GSE319419",
        "GSE310954",
        "10.5281/ZENODO.18271042",
        "ZENODO.18271042",
        "18271042",
        "GSE196336",
        "GSE20286",
        "GSE20285",
    }
)

_BUILTIN_PMCIDS: frozenset[str] = frozenset(
    {
        "PMC5108575",
        "PMC7644276",
        "PMC12916478",
        "PMC13014609",
        "PMC13314319",
        "PMC13440254",
        "PMC7310667",
        "PMC7171978",
        "PMC6175906",
        "PMC5679295",
        "PMC5675572",
        "PMC6420223",
        "PMC6589353",
        "PMC11392819",
        "PMC11059358",
    }
)

# Matches Bradbury, Wiesner, or Overholtzer across standard citations AND GEO SOFT
# middle-initial-omitted contributor strings such as 'Michelle,,Bradbury'
_AUTHOR_QUARANTINE_RE = re.compile(r"(?i)(?:^|[^a-z])(bradbury|wiesner|overholtzer)(?:$|[^a-z])")


class QuarantineViolationError(RuntimeError):
    """Raised whenever a quarantined dataset accession, DOI, PMCID, or contributor is detected."""


def _read_manifest_lines(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    items: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            items.add(line.upper())
    return items


@lru_cache(maxsize=8)
def load_quarantine_sets(
    accessions_file: Path | str = DEFAULT_ACCESSIONS_FILE,
    pmcids_file: Path | str = DEFAULT_PMCIDS_FILE,
) -> tuple[frozenset[str], frozenset[str]]:
    """Load canonical upper-cased quarantined accession/DOI and PMCID sets."""
    acc_path = Path(accessions_file)
    pmc_path = Path(pmcids_file)
    accs = set(_BUILTIN_ACCESSIONS) | _read_manifest_lines(acc_path)
    pmcs = set(_BUILTIN_PMCIDS) | _read_manifest_lines(pmc_path)
    return frozenset(accs), frozenset(pmcs)


@lru_cache(maxsize=8)
def _compiled_id_patterns(
    accessions_file: Path | str = DEFAULT_ACCESSIONS_FILE,
    pmcids_file: Path | str = DEFAULT_PMCIDS_FILE,
    include_bare_zenodo_numeric: bool = True,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    accs, pmcs = load_quarantine_sets(accessions_file, pmcids_file)
    all_tokens = sorted(accs | pmcs, key=len, reverse=True)
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for tok in all_tokens:
        if tok == "18271042" and not include_bare_zenodo_numeric:
            continue
        # Require non-alphanumeric before token and non-digit after token so e.g.
        # 'pmc12916478.1', 'https://zenodo.org/records/18271042', and '.../GSE310954/'
        # all match while a hypothetical 6-digit 'GSE202850' does not false-positive.
        pat = re.compile(rf"(?i)(?:^|[^A-Z0-9])({re.escape(tok)})(?:$|[^0-9])")
        compiled.append((tok, pat))
    return tuple(compiled)


def _find_quarantine_hit(
    *texts: str | None,
    authors: str | Iterable[str] | None = None,
    include_bare_zenodo_numeric: bool = True,
) -> str | None:
    patterns = _compiled_id_patterns(include_bare_zenodo_numeric=include_bare_zenodo_numeric)

    all_strings: list[str] = []
    for t in texts:
        if t:
            all_strings.append(str(t))

    author_strings: list[str] = []
    if authors is not None:
        if isinstance(authors, str):
            author_strings.append(authors)
        else:
            author_strings.extend(str(a) for a in authors if a)

    for s in all_strings + author_strings:
        for tok, pat in patterns:
            if pat.search(s):
                return f"quarantined_identifier:{tok} in {s!r}"
        m_auth = _AUTHOR_QUARANTINE_RE.search(s)
        if m_auth:
            return f"quarantined_investigator:{m_auth.group(1)} in {s!r}"

    return None


def is_quarantined(
    identifier: str | None,
    authors: str | Iterable[str] | None = None,
    extra_metadata: str | None = None,
) -> bool:
    """Return True if identifier, authors, or metadata match any Tier-3 holdout rule."""
    return _find_quarantine_hit(identifier, extra_metadata, authors=authors) is not None


def assert_not_quarantined(
    identifier: str | None,
    authors: str | Iterable[str] | None = None,
    extra_metadata: str | None = None,
) -> None:
    """Raise QuarantineViolationError immediately if any holdout identifier or author matches."""
    hit = _find_quarantine_hit(identifier, extra_metadata, authors=authors)
    if hit is not None:
        raise QuarantineViolationError(
            f"HOLD-OUT QUARANTINE VIOLATION BLOCKED ({hit}). "
            "Tier-3 Bradbury/Overholtzer/Wiesner datasets are strictly locked out prior to Stage-0."
        )


def filter_dataframe_quarantine(
    df: pd.DataFrame,
    id_columns: Sequence[str] | None = None,
    author_columns: Sequence[str] | None = None,
    raise_on_hit: bool = False,
) -> pd.DataFrame:
    """Filter (or fail hard on) any rows in a DataFrame matching quarantined IDs or authors."""
    if df.empty:
        return df.copy()

    if id_columns is None:
        id_columns = [
            c
            for c in df.columns
            if any(
                k in c.lower()
                for k in ("accession", "dataset", "source", "pmcid", "pmid", "doi", "url", "study", "id")
            )
        ]
    if author_columns is None:
        author_columns = [
            c for c in df.columns if any(k in c.lower() for k in ("author", "contributor", "pi", "lab"))
        ]

    check_cols = [c for c in dict.fromkeys(list(id_columns) + list(author_columns)) if c in df.columns]
    if not check_cols:
        check_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    if not check_cols:
        return df.copy()

    bad_mask = pd.Series(False, index=df.index)
    for col in check_cols:
        col_is_id_like = any(
            k in col.lower() for k in ("accession", "dataset", "source", "pmcid", "doi", "url", "study")
        )
        s_vals = df[col].fillna("").astype(str)
        hits = s_vals.map(
            lambda v: _find_quarantine_hit(v, include_bare_zenodo_numeric=col_is_id_like) is not None
            if v
            else False
        )
        bad_mask = bad_mask | hits

    if bad_mask.any() and raise_on_hit:
        first_idx = bad_mask[bad_mask].index[0]
        row_dict = df.loc[first_idx, check_cols].to_dict()
        raise QuarantineViolationError(f"Quarantined row detected at index {first_idx}: {row_dict}")

    return df.loc[~bad_mask].reset_index(drop=True)


def _arrow_type_can_hold_text(dtype: pa.DataType) -> bool:
    """Return True if an Arrow column type can hold string/categorical/nested text; False for pure numerics."""
    if (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
    ):
        return True
    if pa.types.is_dictionary(dtype):
        return _arrow_type_can_hold_text(dtype.value_type)
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        return _arrow_type_can_hold_text(dtype.value_type)
    if pa.types.is_struct(dtype):
        return any(_arrow_type_can_hold_text(f.type) for f in dtype)
    if pa.types.is_map(dtype):
        return _arrow_type_can_hold_text(dtype.key_type) or _arrow_type_can_hold_text(dtype.item_type)
    return False


def audit_parquet_directory(
    data_dir: Path | str,
    max_files: int = 500,
) -> dict[str, Any]:
    """Audit a processed Parquet directory for <max_files inode compliance and zero holdout leakage."""
    target_dir = Path(data_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Audit target directory does not exist: {target_dir}")

    data_root = target_dir.parent if target_dir.name == "processed" else target_dir
    all_files = [
        p
        for p in data_root.rglob("*")
        if p.is_file() and "stage0_blind_lockbox" not in str(p)
    ]
    if len(all_files) >= max_files:
        raise QuarantineViolationError(
            f"Inode budget violated under {data_root}: found {len(all_files)} files (strict cap < {max_files})."
        )

    parquet_files = sorted(target_dir.glob("*.parquet"))
    total_rows = 0
    for pq_path in parquet_files:
        fname_hit = _find_quarantine_hit(pq_path.name, include_bare_zenodo_numeric=True)
        if fname_hit is not None:
            raise QuarantineViolationError(f"ZERO-LEAKAGE AUDIT FAILED in filename {pq_path.name!r}: {fname_hit}")

        pf = pq.ParquetFile(pq_path)
        total_rows += pf.metadata.num_rows
        schema_arrow = pf.schema_arrow

        if schema_arrow.metadata:
            for m_key, m_val in schema_arrow.metadata.items():
                if m_key in (b"ARROW:schema", b"pandas"):
                    continue
                meta_text = f"{m_key.decode('utf-8', 'ignore')}={m_val.decode('utf-8', 'ignore')}"
                m_hit = _find_quarantine_hit(meta_text, include_bare_zenodo_numeric=True)
                if m_hit is not None:
                    raise QuarantineViolationError(
                        f"ZERO-LEAKAGE AUDIT FAILED in {pq_path.name} schema metadata: {m_hit}"
                    )

        text_cols = [field.name for field in schema_arrow if _arrow_type_can_hold_text(field.type)]
        if not text_cols:
            continue

        accs_set, pmcs_set = load_quarantine_sets()
        re2_pat = "|".join([re.escape(tok) for tok in sorted(accs_set | pmcs_set, key=len, reverse=True)] + ["bradbury", "wiesner", "overholtzer"])

        table = pq.read_table(pq_path, columns=text_cols)
        for col in text_cols:
            col_is_id_like = any(
                k in col.lower() for k in ("accession", "dataset", "source", "pmcid", "doi", "url", "study")
            )
            arr = table[col]
            dtype = arr.type
            if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_dictionary(dtype):
                uniq_str = pc.unique(arr.cast(pa.large_string()) if pa.types.is_dictionary(dtype) else arr).drop_null()
                if len(uniq_str) == 0:
                    continue
                mask = pc.match_substring_regex(uniq_str, pattern=re2_pat, ignore_case=True)
                matched = pc.filter(uniq_str, mask)
                if len(matched) == 0:
                    continue
                unique_vals = [str(v) for v in matched.to_pylist() if v is not None]
            else:
                s_pd = arr.to_pandas()
                unique_vals = [str(v) for v in s_pd.dropna().astype(str).unique()]

            for val in unique_vals:
                hit = _find_quarantine_hit(val, include_bare_zenodo_numeric=col_is_id_like)
                if hit is not None:
                    raise QuarantineViolationError(
                        f"ZERO-LEAKAGE AUDIT FAILED in {pq_path.name} column {col!r}: {hit}"
                    )

    return {
        "audited_dir": str(target_dir),
        "total_files_in_data_tree": len(all_files),
        "parquet_shards_audited": len(parquet_files),
        "total_rows_audited": total_rows,
        "leaks_found": 0,
        "status": "CLEAN",
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_checkpoint_sha256(
    checkpoint_path: Path | str,
    manifest_path: Path | str = DEFAULT_FROZEN_MANIFEST,
) -> dict[str, Any]:
    """Compute SHA-256 of a pretrained checkpoint and record an immutable lock entry."""
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint to freeze not found: {ckpt}")

    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    digest = _sha256_file(ckpt)
    payload: dict[str, Any] = {"checkpoints": {}}
    if manifest_file.is_file():
        try:
            loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and "checkpoints" in loaded:
                payload = loaded
        except json.JSONDecodeError:
            pass

    entry = {
        "path": str(ckpt),
        "name": ckpt.name,
        "sha256": digest,
        "size_bytes": ckpt.stat().st_size,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["checkpoints"][str(ckpt)] = entry
    payload["checkpoints"][ckpt.name] = entry
    manifest_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return entry


def verify_frozen_checkpoint(
    checkpoint_path: Path | str,
    manifest_path: Path | str = DEFAULT_FROZEN_MANIFEST,
) -> dict[str, Any]:
    """Verify a checkpoint's live SHA-256 matches its frozen pre-Stage-0 lock entry."""
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise QuarantineViolationError(f"Frozen SHA-256 manifest missing: {manifest_file}")

    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    ckpts = payload.get("checkpoints", {})
    entry = ckpts.get(str(ckpt)) or ckpts.get(ckpt.name)
    if entry is None:
        raise QuarantineViolationError(f"Checkpoint {ckpt} has no frozen SHA-256 entry in {manifest_file}")

    live_digest = _sha256_file(ckpt)
    expected_digest = entry.get("sha256")
    if live_digest != expected_digest:
        raise QuarantineViolationError(
            f"FROZEN CHECKPOINT TAMPERING DETECTED for {ckpt}: "
            f"expected SHA-256 {expected_digest}, got {live_digest}!"
        )
    return entry


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FerroFlux-FM Zero-Leakage Quarantine & SHA-256 Lock CLI")
    parser.add_argument("--audit-data-dir", type=Path, help="Audit Parquet directory for zero leakage & <500 files")
    parser.add_argument("--max-files", type=int, default=500, help="Maximum total files allowed under data/")
    parser.add_argument("--freeze-ckpt", type=Path, help="Freeze checkpoint SHA-256 into manifest")
    parser.add_argument("--verify-frozen-sha256", type=Path, help="Verify checkpoint SHA-256 against manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST, help="Path to SHA-256 manifest JSON")
    parser.add_argument("--check-id", type=str, help="Verify a single accession/PMCID/DOI is not quarantined")
    parser.add_argument("--authors", type=str, default=None, help="Optional author string for --check-id")
    args = parser.parse_args(argv)

    if args.check_id:
        assert_not_quarantined(args.check_id, authors=args.authors)
        print(json.dumps({"checked_id": args.check_id, "status": "ALLOWED"}))
    if args.audit_data_dir:
        report = audit_parquet_directory(args.audit_data_dir, max_files=args.max_files)
        print(json.dumps(report, indent=2))
    if args.freeze_ckpt:
        entry = freeze_checkpoint_sha256(args.freeze_ckpt, manifest_path=args.manifest)
        print(json.dumps(entry, indent=2))
    if args.verify_frozen_sha256:
        entry = verify_frozen_checkpoint(args.verify_frozen_sha256, manifest_path=args.manifest)
        print(json.dumps({"verified": entry["path"], "sha256": entry["sha256"], "status": "MATCH"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
