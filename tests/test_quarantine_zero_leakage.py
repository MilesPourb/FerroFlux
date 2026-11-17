"""Zero-Leakage Quarantine, AlphaGenome 2 ToS, License Guard, and Inode Audit Tests.

Verifies that:
1. All 7 quarantined Bradbury/Overholtzer/Wiesner accessions (including Round-2 catch GSE310954)
   and all 15 quarantined PMCIDs are present in configs/ and trigger QuarantineViolationError.
2. Contributor/author strings (including middle-initial-free 'Michelle,,Bradbury' SOFT format)
   trigger QuarantineViolationError, while unrelated investigators pass cleanly.
3. Frozen-checkpoint SHA-256 lock detects any post-freeze weight tampering before Stage-0 evaluation.
4. AlphaGenome 2 ToS guard raises RuntimeError whenever AG2 outputs are invoked during model.train().
5. Dual-track license filter ('open' vs 'academic') cleanly removes CC-BY-NC rows from the Open track.
6. Processed Parquet directory (/anvil/projects/x-cis261485/Ferroptosis_ML/data/processed) stays
   strictly under <500 files (<1M inode cap) and contains 0 quarantined IDs across all columns.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
CONFIGS_DIR = ROOT_DIR / "configs"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

QUARANTINED_ACCESSIONS = [
    "GSE319397",
    "GSE319419",
    "GSE310954",
    "10.5281/zenodo.18271042",
    "18271042",
    "GSE196336",
    "GSE20286",
    "GSE20285",
]

QUARANTINED_PMCIDS = [
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
]


def test_quarantine_config_manifests_exist_and_complete() -> None:
    acc_file = CONFIGS_DIR / "exclude_accessions.txt"
    pmc_file = CONFIGS_DIR / "exclude_pmcids.txt"
    assert acc_file.is_file(), f"Missing {acc_file}"
    assert pmc_file.is_file(), f"Missing {pmc_file}"

    acc_text = acc_file.read_text(encoding="utf-8")
    for acc in QUARANTINED_ACCESSIONS:
        assert acc in acc_text, f"Quarantined accession {acc!r} missing from {acc_file}"

    pmc_text = pmc_file.read_text(encoding="utf-8")
    for pmcid in QUARANTINED_PMCIDS:
        assert pmcid in pmc_text, f"Quarantined PMCID {pmcid!r} missing from {pmc_file}"


@pytest.mark.parametrize(
    "blocked_id",
    QUARANTINED_ACCESSIONS
    + QUARANTINED_PMCIDS
    + [
        "gse310954",
        "Zenodo:10.5281/zenodo.18271042",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE310nnn/GSE310954/",
        "https://zenodo.org/records/18271042",
        "pmc12916478.1",
    ],
)
def test_assert_not_quarantined_blocks_every_holdout_id(blocked_id: str) -> None:
    from ferroflux.security import quarantine

    err_cls = getattr(quarantine, "QuarantineViolationError", Exception)
    assert issubclass(err_cls, Exception)
    with pytest.raises(err_cls):
        quarantine.assert_not_quarantined(blocked_id)


@pytest.mark.parametrize(
    "blocked_author",
    [
        "Michelle S. Bradbury",
        "Michelle,,Bradbury",  # Exact GEO SOFT format that hid GSE310954!
        "Bradbury MS",
        "Ulrich Wiesner",
        "Wiesner, U.",
        "Michael Overholtzer",
        "Overholtzer M",
    ],
)
def test_assert_not_quarantined_blocks_bradbury_wiesner_overholtzer_authors(blocked_author: str) -> None:
    from ferroflux.security import quarantine

    err_cls = getattr(quarantine, "QuarantineViolationError", Exception)
    sig = inspect.signature(quarantine.assert_not_quarantined)
    with pytest.raises(err_cls):
        if "authors" in sig.parameters:
            quarantine.assert_not_quarantined("GSE999999", authors=blocked_author)
        else:
            quarantine.assert_not_quarantined(f"GSE999999 {blocked_author}")


@pytest.mark.parametrize(
    "safe_id,safe_author",
    [
        ("GSE261778", "Korangath P, Ivkov R"),
        ("GSE312613", "Ghandehari H"),
        ("GSE221165", "Zhao J, Stockwell BR"),
        ("PXD030304", "Goncalves E, Reddel RR"),
        ("ST004225", "Li Y, Jiang X"),
        ("MTBLS13306", "Li M, Gan B"),
        ("PMC13241834", "Conrad M, Dixon SJ"),
    ],
)
def test_assert_not_quarantined_allows_clean_public_datasets(safe_id: str, safe_author: str) -> None:
    from ferroflux.security import quarantine

    sig = inspect.signature(quarantine.assert_not_quarantined)
    if "authors" in sig.parameters:
        quarantine.assert_not_quarantined(safe_id, authors=safe_author)
    else:
        quarantine.assert_not_quarantined(safe_id)


def test_frozen_checkpoint_sha256_lock_detects_tampering(tmp_path: Path) -> None:
    from ferroflux.security import quarantine

    ckpt = tmp_path / "ferroflux_frozen.pt"
    manifest = tmp_path / "frozen_sha256.json"
    torch.save({"weights": torch.randn(8, 8)}, ckpt)

    quarantine.freeze_checkpoint_sha256(ckpt, manifest)
    assert manifest.is_file()
    # Unmodified checkpoint must verify cleanly
    quarantine.verify_frozen_checkpoint(ckpt, manifest)

    # Mutating a single byte after freeze MUST fail verification
    raw = bytearray(ckpt.read_bytes())
    raw[-1] ^= 0xFF
    ckpt.write_bytes(bytes(raw))
    with pytest.raises(Exception):
        quarantine.verify_frozen_checkpoint(ckpt, manifest)


def test_alphagenome2_tos_guard_in_license_guard_and_cis_reg_slot() -> None:
    from ferroflux.layers import cis_reg_slot
    from ferroflux.security import license_guard

    # 1. Direct guard check
    license_guard.assert_ag2_inference_only(is_training=False)
    with pytest.raises(RuntimeError):
        license_guard.assert_ag2_inference_only(is_training=True)

    # 2. CisRegPriorSlot module check
    slot_cls = getattr(cis_reg_slot, "CisRegPriorSlot")
    sig = inspect.signature(slot_cls.__init__)
    kwargs = {}
    if "d_model" in sig.parameters:
        kwargs["d_model"] = 64
    elif "hidden_dim" in sig.parameters:
        kwargs["hidden_dim"] = 64
    slot = slot_cls(**kwargs)

    # Clean-room experimental epigenomic features MUST work in both train() and eval()
    slot.train()
    f_sig = inspect.signature(slot.forward)
    b = 4
    n_loci = getattr(slot, "n_loci", 64)
    in_dim = getattr(slot, "in_features", getattr(slot, "epigenomic_dim", getattr(slot, "feat_dim", 8)))
    clean_epigen = torch.randn(b, n_loci, in_dim)

    if "use_ag2" in f_sig.parameters:
        out_train = slot(clean_epigen, use_ag2=False)
        assert torch.isfinite(out_train if isinstance(out_train, torch.Tensor) else out_train[0]).all()
        with pytest.raises(RuntimeError):
            slot(clean_epigen, use_ag2=True)
    elif "ag2_tracks" in f_sig.parameters:
        out_train = slot(clean_epigen, ag2_tracks=None)
        assert torch.isfinite(out_train if isinstance(out_train, torch.Tensor) else out_train[0]).all()
        with pytest.raises(RuntimeError):
            slot(clean_epigen, ag2_tracks=torch.randn_like(clean_epigen))
    elif "source" in f_sig.parameters:
        out_train = slot(clean_epigen, source="experimental")
        assert torch.isfinite(out_train if isinstance(out_train, torch.Tensor) else out_train[0]).all()
        with pytest.raises(RuntimeError):
            slot(clean_epigen, source="alphagenome2")
    else:
        # Check dedicated AG2 adapter method if separate from forward()
        ag2_methods = [
            m for m in dir(slot) if "ag2" in m.lower() or "alphagenome" in m.lower()
        ]
        assert ag2_methods, "CisRegPriorSlot must expose an AG2 inference path guarded by assert_ag2_inference_only"
        method = getattr(slot, ag2_methods[0])
        with pytest.raises(RuntimeError):
            method(clean_epigen)


def test_dual_license_track_filtering() -> None:
    from ferroflux.security.license_guard import filter_by_license_track

    df = pd.DataFrame(
        {
            "dataset_id": ["Tahoe-100M", "CCLE", "X-Atlas/Orion", "Parse-10M", "RxRx3-core"],
            "license_track": ["permissive", "CC-BY-4.0", "CC-BY-NC-SA-4.0", "academic", "CC-BY-NC-SA-4.0"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    open_df = filter_by_license_track(df, track="open")
    acad_df = filter_by_license_track(df, track="academic")

    assert set(open_df["dataset_id"]) == {"Tahoe-100M", "CCLE"}
    assert len(acad_df) == 5


def test_processed_parquet_directory_inode_cap_and_zero_leakage() -> None:
    from ferroflux.security import quarantine

    assert PROCESSED_DIR.is_dir(), f"Processed directory {PROCESSED_DIR} not yet created"
    all_files = [p for p in (ROOT_DIR / "data").rglob("*") if p.is_file() and "stage0_blind_lockbox" not in str(p)]
    assert len(all_files) < 500, f"Inode cap guard violated: {len(all_files)} files in data/ (limit < 500)"

    parquet_files = sorted(PROCESSED_DIR.glob("*.parquet"))
    assert len(parquet_files) >= 10, f"Expected >=10 processed Parquet shards, found {len(parquet_files)}"

    # Run official CLI directory audit (which executes quarantine.audit_parquet_directory)
    cli_res = subprocess.run(
        [sys.executable, "-m", "ferroflux.security.quarantine", "--audit-data-dir", str(PROCESSED_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli_res.returncode == 0, f"CLI quarantine audit failed: {cli_res.stderr}\n{cli_res.stdout}"

    # Independent verifier scan across every string column of every Parquet shard
    forbidden_tokens = {tok.upper() for tok in QUARANTINED_ACCESSIONS + QUARANTINED_PMCIDS if tok != "18271042"} | {
        "ZENODO.18271042"
    }
    for pq_path in parquet_files:
        meta = pq.read_metadata(pq_path)
        assert meta.num_rows > 0, f"Empty Parquet shard: {pq_path}"
        schema = pq.read_schema(pq_path)
        import pyarrow as pa

        str_cols = [
            f.name
            for f in schema
            if pa.types.is_string(f.type)
            or pa.types.is_large_string(f.type)
            or (pa.types.is_dictionary(f.type) and pa.types.is_string(f.type.value_type))
        ]
        if not str_cols:
            continue
        df = pq.read_table(pq_path, columns=str_cols).to_pandas()
        for col in str_cols:
            joined = " ".join(df[col].dropna().astype(str).unique()).upper()
            for bad in forbidden_tokens:
                assert bad not in joined, f"LEAK DETECTED: {bad} found in {pq_path.name} column {col!r}!"
