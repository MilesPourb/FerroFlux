"""License Track Separation ('open' vs 'academic') & AlphaGenome 2 Terms-of-Service Guard.

Enforces two legal/licensing invariants across FerroFlux-FM:
1. AlphaGenome 2 API outputs (governed by https://alphagenome.google/terms non-commercial
   anti-distillation terms) are STRICTLY forbidden during gradient-based model training
   (`is_training=True` raises `LicenseViolationError(RuntimeError)`). Pretraining uses only
   clean-room Apache-2.0 / CC0 / CC-BY experimental ENCODE-4 & Epigenome Roadmap tracks.
2. Dual-checkpoint weight release hygiene:
   - `track='open'` retains ONLY permissive / commercial-safe datasets (CC0, Public Domain,
     Apache-2.0, MIT, CC-BY-4.0, permissive) -> produces Apache-2.0 `FerroFlux-Open`.
   - `track='academic'` retains both `'open'` and non-commercial academic datasets
     (CC-BY-NC-4.0, CC-BY-NC-SA-4.0, Parse-10M academic, X-Atlas/Orion, RxRx3-core)
     -> produces non-commercial `FerroFlux-Academic`.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


class LicenseViolationError(RuntimeError):
    """Raised when AlphaGenome 2 ToS or open-weights license constraints are violated."""


_OPEN_EXACT = frozenset(
    {
        "open",
        "permissive",
        "public_domain",
        "public-domain",
        "public domain",
        "ncbi-public",
        "unrestricted",
        "cc0",
        "cc0-1.0",
        "cc-by",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa",
        "cc-by-sa-4.0",
        "apache-2.0",
        "apache2",
        "apache",
        "mit",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "odc-by",
        "odb-by",
    }
)

_NC_PATTERN = re.compile(r"(?i)(?:^|[-_\s/])(nc|non[-_\s]?commercial|academic|research[-_\s]?only)(?:$|[-_\s/])")
_RESTRICTED_PATTERN = re.compile(
    r"(?i)(?:^|[-_\s/])(proprietary|quarantined|forbidden|alphagenome[-_\s]?tos|dua[-_\s]?gated)(?:$|[-_\s/])"
)


def assert_ag2_inference_only(is_training: bool, context: str = "CisRegPriorSlot") -> None:
    """Enforce AlphaGenome 2 Terms of Service: never train/distill weights on AG2 API outputs."""
    if is_training:
        raise LicenseViolationError(
            f"ALPHAGENOME 2 ToS VIOLATION in {context}: AlphaGenome 2 API outputs cannot be used "
            "when model.training=True (https://alphagenome.google/terms prohibits training/distilling "
            "downstream ML models on AG2 predictions). Use clean-room ENCODE/Roadmap experimental "
            "tracks during training and restrict AG2 to frozen zero-shot eval() inference."
        )


def normalize_license_label(label: str | None) -> str:
    """Normalize raw SPDX / dataset license strings to {'open', 'academic', 'restricted'}."""
    if label is None:
        return "restricted"
    cleaned = str(label).strip().lower()
    if not cleaned:
        return "restricted"
    if _RESTRICTED_PATTERN.search(cleaned):
        return "restricted"
    if _NC_PATTERN.search(cleaned):
        return "academic"
    if cleaned in _OPEN_EXACT or cleaned.startswith(("cc0", "apache", "mit", "bsd", "public")):
        return "open"
    if cleaned.startswith("cc-by") and "nc" not in cleaned and "nd" not in cleaned:
        return "open"
    return "restricted"


def filter_by_license_track(
    data: pd.DataFrame | list[dict[str, Any]],
    track: str = "open",
    license_col: str | None = None,
) -> pd.DataFrame | list[dict[str, Any]]:
    """Filter a DataFrame or list of records according to 'open' or 'academic' track rules."""
    norm_track = track.strip().lower()
    if norm_track in ("open", "permissive", "commercial"):
        allowed = {"open"}
    elif norm_track in ("academic", "non-commercial", "non_commercial", "all_eligible"):
        allowed = {"open", "academic"}
    else:
        raise ValueError(f"Unknown license track {track!r}; expected 'open' or 'academic'.")

    if isinstance(data, list):
        if not data:
            return []
        col = license_col or next(
            (k for k in ("license_track", "license", "data_license", "spdx_license") if k in data[0]),
            "license_track",
        )
        return [row for row in data if normalize_license_label(row.get(col)) in allowed]

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame or list[dict], got {type(data)}")

    if data.empty:
        return data.copy()

    if license_col is None:
        for candidate in ("license_track", "license", "data_license", "spdx_license"):
            if candidate in data.columns:
                license_col = candidate
                break
    if license_col is None or license_col not in data.columns:
        raise KeyError(
            f"DataFrame lacks a license column (expected 'license_track' or 'license'); got {list(data.columns)}"
        )

    normalized = data[license_col].map(normalize_license_label)
    mask = normalized.isin(allowed)
    return data.loc[mask].reset_index(drop=True)
