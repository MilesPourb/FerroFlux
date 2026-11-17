"""Zero-leakage quarantine and license/ToS compliance firewalls for FerroFlux-FM."""

from __future__ import annotations

from typing import Any

__all__ = [
    "QuarantineViolationError",
    "LicenseViolationError",
    "load_quarantine_sets",
    "is_quarantined",
    "assert_not_quarantined",
    "filter_dataframe_quarantine",
    "audit_parquet_directory",
    "freeze_checkpoint_sha256",
    "is_quarantined",
    "verify_frozen_checkpoint",
    "assert_ag2_inference_only",
    "normalize_license_label",
    "filter_by_license_track",
]


def __getattr__(name: str) -> Any:
    if name in {
        "LicenseViolationError",
        "assert_ag2_inference_only",
        "filter_by_license_track",
        "normalize_license_label",
    }:
        from ferroflux.security import license_guard as _lg

        return getattr(_lg, name)
    if name in {
        "QuarantineViolationError",
        "assert_not_quarantined",
        "audit_parquet_directory",
        "filter_dataframe_quarantine",
        "freeze_checkpoint_sha256",
        "is_quarantined",
        "load_quarantine_sets",
        "verify_frozen_checkpoint",
    }:
        from ferroflux.security import quarantine as _q

        return getattr(_q, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
