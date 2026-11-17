"""Locked Stage-0 Blind Bradbury/Overholtzer/Wiesner Validation & Nature Figure Suite."""

from __future__ import annotations

from typing import Any

__all__ = [
    "verify_all_checkpoints_locked",
    "run_stage0_blind_evaluation",
    "generate_all_nature_figures",
]


def __getattr__(name: str) -> Any:
    if name in ("verify_all_checkpoints_locked", "run_stage0_blind_evaluation"):
        from ferroflux.validation import stage0_bradbury_blind_eval as _s0

        return getattr(_s0, name)
    if name == "generate_all_nature_figures":
        from ferroflux.validation import generate_nature_figures as _fig

        return getattr(_fig, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
