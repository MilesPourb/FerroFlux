"""Training & Compositional OOD / Ablation Evaluation Suite for FerroFlux-FM."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ABLATION_7_SPECS",
    "CANONICAL_6_OOD_SPLITS",
    "bootstrap_ci_1000",
    "SyntheticCurriculumDataset",
    "load_training_dataset",
    "train_single_seed",
]


def __getattr__(name: str) -> Any:
    if name in {"ABLATION_7_SPECS", "CANONICAL_6_OOD_SPLITS", "bootstrap_ci_1000"}:
        from ferroflux.training import evaluate_ood_and_ablations as _eval_mod

        return getattr(_eval_mod, name)
    if name in {"SyntheticCurriculumDataset", "load_training_dataset", "train_single_seed"}:
        from ferroflux.training import train_curriculum as _train_mod

        return getattr(_train_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
