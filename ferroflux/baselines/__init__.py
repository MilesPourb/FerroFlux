"""Reviewer #2 Multi-Omic, Chemical-Property, and Linear Perturbation Baselines."""

from __future__ import annotations

from typing import Any

__all__ = [
    "RidgeMultiOmicBaseline",
    "ElasticNetMultiOmicBaseline",
    "LightGBMMultiOmicBaseline",
    "XGBoostMultiOmicBaseline",
    "ChempropCellStateMLPBaseline",
    "LinearPerturbBaseline",
    "extract_multimodal_arrays",
    "build_synthetic_multimodal_arrays",
    "bootstrap_metric_ci",
    "evaluate_all_baselines_ood",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from ferroflux.baselines import tabular_and_chem_baselines as _mod

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
