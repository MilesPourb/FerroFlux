"""Data ingestion, harmonization, and multimodal PyTorch Dataset utilities for FerroFlux-FM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_PROCESSED_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML/data/processed")
DEFAULT_GENE_VOCAB_PATH = DEFAULT_PROCESSED_DIR / "gene_vocab.json"

__all__ = [
    "DEFAULT_PROCESSED_DIR",
    "DEFAULT_GENE_VOCAB_PATH",
    "FerroFluxMultimodalDataset",
    "load_gene_vocab",
    "scaffold_split",
    "lineage_split",
    "organoid_2d_to_3d_split",
    "crispr_fin_epistasis_split",
]


def __getattr__(name: str) -> Any:
    if name in {
        "FerroFluxMultimodalDataset",
        "load_gene_vocab",
        "scaffold_split",
        "lineage_split",
        "organoid_2d_to_3d_split",
        "crispr_fin_epistasis_split",
    }:
        from ferroflux.data import dataset as _dataset_mod

        return getattr(_dataset_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
