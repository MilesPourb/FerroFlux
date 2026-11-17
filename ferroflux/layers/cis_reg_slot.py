"""Clean-Room Cis-Regulatory Epigenomic Prior Slot & Zero-Shot AlphaGenome 2 Adapter.

Pretraining / fine-tuning strictly uses Apache-2.0 / CC0 / CC-BY experimental chromatin,
DNA-methylation, and copy-number features (ENCODE-4, Roadmap Epigenomics, CCLE RRBS, TCGA ATAC)
at 64 curated hg38 ferroptosis loci.

During frozen `model.eval()` inference ONLY, `CisRegPriorSlot` optionally accepts zero-shot
`alphagenome==2.0.0` regulatory track or non-coding variant-effect predictions, downweighted by
observation-mode confidence (`observed=1.0 > imputed_supported=0.65 > imputed_exploratory=0.30`)
and gated by expressed trans-acting transcription factors (`NFE2L2`, `BACH1`, `TP53`, `ATF4`,
`YAP1`, `EPAS1`, `STAT1`, `ZEB1`). Any attempt to invoke AlphaGenome 2 inputs while
`self.training == True` immediately raises `LicenseViolationError(RuntimeError)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ferroflux.security.license_guard import assert_ag2_inference_only

CANONICAL_64_FERROPTOSIS_LOCI: tuple[str, ...] = (
    # Core peroxidase, selenocysteine & system xc- / GSH axis (14)
    "GPX4", "SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR",
    "PSTK", "SEPSECS", "SEPHS2", "EEFSEC", "SECISBP2", "LRP8", "PRDX6",
    # Parallel radical-trapping & redox shields (10)
    "AIFM2", "VKORC1L1", "DHODH", "GCH1", "PTS", "SPR", "TXNRD1", "TXN", "NQO1", "MGST1",
    # Phospholipid PUFA/MUFA & peroxisomal ether-lipid remodeling (12)
    "ACSL4", "LPCAT3", "ACSL3", "MBOAT1", "MBOAT2", "SCD", "FADS1", "FADS2",
    "FAR1", "AGPS", "GNPAT", "TMEM189",
    # Distal sterol / mevalonate biosynthesis & peroxidation enzymes (10)
    "DHCR7", "EBP", "SC5D", "SQLE", "HMGCR", "FDPS", "POR", "CYB5R1", "ALOX15", "ALOX12",
    # Labile iron, endolysosomal acidification & polyamine buffering (10)
    "TFRC", "FTH1", "FTL", "NCOA4", "SLC40A1", "STEAP3", "SLC11A2", "HMOX1", "ODC1", "SMS",
    # Master cis/trans regulators, Hippo/mechanosensing & competing RCD sensors (8)
    "NFE2L2", "KEAP1", "BACH1", "TP53", "YAP1", "NF2", "FDX1", "CD274",
)

TRANS_TF_GATE_GENES: tuple[str, ...] = (
    "NFE2L2", "BACH1", "TP53", "ATF4", "YAP1", "EPAS1", "STAT1", "ZEB1",
)

# Observation-mode confidence multipliers for AlphaGenome 2 tissue tracks
AG2_TIER_WEIGHTS: tuple[float, float, float] = (1.00, 0.65, 0.30)


class CisRegPriorSlot(nn.Module):
    """64-locus hg38 cis-regulatory prior token generator with AG2 ToS firewall."""

    def __init__(
        self,
        d_model: int = 384,
        n_loci: int = 64,
        in_features: int = 8,
        n_tfs: int = 8,
        ablate_cis_reg: bool = False,
        loci_tsv_path: str | Path | None = None,
        *,
        hidden_dim: int | None = None,
        epigenomic_dim: int | None = None,
        feat_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(hidden_dim if hidden_dim is not None else d_model)
        self.n_loci = int(n_loci)
        resolved_in = (
            epigenomic_dim
            if epigenomic_dim is not None
            else (feat_dim if feat_dim is not None else in_features)
        )
        self.in_features = int(resolved_in)
        self.epigenomic_dim = self.in_features
        self.feat_dim = self.in_features
        self.n_tfs = int(n_tfs)
        self.ablate_cis_reg = bool(ablate_cis_reg)

        self.loci_names: list[str] = self._load_loci_names(loci_tsv_path, self.n_loci)

        # Learnable locus identity embeddings (trained strictly on open experimental data)
        self.locus_embed = nn.Parameter(torch.randn(self.n_loci, self.d_model) * 0.02)

        # Clean-room experimental chromatin/methylation/CNV MLP projector
        self.clean_norm = nn.LayerNorm(self.in_features)
        self.clean_proj = nn.Sequential(
            nn.Linear(self.in_features, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        # Deterministic, non-distilled linear alignment buffer for zero-shot AG2 eval() tracks
        ag2_basis = torch.zeros(self.in_features, self.d_model)
        eye_dim = min(self.in_features, self.d_model)
        ag2_basis[:eye_dim, :eye_dim] = torch.eye(eye_dim) * 0.25
        self.register_buffer("ag2_frozen_basis", ag2_basis)
        self.register_buffer(
            "ag2_tier_table",
            torch.tensor(AG2_TIER_WEIGHTS, dtype=torch.float32),
        )

    @staticmethod
    def _load_loci_names(loci_tsv_path: str | Path | None, n_loci: int) -> list[str]:
        if loci_tsv_path is None:
            default_path = (
                Path(__file__).resolve().parent.parent.parent / "configs" / "cis_reg_64loci_hg38.tsv"
            )
            loci_tsv_path = default_path
        path = Path(loci_tsv_path)
        if path.is_file():
            try:
                lines = [
                    ln.strip()
                    for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.strip().startswith("#")
                ]
                if len(lines) > 1:
                    header = lines[0].split("\t")
                    gene_col = 0
                    for idx, col in enumerate(header):
                        if col.strip().lower() in ("gene", "gene_symbol", "locus", "symbol"):
                            gene_col = idx
                            break
                    parsed = [row.split("\t")[gene_col].strip() for row in lines[1:] if row.strip()]
                    if len(parsed) >= n_loci:
                        return parsed[:n_loci]
            except OSError:
                pass
        base = list(CANONICAL_64_FERROPTOSIS_LOCI)
        if len(base) >= n_loci:
            return base[:n_loci]
        return base + [f"LOCUS_{i}" for i in range(len(base), n_loci)]

    def _align_feature_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Align `[B, L, F]` or `[B, F]` experimental features to `[B, n_loci, in_features]`."""
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        if x.ndim == 2:
            x = x.unsqueeze(1).expand(-1, self.n_loci, -1)
        bsz, loci, feat = x.shape
        if loci < self.n_loci:
            pad_l = torch.zeros(bsz, self.n_loci - loci, feat, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_l], dim=1)
        elif loci > self.n_loci:
            x = x[:, : self.n_loci, :]

        if feat < self.in_features:
            pad_f = torch.zeros(bsz, self.n_loci, self.in_features - feat, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_f], dim=-1)
        elif feat > self.in_features:
            x = x[:, :, : self.in_features]
        return x

    def encode_ag2_zero_shot(
        self,
        ag2_tracks: torch.Tensor,
        *,
        ag2_variant_scores: torch.Tensor | None = None,
        ag2_imputation_tier: torch.Tensor | None = None,
        tf_expr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Zero-shot `eval()`-only projection of AlphaGenome 2 regulatory tracks & variant deltas."""
        assert_ag2_inference_only(is_training=self.training, context="CisRegPriorSlot.encode_ag2_zero_shot")
        with torch.no_grad():
            aligned = self._align_feature_shape(ag2_tracks)
            if ag2_variant_scores is not None:
                aligned = aligned + 0.5 * self._align_feature_shape(ag2_variant_scores)

            # Observation-mode reliability scaling (observed=1.0, imputed_supported=0.65, exploratory=0.30)
            if ag2_imputation_tier is not None:
                tiers = ag2_imputation_tier.long().clamp(0, 2)
                weights = self.ag2_tier_table.to(device=aligned.device, dtype=aligned.dtype)[tiers]
                while weights.ndim < aligned.ndim:
                    weights = weights.unsqueeze(-1)
                aligned = aligned * weights

            # Gate sequence-only cis predictions by expressed trans-acting TFs
            if tf_expr is not None:
                tf_clean = torch.nan_to_num(tf_expr, nan=0.0).to(device=aligned.device, dtype=aligned.dtype)
                tf_gate = torch.sigmoid(tf_clean.mean(dim=-1, keepdim=True)).unsqueeze(-1)
                aligned = aligned * tf_gate

            basis = self.ag2_frozen_basis.to(device=aligned.device, dtype=aligned.dtype)
            return torch.matmul(aligned, basis)

    def forward(
        self,
        epigenomic_features: torch.Tensor | None = None,
        *,
        use_ag2: bool = False,
        ag2_tracks: torch.Tensor | None = None,
        ag2_variant_scores: torch.Tensor | None = None,
        ag2_imputation_tier: torch.Tensor | None = None,
        tf_expr: torch.Tensor | None = None,
        source: str = "experimental",
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Produce `[B, n_loci, d_model]` cis-regulatory prior tokens."""
        ag2_requested = (
            bool(use_ag2)
            or (ag2_tracks is not None)
            or (ag2_variant_scores is not None)
            or (str(source).strip().lower() in {"alphagenome2", "alphagenome", "ag2"})
        )
        if ag2_requested:
            assert_ag2_inference_only(is_training=self.training, context="CisRegPriorSlot.forward")

        target_device = (
            epigenomic_features.device
            if epigenomic_features is not None
            else (ag2_tracks.device if ag2_tracks is not None else (device or self.locus_embed.device))
        )
        target_dtype = (
            epigenomic_features.dtype
            if (epigenomic_features is not None and epigenomic_features.is_floating_point())
            else (dtype or self.locus_embed.dtype)
        )
        bsz = (
            epigenomic_features.shape[0]
            if epigenomic_features is not None
            else (ag2_tracks.shape[0] if ag2_tracks is not None else int(batch_size))
        )

        if self.ablate_cis_reg:
            return torch.zeros(bsz, self.n_loci, self.d_model, device=target_device, dtype=target_dtype)

        tokens = self.locus_embed.to(device=target_device, dtype=target_dtype).unsqueeze(0).expand(bsz, -1, -1)

        if epigenomic_features is not None and not (
            ag2_requested and ag2_tracks is None and str(source).strip().lower() != "experimental"
        ):
            clean_x = self._align_feature_shape(epigenomic_features.to(device=target_device, dtype=target_dtype))
            tokens = tokens + self.clean_proj(self.clean_norm(clean_x))

        if ag2_requested:
            raw_ag2 = ag2_tracks if ag2_tracks is not None else epigenomic_features
            if raw_ag2 is not None:
                ag2_delta = self.encode_ag2_zero_shot(
                    raw_ag2.to(device=target_device, dtype=target_dtype),
                    ag2_variant_scores=ag2_variant_scores,
                    ag2_imputation_tier=ag2_imputation_tier,
                    tf_expr=tf_expr,
                )
                tokens = tokens + ag2_delta

        return tokens


CisRegulatoryPriorSlot = CisRegPriorSlot

__all__ = [
    "CANONICAL_64_FERROPTOSIS_LOCI",
    "TRANS_TF_GATE_GENES",
    "CisRegPriorSlot",
    "CisRegulatoryPriorSlot",
]
