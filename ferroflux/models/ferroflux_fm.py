r"""FerroFlux-FM v3: Leak-Free Grey-Box Neural-Biophysical Foundation Model for Ferroptosis.

Key Upgrades in v3 over v2:
- Multi-scale pure-SMILES `CompositionalPerturbationOperator` (`3,239`-bit ECFP4/ECFP6/MACCS +
  `24` RDKit/SMARTS descriptors) pretrained in Stage 0.5 on non-overlapping biochemical SAR + CysDB ABPP.
- Rich multi-omic basal cell context (`138`-dim CNV/LOF/hotspot genomics + `64` MS proteins + `32` LC-MS
  lipid/redox species; zeroed cleanly when `ablate_physical_anchors=True`).
- APHYNITY norm-penalized multiplicative grey-box residual (`gem_out["aphynity_gain"]`) inside `DiffFerroGEM`,
  preserving 100% causal rescue monotonicity (`Lip-1`, `DFO`, `ConA`), opposing `EBP`/`DHCR7` sterol sign
  laws, C'-dot iron scavenging, cuproptosis invariance, and zero-bypass identifiability around `DiffFerroGEM`.
- Rank-48 gene-by-cell bilinear dependency head + exact 4-corner analytical population-genetics counterfactual
  epistasis ($-\frac{1}{\ln 2}\Delta_{\text{KO}}\Delta_{\text{Drug}}\log(1 + H)$) for zero-shot `CRISPR + FIN` screens.
- Pathway-block (`32`-gene) + `15%` random Masked Gene Modeling for cross-pathway transcriptomic coupling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ferroflux.layers.cis_reg_slot import CANONICAL_64_FERROPTOSIS_LOCI, CisRegPriorSlot
from ferroflux.layers.diff_ferro_gem import EXTRA_CELL_CTX_DIM, DiffFerroGEM
from ferroflux.layers.perceiver_trunk import (
    ANCHOR_12_METABOLITES,
    ANCHOR_28_PROTEINS,
    METABOLITE_TO_IDX,
    PROTEIN_TO_IDX,
    GeneToModulePerceiverTrunk,
)
from ferroflux.layers.perturbation_ops import (
    CHEM_14_TARGETS,
    MEDIA_DEFAULTS,
    TARGET_TO_IDX,
    CompositionalPerturbationOperator,
)
from ferroflux.layers.sec46_gate import IPP_MEVALONATE_GENES, SEC_MACHINERY_GENES, Sec46Gate

PRIMARY_5_HAZARDS: tuple[str, ...] = (
    "H_ferro",
    "H_apop",
    "H_necro",
    "H_cupro",
    "H_disulf",
)

IMMUNOGENIC_6_PATHWAYS: tuple[str, ...] = (
    "cGAS_STING",
    "TLR_ferroptosis",
    "IFN_response",
    "CD274_PDL1",
    "sterol_biosynthesis_suppression",
    "M1_vs_M2_TAM",
)

_CANONICAL_PATHWAY_GENES: dict[str, tuple[str, ...]] = {
    "cgas_sting": ("CGAS", "MB21D1", "STING1", "TMEM173", "TBK1", "IRF3", "CXCL10", "CCL5", "IFNB1", "ISG15"),
    "pdl1_ifn": ("CD274", "PDCD1LG2", "STAT1", "IRF1", "CXCL9", "IDO1"),
    "tlr_ferro": ("TLR2", "TLR7", "TLR8", "MYD88", "ACSL4", "TFRC", "HMOX1", "PTGS2", "CHAC1", "NCOA4"),
    "m1_tam": ("NOS2", "CD86", "CD80", "TNF", "IL1B"),
    "m2_tam": ("MRC1", "CD163", "ARG1", "IL10", "TGFB1"),
    "sterol_up": ("EBP", "SC5D"),
    "sterol_down": ("DHCR7",),
    "sterol_7dhc_mask": ("EBP", "SC5D", "DHCR7"),
    "sterol_all": ("EBP", "SC5D", "DHCR7", "SQLE", "HMGCR", "CYP51A1", "MSMO1", "FDFT1"),
    "cupro_tca": ("FDX1", "LIAS", "LIPT1", "DLD", "DLAT", "PDHA1", "PDHB", "ACO2", "SDHB", "IDH2"),
    "anchor_28": ANCHOR_28_PROTEINS,
    "sec46_full": SEC_MACHINERY_GENES + IPP_MEVALONATE_GENES + ("GPX4",),
    "cis_64": CANONICAL_64_FERROPTOSIS_LOCI,
}


def _resolve_pathway_idx_map(n_genes: int) -> dict[str, list[int]]:
    vocab_p = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "gene_vocab.json"
    g2i: dict[str, int] = {}
    if vocab_p.is_file():
        try:
            raw = json.loads(vocab_p.read_text(encoding="utf-8"))
            g2i = {str(k).upper(): int(v) for k, v in raw.get("gene_to_idx", {}).items()}
            for al, cn in raw.get("alias_to_canonical", {}).items():
                if str(cn).upper() in g2i:
                    g2i[str(al).upper()] = g2i[str(cn).upper()]
        except Exception:
            g2i = {}
    return {
        k: sorted({g2i[g.upper()] for g in syms if g.upper() in g2i and 0 <= g2i[g.upper()] < n_genes})
        for k, syms in _CANONICAL_PATHWAY_GENES.items()
    }


class FerroFluxFM(nn.Module):
    """End-to-end grey-box neural-biophysical ferroptosis foundation model (v3)."""

    def __init__(
        self,
        n_genes: int = 512,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        n_modules: int = 128,
        chem_fp_dim: int = 3239,
        chem_desc_dim: int = 24,
        epigenomic_dim: int = 8,
        variant: str | None = None,
        gem_json_path: str | Path | None = None,
        loci_tsv_path: str | Path | None = None,
        ablate_diff_ferro_gem: bool = False,
        ablate_sec46_gate: bool = False,
        ablate_survivorship: bool = False,
        ablate_chem_prior: bool = False,
        ablate_physical_anchors: bool = False,
        ablate_cis_reg: bool = False,
        ablate_cis_reg_slot: bool = False,
        ablate_aphynity_residual: bool = False,
        ablate_sar_pretrain: bool = False,
    ) -> None:
        super().__init__()
        if variant is not None:
            v = variant.strip().lower()
            if v == "v3":
                d_model, n_layers, n_heads, n_modules = 256, 4, 4, 96
            elif v == "base":
                d_model, n_layers, n_heads = 384, 8, 6
            elif v == "xl":
                d_model, n_layers, n_heads = 1024, 16, 16

        self.n_genes = int(n_genes)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_modules = int(n_modules)
        self.chem_fp_dim = int(chem_fp_dim)
        self.chem_desc_dim = int(chem_desc_dim)
        self.epigenomic_dim = int(epigenomic_dim)

        self.ablate_diff_ferro_gem = bool(ablate_diff_ferro_gem)
        self.ablate_sec46_gate = bool(ablate_sec46_gate)
        self.ablate_survivorship = bool(ablate_survivorship)
        self.ablate_chem_prior = bool(ablate_chem_prior)
        self.ablate_physical_anchors = bool(ablate_physical_anchors)
        self.ablate_cis_reg = bool(ablate_cis_reg or ablate_cis_reg_slot)
        self.ablate_cis_reg_slot = self.ablate_cis_reg
        self.ablate_aphynity_residual = bool(ablate_aphynity_residual)
        self.ablate_sar_pretrain = bool(ablate_sar_pretrain)

        d_chem_lat = min(256, max(128, self.d_model // 2 + 64))

        # 1. Clean-room cis-regulatory prior slot + zero-shot eval() AG2 adapter
        self.cis_reg_slot = CisRegPriorSlot(
            d_model=self.d_model,
            n_loci=64,
            in_features=self.epigenomic_dim,
            ablate_cis_reg=self.ablate_cis_reg,
            loci_tsv_path=loci_tsv_path,
        )

        # 2. Gene-to-module Perceiver + RoPE-SwiGLU trunk + 28 protein & 12 metabolite anchors
        self.perceiver_trunk = GeneToModulePerceiverTrunk(
            n_genes=self.n_genes,
            d_model=self.d_model,
            n_modules=self.n_modules,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ablate_physical_anchors=self.ablate_physical_anchors,
        )

        # Multi-omic context fusion (138 genomic CNV/LOF/hotspots + 64 MS proteome + 32 LC-MS lipids)
        self.multiomic_ctx_fusion = nn.Sequential(
            nn.Linear(EXTRA_CELL_CTX_DIM, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        nn.init.zeros_(self.multiomic_ctx_fusion[-1].weight)
        nn.init.zeros_(self.multiomic_ctx_fusion[-1].bias)

        # Row-centered observed transcriptome projection (`[B, n_genes] -> [B, 128]`, respects all masks!)
        self.expr_ctx_proj = nn.Sequential(
            nn.Linear(self.n_genes, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.08),
        )
        self.mgm_ctx_decoder = nn.Linear(self.d_model + 128, self.n_genes)
        nn.init.zeros_(self.mgm_ctx_decoder.weight)
        nn.init.zeros_(self.mgm_ctx_decoder.bias)
        self._pathway_idx_map: dict[str, list[int]] = _resolve_pathway_idx_map(self.n_genes)

        # 3. Sec46 translational readthrough + mevalonate IPP + covalent warhead gate
        self.sec46_gate = Sec46Gate(
            in_features=6,
            ablate_sec46_gate=self.ablate_sec46_gate,
        )

        # 4. Compositional small-molecule, CRISPR, media & physical NP operator
        self.perturbation_op = CompositionalPerturbationOperator(
            chem_fp_dim=self.chem_fp_dim,
            chem_desc_dim=self.chem_desc_dim,
            d_latent=d_chem_lat,
            d_cell=self.d_model,
            ablate_chem_prior=self.ablate_chem_prior,
        )

        # 5. Differentiable stoichiometric-kinetic + APHYNITY grey-box ferroptosis layer
        self.diff_ferro_gem = DiffFerroGEM(
            d_model=self.d_model,
            n_reactions=94,
            gem_json_path=gem_json_path,
            n_ode_substeps=12,
            d_chem_latent=d_chem_lat,
            n_genes=self.n_genes,
            ablate_aphynity_residual=self.ablate_aphynity_residual,
        )
        self.perturbation_op.ablate_sec46_gate = self.ablate_sec46_gate
        self.diff_ferro_gem.ablate_sec46_gate = self.ablate_sec46_gate
        self.diff_ferro_gem.ablate_physical_anchors = self.ablate_physical_anchors
        self.diff_ferro_gem.ablate_cis_reg = self.ablate_cis_reg
        self.diff_ferro_gem._anchor_gene_indices = tuple(self._pathway_idx_map.get("anchor_28", []))
        self.diff_ferro_gem._sec46_gene_indices = tuple(self._pathway_idx_map.get("sec46_full", []))
        self.diff_ferro_gem._cis_locus_gene_indices = tuple(self._pathway_idx_map.get("cis_64", []))

        # Biophysical link scalars mapping DiffFerroGEM ([PL-OOH], g_rupture, aphynity_gain) -> H_ferro
        self.raw_w_h_plooh = nn.Parameter(torch.tensor(0.85))
        self.raw_w_h_pore = nn.Parameter(torch.tensor(1.35))
        self.raw_k_pore = nn.Parameter(torch.tensor(-0.35))
        self.raw_w_h_rup = nn.Parameter(torch.tensor(0.55))

        pert_summary_dim = len(CHEM_14_TARGETS) + len(ANCHOR_28_PROTEINS) + len(MEDIA_DEFAULTS) + 4
        self.ablation_blackbox_ferro_head = nn.Sequential(
            nn.Linear(self.d_model + pert_summary_dim + d_chem_lat, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, 1),
        )

        # Competing-risks RCD heads (strictly invariant to Lip-1 & DFO!)
        self.apop_cell_head = nn.Linear(self.d_model, 1)
        self.apop_intrinsic_drug_head = nn.Linear(128, 1)
        nn.init.zeros_(self.apop_intrinsic_drug_head.weight)
        nn.init.constant_(self.apop_intrinsic_drug_head.bias, -2.2)
        self.apop_cell_proj = nn.Sequential(
            nn.Linear(self.d_model + 128 + EXTRA_CELL_CTX_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.apop_chem_proj = nn.Sequential(
            nn.Linear(d_chem_lat, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.apop_bilinear_head = nn.Sequential(
            nn.Linear(128 * 3, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )
        nn.init.zeros_(self.apop_bilinear_head[-1].weight)
        nn.init.zeros_(self.apop_bilinear_head[-1].bias)

        self.necro_cell_head = nn.Linear(self.d_model, 1)
        self.necro_bilinear_head = nn.Sequential(
            nn.Linear(self.d_model + d_chem_lat, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.necro_bilinear_head[-1].weight)
        nn.init.zeros_(self.necro_bilinear_head[-1].bias)
        self.cupro_cell_head = nn.Linear(self.d_model + 128 + 16, 1)
        self.cupro_ctx_proj = nn.Linear(EXTRA_CELL_CTX_DIM, 16)
        self.disulf_cell_head = nn.Linear(self.d_model, 1)
        self.raw_k_txnrd1_tox = nn.Parameter(torch.tensor(0.45))
        self.other_17_rcd_head = nn.Linear(self.d_model, 17)

        # Readout Head 2: Rank-64 Gene x Cell Bilinear KO Dependency + Analytical Epistasis Head
        g_emb_d = self.perceiver_trunk.gene_emb_dim
        self.gene_rank_d = 64
        self.gene_essentiality_prior = nn.Parameter(torch.zeros(self.n_genes))
        self.gene_expr_slope_prior = nn.Parameter(torch.zeros(self.n_genes))
        self.gene_ctx_emb = nn.Parameter(torch.randn(self.n_genes, self.gene_rank_d) * 0.04)
        self.cell_chronos_offset = nn.Linear(128, 1)
        nn.init.zeros_(self.cell_chronos_offset.weight)
        nn.init.zeros_(self.cell_chronos_offset.bias)
        cell_ko_in_dim = self.d_model + len(ANCHOR_28_PROTEINS) + len(ANCHOR_12_METABOLITES) + EXTRA_CELL_CTX_DIM + 128
        self.cell_to_gene_ctx = nn.Sequential(
            nn.Linear(cell_ko_in_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.gene_rank_d),
        )

        self.pert_to_lat = nn.Sequential(
            nn.Linear(self.d_model + pert_summary_dim + g_emb_d + 4, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, g_emb_d),
        )
        chronos_in_dim = (
            self.d_model
            + len(ANCHOR_28_PROTEINS)
            + g_emb_d
            + self.gene_rank_d
            + len(CHEM_14_TARGETS)
            + 4
            + 6
        )
        self.chronos_head = nn.Sequential(
            nn.Linear(chronos_in_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, 1),
        )
        nn.init.zeros_(self.chronos_head[-1].weight)
        nn.init.zeros_(self.chronos_head[-1].bias)
        self.raw_epistasis_gain = nn.Parameter(torch.tensor(0.85))

        self.synergy_head = nn.Linear(self.d_model + 4, 1)
        self.immunogenic_head = nn.Sequential(
            nn.Linear(self.d_model + 128 + 7, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, len(IMMUNOGENIC_6_PATHWAYS) + 1),
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        variant: str | None = None,
        n_genes: int = 2048,
        **overrides: Any,
    ) -> "FerroFluxFM":
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "train_config.yaml"
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        var_name = variant or cfg.get("curriculum", {}).get("active_variant", "v3")
        v_cfg = cfg.get("model_variants", {}).get(var_name, {})
        if not v_cfg and var_name == "v3":
            v_cfg = {"d_model": 384, "n_layers": 6, "n_heads": 6, "n_modules": 128}
        ab_cfg = cfg.get("ablations", {})
        kw: dict[str, Any] = {
            "n_genes": n_genes,
            "d_model": v_cfg.get("d_model", 256 if var_name == "v3" else 384),
            "n_layers": v_cfg.get("n_layers", 4 if var_name == "v3" else 8),
            "n_heads": v_cfg.get("n_heads", 4 if var_name == "v3" else 6),
            "n_modules": v_cfg.get("n_modules", 96 if var_name == "v3" else 128),
            "chem_fp_dim": v_cfg.get("chem_fp_dim", 3239),
            "chem_desc_dim": v_cfg.get("chem_desc_dim", 24),
            "epigenomic_dim": v_cfg.get("epigenomic_dim", 8),
            "gem_json_path": cfg.get("stoichiometry_and_loci", {}).get("ferro_gem_json"),
            "loci_tsv_path": cfg.get("stoichiometry_and_loci", {}).get("cis_reg_loci_tsv"),
            "ablate_diff_ferro_gem": ab_cfg.get("ablate_diff_ferro_gem", False),
            "ablate_sec46_gate": ab_cfg.get("ablate_sec46_gate", False),
            "ablate_survivorship": ab_cfg.get("ablate_survivorship", False),
            "ablate_chem_prior": ab_cfg.get("ablate_chem_prior", False),
            "ablate_physical_anchors": ab_cfg.get("ablate_physical_anchors", False),
            "ablate_cis_reg_slot": ab_cfg.get("ablate_cis_reg_slot", False),
            "ablate_aphynity_residual": ab_cfg.get("ablate_aphynity_residual", False),
            "ablate_sar_pretrain": ab_cfg.get("ablate_sar_pretrain", False),
        }
        kw.update(overrides)
        return cls(**kw)

    @staticmethod
    def _align_ctx_field(batch: dict[str, Any], key: str, mask_key: str | None, bsz: int, dim: int, dev: torch.device, dtp: torch.dtype) -> torch.Tensor:
        raw = batch.get(key)
        if not isinstance(raw, torch.Tensor):
            return torch.zeros(bsz, dim, device=dev, dtype=dtp)
        x = torch.nan_to_num(raw.to(device=dev, dtype=dtp), nan=0.0, posinf=10.0, neginf=-10.0)
        if x.ndim == 1:
            x = x.unsqueeze(0).expand(bsz, -1)
        if x.shape[-1] < dim:
            x = torch.cat([x, torch.zeros(bsz, dim - x.shape[-1], device=dev, dtype=dtp)], dim=-1)
        elif x.shape[-1] > dim:
            x = x[:, :dim]
        if mask_key and isinstance(batch.get(mask_key), torch.Tensor):
            m = torch.nan_to_num(batch[mask_key].to(device=dev, dtype=dtp), nan=0.0).clamp(0.0, 1.0)
            if m.ndim == 1:
                m = m.unsqueeze(0).expand(bsz, -1)
            if m.shape[-1] == dim:
                x = x * m
        return x

    def _compute_sec46_and_gem(
        self,
        z_cell: torch.Tensor,
        p_hat: torch.Tensor,
        m0_hat: torch.Tensor,
        pert_dict: dict[str, torch.Tensor],
        extra_cell_ctx: torch.Tensor,
        gene_expr_ctx: torch.Tensor,
        dtype: torch.dtype,
        p_hat_log: torch.Tensor | None = None,
        m0_hat_log: torch.Tensor | None = None,
        gene_expr_raw: torch.Tensor | None = None,
        gene_expr_mask: torch.Tensor | None = None,
        cis_locus_ctx: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
        target_occ = pert_dict["target_occ"]
        gene_mult = pert_dict["gene_mult"]
        media_vec = pert_dict["media_vec"]

        p_eff_log = torch.log((p_hat * gene_mult.clamp(min=1e-4)).clamp(min=1e-5))
        gpx4_log = p_eff_log[:, PROTEIN_TO_IDX["GPX4"]]
        sec_machinery_log = torch.stack(
            [
                p_eff_log[:, PROTEIN_TO_IDX["PSTK"]],
                p_eff_log[:, PROTEIN_TO_IDX["SEPSECS"]],
                p_eff_log[:, PROTEIN_TO_IDX["SEPHS2"]],
                p_eff_log[:, PROTEIN_TO_IDX["EEFSEC"]],
            ],
            dim=-1,
        )
        ipp_log = (
            0.5 * (p_eff_log[:, PROTEIN_TO_IDX["HMGCR"]] + p_eff_log[:, PROTEIN_TO_IDX["SQLE"]])
            - 3.2 * target_occ[:, TARGET_TO_IDX["HMGCR"]]
        )
        sec_out = self.sec46_gate.compute_active_gpx4(
            gpx4_expr=gpx4_log,
            sec_machinery=sec_machinery_log,
            ipp_level=ipp_log,
            se_media=media_vec[:, 3],
            gpx4_covalent_occ=target_occ[:, TARGET_TO_IDX["GPX4_cov"]],
            gpx4_degrader_occ=target_occ[:, TARGET_TO_IDX["GPX4_deg"]],
        )
        gem_out = self.diff_ferro_gem(
            z_cell,
            p_hat=p_hat,
            m0_hat=m0_hat,
            p_hat_log=p_hat_log,
            m0_hat_log=m0_hat_log,
            e_gpx4_active=sec_out["e_gpx4_active"],
            sec_readthrough_eff=sec_out["sec_readthrough_eff"],
            ipp_trna_mod_eff=sec_out["ipp_trna_mod_eff"],
            pert_effects=pert_dict,
            extra_cell_ctx=extra_cell_ctx,
            gene_expr_ctx=gene_expr_ctx,
            gene_expr_raw=gene_expr_raw,
            gene_expr_mask=gene_expr_mask,
            cis_locus_ctx=cis_locus_ctx,
        )
        pl_ooh = gem_out["pl_ooh"]
        g_rupture = gem_out["g_rupture"]
        aphynity_gain = gem_out["aphynity_gain"]

        w_plooh = F.softplus(self.raw_w_h_plooh).to(dtype=dtype) + 0.15
        w_pore = F.softplus(self.raw_w_h_pore).to(dtype=dtype) + 0.25
        k_pore = F.softplus(self.raw_k_pore).to(dtype=dtype) + 0.15
        w_rup = F.softplus(self.raw_w_h_rup).to(dtype=dtype) + 0.10
        pl_sq = pl_ooh.pow(2)
        pore_hill = pl_sq / (k_pore.pow(2) + pl_sq)
        h_ferro_phys = (w_plooh * pl_ooh + w_pore * pore_hill * (1.0 + w_rup * g_rupture)).clamp(min=1e-7)
        gem_out["h_ferro_phys"] = h_ferro_phys
        gb_w = gem_out["aphynity_gb_weight"]
        h_ferro = ((1.0 - gb_w) * (h_ferro_phys * aphynity_gain) + gb_w * gem_out["aphynity_add_haz"]).clamp(min=1e-7)
        return sec_out, gem_out, h_ferro

    def forward(
        self,
        batch_or_expr: dict[str, Any] | torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run full neural-biophysical forward pass from batch dict or `gene_expr` tensor."""
        if isinstance(batch_or_expr, torch.Tensor):
            batch: dict[str, Any] = {"gene_expr": batch_or_expr}
            batch.update(kwargs)
        else:
            batch = dict(batch_or_expr)
            batch.update(kwargs)

        gene_expr = batch["gene_expr"]
        if gene_expr.ndim == 1:
            gene_expr = gene_expr.unsqueeze(0)
        bsz, g_in = gene_expr.shape
        device = gene_expr.device
        dtype = gene_expr.dtype

        expr_mask = next(
            (batch[k] for k in ("gene_mask", "expr_mask") if k in batch and isinstance(batch[k], torch.Tensor)),
            None,
        )

        # Stage-1 Canonical Pathway-Module + `15%` Random Masked Gene Modeling during training ONLY
        mgm_keep_mask: torch.Tensor | None = None
        mgm_eval_mask: torch.Tensor | None = None
        gt_imm_target: torch.Tensor | None = None
        gt_imm_mask: torch.Tensor | None = None
        if self.training:
            x_full_tr, obs_full_tr = self.perceiver_trunk._align_genes(gene_expr, expr_mask)
            gt_cols: list[torch.Tensor] = []
            gt_mcols: list[torch.Tensor] = []
            pmap = self._pathway_idx_map
            for p_spec in ("cgas_sting", "tlr_ferro", "pdl1_ifn", "pdl1_ifn", "sterol", "tam"):
                if p_spec == "sterol":
                    u_i, d_i = pmap.get("sterol_up", []), pmap.get("sterol_down", [])
                    if u_i and d_i:
                        mu = obs_full_tr[:, u_i].min(dim=-1).values * obs_full_tr[:, d_i].min(dim=-1).values
                        val = x_full_tr[:, u_i].mean(dim=-1) - x_full_tr[:, d_i].mean(dim=-1)
                        gt_cols.append(val)
                        gt_mcols.append((mu > 0.5).to(dtype=dtype))
                    else:
                        gt_cols.append(torch.zeros(bsz, device=device, dtype=dtype))
                        gt_mcols.append(torch.zeros(bsz, device=device, dtype=dtype))
                elif p_spec == "tam":
                    m1_i, m2_i = pmap.get("m1_tam", []), pmap.get("m2_tam", [])
                    if m1_i and m2_i:
                        mu = (obs_full_tr[:, m1_i].sum(dim=-1) > 0) & (obs_full_tr[:, m2_i].sum(dim=-1) > 0)
                        val = x_full_tr[:, m1_i].mean(dim=-1) - x_full_tr[:, m2_i].mean(dim=-1)
                        gt_cols.append(val)
                        gt_mcols.append(mu.to(dtype=dtype))
                    else:
                        gt_cols.append(torch.zeros(bsz, device=device, dtype=dtype))
                        gt_mcols.append(torch.zeros(bsz, device=device, dtype=dtype))
                else:
                    idx_l = pmap.get(p_spec, [])
                    if idx_l:
                        cnt_l = obs_full_tr[:, idx_l].sum(dim=-1)
                        val = (x_full_tr[:, idx_l] * obs_full_tr[:, idx_l]).sum(dim=-1) / cnt_l.clamp(min=1.0)
                        gt_cols.append(val)
                        gt_mcols.append((cnt_l >= max(1, len(idx_l) // 2)).to(dtype=dtype))
                    else:
                        gt_cols.append(torch.zeros(bsz, device=device, dtype=dtype))
                        gt_mcols.append(torch.zeros(bsz, device=device, dtype=dtype))
            gt_imm_target = torch.stack(gt_cols, dim=-1)
            gt_imm_mask = torch.stack(gt_mcols, dim=-1)

        if self.training and bool(batch.get("enable_mgm_masking", True)):
            obs_m = (
                torch.nan_to_num(expr_mask.to(device=device, dtype=dtype), nan=1.0).clamp(0.0, 1.0)
                if expr_mask is not None
                else torch.ones(bsz, g_in, device=device, dtype=dtype)
            )
            rand_u = torch.rand(bsz, g_in, device=device, dtype=dtype)
            drop_mask = (rand_u < 0.15).to(dtype=dtype)
            if g_in >= 64:
                blk_len = min(32, g_in // 4)
                starts = torch.randint(0, max(1, g_in - blk_len), (bsz, 1), device=device)
                pos_grid = torch.arange(g_in, device=device).unsqueeze(0)
                in_blk = ((pos_grid >= starts) & (pos_grid < starts + blk_len)).to(dtype=dtype)
                use_blk = (torch.rand(bsz, 1, device=device, dtype=dtype) < 0.35).to(dtype=dtype)
                drop_mask = torch.clamp(drop_mask + in_blk * use_blk, 0.0, 1.0)

                # Functional pathway-module masking (`cGAS-STING + PD-L1/IFN` or `Distal 7-DHC Sterol`)
                sting_pdl1_idx = [i for i in (self._pathway_idx_map.get("cgas_sting", []) + self._pathway_idx_map.get("pdl1_ifn", [])) if i < g_in]
                sterol_7dhc_idx = [i for i in self._pathway_idx_map.get("sterol_7dhc_mask", []) if i < g_in]
                draw_pw = torch.rand(bsz, device=device)
                if sting_pdl1_idx:
                    m_sp = (draw_pw < 0.30).to(dtype=dtype).unsqueeze(-1)
                    sp_mask = torch.zeros(1, g_in, device=device, dtype=dtype)
                    sp_mask[0, sting_pdl1_idx] = 1.0
                    drop_mask = torch.clamp(drop_mask + m_sp * sp_mask, 0.0, 1.0)
                if sterol_7dhc_idx:
                    m_st = ((draw_pw >= 0.30) & (draw_pw < 0.55)).to(dtype=dtype).unsqueeze(-1)
                    st_mask = torch.zeros(1, g_in, device=device, dtype=dtype)
                    st_mask[0, sterol_7dhc_idx] = 1.0
                    drop_mask = torch.clamp(drop_mask + m_st * st_mask, 0.0, 1.0)

            drop_mask = drop_mask * obs_m
            has_enough = (obs_m.sum(dim=-1, keepdim=True) >= 12).to(dtype=dtype)
            drop_mask = drop_mask * has_enough
            mgm_keep_mask = 1.0 - drop_mask
            mgm_eval_mask = torch.where(drop_mask.sum(dim=-1, keepdim=True) > 0, drop_mask, obs_m)

        # 1. Cis-regulatory prior tokens (strictly enforces AG2 ToS lock in self.cis_reg_slot)
        epigen = next(
            (batch[k] for k in ("cis_reg_epigen", "epigenomics", "epigen_features") if k in batch and isinstance(batch[k], torch.Tensor)),
            None,
        )
        cis_mask = batch.get("cis_reg_mask") if isinstance(batch.get("cis_reg_mask"), torch.Tensor) else None
        if epigen is not None and cis_mask is not None:
            m_epi = torch.nan_to_num(cis_mask.to(device=device, dtype=dtype), nan=0.0)
            while m_epi.ndim < epigen.ndim:
                m_epi = m_epi.unsqueeze(-1)
            epigen = epigen.to(device=device, dtype=dtype) * m_epi

        cis_tokens = self.cis_reg_slot(
            epigen,
            use_ag2=bool(batch.get("use_ag2", False)),
            ag2_tracks=batch.get("ag2_tracks"),
            ag2_variant_scores=batch.get("ag2_variant_scores"),
            ag2_imputation_tier=batch.get("ag2_imputation_tier"),
            tf_expr=batch.get("tf_expr"),
            source=str(batch.get("epigen_source", "experimental")),
            batch_size=bsz,
            device=device,
            dtype=dtype,
        )

        raw_gko = batch.get("gene_ko_mult") if isinstance(batch.get("gene_ko_mult"), torch.Tensor) else None

        # Extract optional basal genomic/proteomic/lipidomic context (zeroed when ablate_physical_anchors=True)
        if self.ablate_physical_anchors:
            extra_cell_ctx = torch.zeros(bsz, EXTRA_CELL_CTX_DIM, device=device, dtype=dtype)
        else:
            genomic_138 = self._align_ctx_field(batch, "genomic_context", None, bsz, 138, device, dtype)
            prot_64 = self._align_ctx_field(batch, "prot64_context", "prot64_mask", bsz, 64, device, dtype)
            lipid_32 = self._align_ctx_field(batch, "lipid32_context", "lipid32_mask", bsz, 32, device, dtype)
            extra_cell_ctx = torch.cat([genomic_138, prot_64, lipid_32], dim=-1)
            if self.training:
                extra_cell_ctx = F.dropout(extra_cell_ctx, p=0.12, training=True)

        # 2. Perceiver module trunk + 28 protein & 12 metabolite physical anchors + observed-only expr context
        x_aligned, obs_mask_aligned = self.perceiver_trunk._align_genes(gene_expr, expr_mask)
        if mgm_keep_mask is not None:
            km_al, _ = self.perceiver_trunk._align_genes(mgm_keep_mask)
            eff_obs_m = obs_mask_aligned * km_al.clamp(0.0, 1.0)
        else:
            eff_obs_m = obs_mask_aligned
        obs_cnt = eff_obs_m.sum(dim=-1, keepdim=True).clamp(min=1.0)
        x_obs_mean = (x_aligned * eff_obs_m).sum(dim=-1, keepdim=True) / obs_cnt
        x_obs_centered = (x_aligned - x_obs_mean) * eff_obs_m
        gene_expr_ctx = self.expr_ctx_proj(x_obs_centered)

        trunk_out = self.perceiver_trunk(
            gene_expr,
            expr_mask=expr_mask,
            cis_reg_tokens=cis_tokens,
            mgm_keep_mask=mgm_keep_mask,
        )
        z_cell = trunk_out["z_cell"] + self.multiomic_ctx_fusion(extra_cell_ctx) + 0.25 * cis_tokens.mean(dim=1)
        p_hat = trunk_out["p_hat"]
        m0_hat = trunk_out["m0_hat"]
        p_hat_log = trunk_out["p_hat_log"]
        m0_hat_log = trunk_out["m0_hat_log"]

        mgm_delta_ctx = self.mgm_ctx_decoder(torch.cat([z_cell, gene_expr_ctx], dim=-1))
        if g_in < self.n_genes:
            mgm_delta_ctx = mgm_delta_ctx[:, :g_in]
        elif g_in > self.n_genes:
            mgm_delta_ctx = torch.cat(
                [mgm_delta_ctx, torch.zeros(bsz, g_in - self.n_genes, device=device, dtype=dtype)], dim=-1
            )
        mgm_recon_full = trunk_out["mgm_recon"] + mgm_delta_ctx

        # 3. Compositional chemical, genetic, media & physical nanoparticle perturbations
        pert_out = self.perturbation_op(batch, bsz=bsz, device=device, dtype=dtype, p_hat=p_hat, z_cell=z_cell)
        target_occ = pert_out["target_occ"]
        gene_mult = pert_out["gene_mult"]
        media_vec = pert_out["media_vec"]
        z_chem_active = pert_out["z_chem_active"]

        is_3d_vec = torch.zeros(bsz, device=device, dtype=dtype)
        if "is_3d_organoid" in batch and isinstance(batch["is_3d_organoid"], torch.Tensor):
            is_3d_vec = torch.nan_to_num(batch["is_3d_organoid"].to(device=device, dtype=dtype).reshape(bsz), nan=0.0).clamp(0.0, 1.0)
            media_cols = list(media_vec.unbind(dim=-1))
            media_cols[0] = media_cols[0] * (1.0 - 0.15 * is_3d_vec)
            media_cols[8] = media_cols[8] * (1.0 - 0.20 * is_3d_vec)
            media_vec = torch.stack(media_cols, dim=-1)
            pert_out["media_vec"] = media_vec

        if self.ablate_cis_reg or epigen is None:
            cis_locus_ctx = torch.zeros(bsz, 64, device=device, dtype=dtype)
        else:
            ep_al = self.cis_reg_slot._align_feature_shape(epigen.to(device=device, dtype=dtype))
            cis_locus_ctx = 0.5 * ep_al[:, :64, 0] + 0.5 * ep_al[:, :64, 2]

        # 4 & 5. Sec46 translational gate + DiffFerroGEM grey-box peroxidation layer
        sec_out, gem_out, h_ferro_gem = self._compute_sec46_and_gem(
            z_cell,
            p_hat,
            m0_hat,
            pert_out,
            extra_cell_ctx,
            gene_expr_ctx,
            dtype,
            p_hat_log=p_hat_log,
            m0_hat_log=m0_hat_log,
            gene_expr_raw=x_aligned * eff_obs_m,
            gene_expr_mask=eff_obs_m,
            cis_locus_ctx=cis_locus_ctx,
        )
        e_gpx4_active = sec_out["e_gpx4_active"]
        pl_ooh = gem_out["pl_ooh"]
        g_rupture = gem_out["g_rupture"]
        fe2_lyso = gem_out["fe2_lyso"]
        seven_dhc_ss = gem_out["seven_dhc_ss"]
        fluxes = gem_out["fluxes"]
        sv_residual = gem_out["mass_balance_residual"]
        aphynity_logit = gem_out["aphynity_logit"]

        pert_summary = torch.cat(
            [
                target_occ,
                gene_mult,
                media_vec,
                pert_out["k_rta_drug"].unsqueeze(-1),
                pert_out["k_chelate_drug"].unsqueeze(-1),
                pert_out["delta_j_lyso_fe"].unsqueeze(-1),
                pert_out["np_innate_immune_stim"].unsqueeze(-1),
            ],
            dim=-1,
        )

        # ------------------------------------------------------------------
        # Readout Head 1: Zero-Bypass H_ferro + Competing-Risks RCD Hazards
        # ------------------------------------------------------------------
        if self.ablate_diff_ferro_gem:
            h_ferro = (
                F.softplus(
                    self.ablation_blackbox_ferro_head(torch.cat([z_cell, pert_summary, z_chem_active], dim=-1)).squeeze(-1)
                )
                + 1e-5
            )
        else:
            h_ferro = h_ferro_gem

        apop_cell_lat = self.apop_cell_proj(torch.cat([z_cell, gene_expr_ctx, extra_cell_ctx], dim=-1))
        apop_chem_lat = self.apop_chem_proj(z_chem_active)
        apop_bi_in = torch.cat([apop_cell_lat, apop_chem_lat, apop_cell_lat * apop_chem_lat], dim=-1)
        apop_gain = torch.exp(2.4 * torch.tanh(self.apop_bilinear_head(apop_bi_in).squeeze(-1)))
        drug_active_any = (z_chem_active.abs().sum(dim=-1) > 1e-6).to(dtype=dtype)
        intrinsic_drug_haz = F.softplus(self.apop_intrinsic_drug_head(apop_cell_lat).squeeze(-1)) * 0.28 * drug_active_any
        h_apop = (
            (F.softplus(self.apop_cell_head(z_cell).squeeze(-1)) * 0.03 + intrinsic_drug_haz + pert_out["k_apop_drug"] * apop_gain)
            / (1.0 + 2.0 * pert_out["zvad_conc"])
        ).clamp(min=1e-6)

        necro_gain = torch.exp(2.0 * torch.tanh(self.necro_bilinear_head(torch.cat([z_cell, z_chem_active], dim=-1)).squeeze(-1)))
        h_necro = (
            F.softplus(self.necro_cell_head(z_cell).squeeze(-1)) * 0.03 + pert_out["k_necro_drug"] * necro_gain
        ).clamp(min=1e-6)

        cupro_in = torch.cat([z_cell, gene_expr_ctx, F.silu(self.cupro_ctx_proj(extra_cell_ctx))], dim=-1)
        cupro_idx = self._pathway_idx_map.get("cupro_tca", [])
        if cupro_idx:
            c_m = eff_obs_m[:, cupro_idx]
            c_score = (x_obs_centered[:, cupro_idx] * c_m).sum(dim=-1) / c_m.sum(dim=-1).clamp(min=1.0)
            cupro_tca_drive = torch.exp(0.35 * c_score.clamp(-2.5, 2.5))
        else:
            cupro_tca_drive = torch.ones(bsz, device=device, dtype=dtype)
        cupro_suscept = (F.softplus(self.cupro_cell_head(cupro_in).squeeze(-1)) * 0.5 + 0.35) * cupro_tca_drive
        h_cupro = (
            cupro_suscept * pert_out["fdx1_mult"] * (1e-4 + 1.8 * pert_out["elesclomol_occ"])
        ).clamp(min=1e-6)

        xc_disulf_drive = (
            p_hat[:, PROTEIN_TO_IDX["SLC7A11"]]
            * gene_mult[:, PROTEIN_TO_IDX["SLC7A11"]]
            * (1.0 - 0.90 * target_occ[:, TARGET_TO_IDX["SLC7A11"]])
            * media_vec[:, 0]
            * torch.sigmoid(4.5 * (0.25 - media_vec[:, 2]))
        )
        k_txn_tox = F.softplus(self.raw_k_txnrd1_tox).to(dtype=dtype) + 0.15
        txnrd1_collapse_drive = (
            k_txn_tox
            * target_occ[:, TARGET_TO_IDX["TXNRD1"]].pow(2)
            * (1.2 / (0.25 + p_hat[:, PROTEIN_TO_IDX["TXNRD1"]]))
        )
        h_disulf = (
            F.softplus(self.disulf_cell_head(z_cell).squeeze(-1)) * 0.02
            + 0.35 * xc_disulf_drive
            + txnrd1_collapse_drive
        ).clamp(min=1e-6)

        hazards_5 = torch.stack([h_ferro, h_apop, h_necro, h_cupro, h_disulf], dim=-1)
        other_17 = F.softplus(self.other_17_rcd_head(z_cell)) * 0.005 + 1e-6
        rcd_22_hazards = torch.cat([hazards_5, other_17], dim=-1)

        total_hazard = hazards_5.sum(dim=-1) + other_17.sum(dim=-1) * 0.10
        h_pos = total_hazard.clamp(min=0.0)
        viability_pred = 0.99 * torch.exp(-h_pos.clamp(max=25.0)) + 0.01 / (1.0 + 0.25 * h_pos).pow(2)
        ferroptosis_prob = (1.0 - viability_pred) * (h_ferro / (total_hazard + 1e-6))

        if self.ablate_survivorship:
            survivorship_factor = torch.ones(bsz, device=device, dtype=dtype)
        else:
            survivorship_factor = 1.0 / (1.0 + 1.2 * g_rupture.detach() * (1.0 - viability_pred.detach()))

        rescue_delta_pred = torch.tanh(0.85 * h_ferro)
        synergy_in = torch.cat(
            [
                z_cell,
                pert_out["bliss_interaction"].unsqueeze(-1),
                pl_ooh.unsqueeze(-1),
                fe2_lyso.unsqueeze(-1),
                seven_dhc_ss.unsqueeze(-1),
            ],
            dim=-1,
        )
        synergy_pred = self.synergy_head(synergy_in).squeeze(-1) + 0.4 * pert_out["bliss_interaction"]

        # ------------------------------------------------------------------
        # Readout Head 2: Rank-64 Gene x Cell Bilinear KO + 4-Corner Analytical Epistasis
        # ------------------------------------------------------------------
        if raw_gko is not None:
            gko_aligned, _ = self.perceiver_trunk._align_genes(raw_gko.to(device=device, dtype=dtype))
            if raw_gko.shape[-1] < self.n_genes:
                gko_aligned[:, raw_gko.shape[-1] :] = 1.0
            ko_delta = (1.0 - gko_aligned.clamp(0.0, 2.0)).clamp(-1.0, 1.0)
        else:
            ko_delta = torch.zeros(bsz, self.n_genes, device=device, dtype=dtype)

        ko_emb = torch.matmul(ko_delta, self.perceiver_trunk.gene_key_emb.to(dtype=dtype))
        ko_basal_expr = (x_aligned * ko_delta).sum(dim=-1, keepdim=True)
        x_cell_centered = x_obs_centered
        ko_expr_centered = (x_cell_centered * ko_delta).sum(dim=-1, keepdim=True)
        ko_slope_shift = (x_cell_centered * ko_delta * self.gene_expr_slope_prior.to(dtype=dtype).unsqueeze(0)).sum(dim=-1)
        ko_prior_shift = torch.matmul(ko_delta, self.gene_essentiality_prior.to(dtype=dtype)) + ko_slope_shift
        cell_base_offset = self.cell_chronos_offset(gene_expr_ctx).squeeze(-1)

        ko_cnv = (extra_cell_ctx[:, :64] * ko_delta[:, : min(64, self.n_genes)]).sum(dim=-1, keepdim=True) if self.n_genes >= 64 else torch.zeros(bsz, 1, device=device, dtype=dtype)
        ko_lof = (extra_cell_ctx[:, 64:128] * ko_delta[:, : min(64, self.n_genes)]).sum(dim=-1, keepdim=True) if self.n_genes >= 64 else torch.zeros(bsz, 1, device=device, dtype=dtype)

        ko_lat64 = torch.matmul(ko_delta, self.gene_ctx_emb.to(dtype=dtype))
        cell_ko_ctx = self.cell_to_gene_ctx(
            torch.cat([z_cell, trunk_out["p_hat_log"], trunk_out["m0_hat_log"], extra_cell_ctx, gene_expr_ctx], dim=-1)
        )
        ko_cell_bilinear = ko_lat64 * cell_ko_ctx

        stress_feats = torch.stack([pl_ooh, g_rupture, fe2_lyso, seven_dhc_ss], dim=-1)
        pert_lat = self.pert_to_lat(torch.cat([z_cell, pert_summary.detach(), ko_emb, stress_feats.detach()], dim=-1))
        raw_delta_expr = torch.matmul(pert_lat, self.perceiver_trunk.gene_val_emb.to(dtype=dtype).t())
        if raw_gko is not None:
            raw_delta_expr = raw_delta_expr - 0.85 * (x_aligned * ko_delta.clamp(min=0.0))

        if g_in < self.n_genes:
            raw_delta_expr = raw_delta_expr[:, :g_in]
        elif g_in > self.n_genes:
            pad_d = torch.zeros(bsz, g_in - self.n_genes, device=device, dtype=dtype)
            raw_delta_expr = torch.cat([raw_delta_expr, pad_d], dim=-1)
        delta_expr_pred = raw_delta_expr * survivorship_factor.unsqueeze(-1)
        post_expr_pred = torch.nan_to_num(gene_expr, nan=0.0) + delta_expr_pred

        gem_ko_feats = torch.stack(
            [
                pl_ooh.log1p(),
                fe2_lyso.log1p(),
                g_rupture,
                seven_dhc_ss.log1p(),
                e_gpx4_active.log1p(),
                is_3d_vec,
            ],
            dim=-1,
        )
        chronos_in = torch.cat(
            [
                z_cell,
                gene_mult,
                ko_emb,
                ko_cell_bilinear,
                target_occ,
                ko_basal_expr,
                ko_expr_centered,
                ko_cnv,
                ko_lof,
                gem_ko_feats,
            ],
            dim=-1,
        )
        chronos_single = (
            self.chronos_head(chronos_in).squeeze(-1)
            + ko_cell_bilinear.sum(dim=-1)
            + ko_prior_shift
            + cell_base_offset
        )

        # Exact 4-Corner Population-Genetics Counterfactual Epistasis when BOTH Gene KO & Drug are Active!
        has_ko = ko_delta.abs().max(dim=-1).values > 0.05
        has_drug = (
            (target_occ.max(dim=-1).values > 0.01)
            | (pert_out["k_rta_drug"] > 0.01)
            | (pert_out["k_chelate_drug"] > 0.01)
        )
        dual_mask = has_ko & has_drug
        if not self.ablate_diff_ferro_gem and bool(dual_mask.any().item()):
            d_idx = torch.where(dual_mask)[0]
            nd = int(d_idx.numel())
            zc_d = z_cell[d_idx]
            ph_d = p_hat[d_idx]
            m0_d = m0_hat[d_idx]
            plog_d = p_hat_log[d_idx]
            mlog_d = m0_hat_log[d_idx]
            ex_d = extra_cell_ctx[d_idx]
            gx_d = gene_expr_ctx[d_idx]
            xraw_d = (x_aligned * eff_obs_m)[d_idx]
            xmsk_d = eff_obs_m[d_idx]
            ones_d = torch.ones(nd, device=device, dtype=dtype)
            ones_28 = torch.ones(nd, len(ANCHOR_28_PROTEINS), device=device, dtype=dtype)
            zeros_14 = torch.zeros(nd, len(CHEM_14_TARGETS), device=device, dtype=dtype)
            zeros_1d = torch.zeros(nd, device=device, dtype=dtype)
            zeros_zchem = torch.zeros(nd, z_chem_active.shape[-1], device=device, dtype=dtype)

            def _slice_pert(overrides: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                sub_p = {
                    k: (v[d_idx] if isinstance(v, torch.Tensor) and v.shape[0] == bsz else v)
                    for k, v in pert_out.items()
                }
                sub_p.update(overrides)
                return sub_p

            # Corner (0, 1): WT + Drug
            _, gem_wd, _ = self._compute_sec46_and_gem(
                zc_d,
                ph_d,
                m0_d,
                _slice_pert(
                    {
                        "gene_mult": ones_28,
                        "epl_biosynth_mult": ones_d,
                        "dhodh_gene_mult": ones_d,
                        "repair_shield_mult": ones_d,
                        "v_atpase_gate": (1.0 - 0.92 * target_occ[d_idx, TARGET_TO_IDX["V_ATPase"]]).clamp(min=0.02),
                    }
                ),
                ex_d,
                gx_d,
                dtype,
                p_hat_log=plog_d,
                m0_hat_log=mlog_d,
                gene_expr_raw=xraw_d,
                gene_expr_mask=xmsk_d,
            )
            # Corner (1, 0): KO + Vehicle
            sec_k0, gem_k0, _ = self._compute_sec46_and_gem(
                zc_d,
                ph_d,
                m0_d,
                _slice_pert(
                    {
                        "target_occ": zeros_14,
                        "k_rta_drug": zeros_1d,
                        "k_chelate_drug": zeros_1d,
                        "z_chem_active": zeros_zchem,
                    }
                ),
                ex_d,
                gx_d,
                dtype,
                p_hat_log=plog_d,
                m0_hat_log=mlog_d,
                gene_expr_raw=xraw_d,
                gene_expr_mask=xmsk_d,
            )
            # Corner (0, 0): WT + Vehicle
            _, gem_w0, _ = self._compute_sec46_and_gem(
                zc_d,
                ph_d,
                m0_d,
                _slice_pert(
                    {
                        "gene_mult": ones_28,
                        "epl_biosynth_mult": ones_d,
                        "dhodh_gene_mult": ones_d,
                        "repair_shield_mult": ones_d,
                        "v_atpase_gate": ones_d,
                        "target_occ": zeros_14,
                        "k_rta_drug": zeros_1d,
                        "k_chelate_drug": zeros_1d,
                        "z_chem_active": zeros_zchem,
                    }
                ),
                ex_d,
                gx_d,
                dtype,
                p_hat_log=plog_d,
                m0_hat_log=mlog_d,
                gene_expr_raw=xraw_d,
                gene_expr_mask=xmsk_d,
            )

            gain_d = gem_out["aphynity_gain"][d_idx]
            hf_kd = gem_out["h_ferro_phys"][d_idx] * gain_d
            hf_wd = gem_wd["h_ferro_phys"] * gain_d
            hf_k0 = gem_k0["h_ferro_phys"] * gain_d
            hf_w0 = gem_w0["h_ferro_phys"] * gain_d
            # Cuproptosis & TXNRD1 analytical interactions
            hc_kd = h_cupro[d_idx] + txnrd1_collapse_drive[d_idx]
            hc_wd = (
                cupro_suscept[d_idx] * (1e-4 + 1.8 * pert_out["elesclomol_occ"][d_idx])
                + txnrd1_collapse_drive[d_idx]
            ).clamp(min=1e-6)
            hc_k0 = (cupro_suscept[d_idx] * pert_out["fdx1_mult"][d_idx] * 1e-4).clamp(min=1e-6)
            hc_w0 = (cupro_suscept[d_idx] * 1e-4).clamp(min=1e-6)

            # Exact Poisson survival log2-fold-change (`log2 S = -H / ln 2`, free of log1p concavity distortion!)
            d_drug_ferro = hf_kd - hf_wd
            d_veh_ferro = hf_k0 - hf_w0
            dd_ferro = d_drug_ferro - d_veh_ferro

            d_drug_cupro = hc_kd - hc_wd
            d_veh_cupro = hc_k0 - hc_w0
            dd_cupro = d_drug_cupro - d_veh_cupro

            inv_ln2 = 1.0 / math.log(2.0)
            epi_gain = F.softplus(self.raw_epistasis_gain).to(dtype=dtype) + 1.8
            combined_hazard_shift = 0.70 * (d_drug_ferro + d_drug_cupro) + 0.30 * (dd_ferro + dd_cupro)
            epistatic_lfc = 3.2 * torch.tanh((-inv_ln2 * epi_gain * combined_hazard_shift) / 3.2)

            gem_k0_feats = torch.stack(
                [
                    gem_k0["pl_ooh"].log1p(),
                    gem_k0["fe2_lyso"].log1p(),
                    gem_k0["g_rupture"],
                    gem_k0["seven_dhc_ss"].log1p(),
                    sec_k0["e_gpx4_active"].log1p(),
                    is_3d_vec[d_idx],
                ],
                dim=-1,
            )
            chronos_in_k0 = torch.cat(
                [
                    zc_d,
                    gene_mult[d_idx],
                    ko_emb[d_idx],
                    ko_cell_bilinear[d_idx],
                    zeros_14,
                    ko_basal_expr[d_idx],
                    ko_expr_centered[d_idx],
                    ko_cnv[d_idx],
                    ko_lof[d_idx],
                    gem_k0_feats,
                ],
                dim=-1,
            )
            basal_ko_fit = (
                self.chronos_head(chronos_in_k0).squeeze(-1)
                + ko_cell_bilinear[d_idx].sum(dim=-1)
                + ko_prior_shift[d_idx]
                + cell_base_offset[d_idx]
            )
            chronos_dual = epistatic_lfc + 0.08 * ko_prior_shift[d_idx] + 0.04 * basal_ko_fit

            chronos_pred = chronos_single.clone()
            chronos_pred[d_idx] = chronos_dual
        else:
            chronos_pred = chronos_single

        imm_in = torch.cat(
            [
                z_cell,
                gene_expr_ctx,
                pl_ooh.unsqueeze(-1),
                g_rupture.unsqueeze(-1),
                pert_out["np_innate_immune_stim"].unsqueeze(-1),
                seven_dhc_ss.unsqueeze(-1),
                media_vec[:, 9:10],
                viability_pred.unsqueeze(-1),
                h_ferro.log1p().unsqueeze(-1),
            ],
            dim=-1,
        )
        imm_raw = self.immunogenic_head(imm_in)
        immunogenic_pathways = imm_raw[:, : len(IMMUNOGENIC_6_PATHWAYS)]
        icb_response_logit = (
            imm_raw[:, -1]
            + 0.8 * g_rupture
            + 0.6 * pert_out["np_innate_immune_stim"]
            + 0.4 * media_vec[:, 9]
        )

        # ------------------------------------------------------------------
        # Readout Head 3: Selective Ferroptotic Therapeutic Index (ZERO Label Leak!)
        # ------------------------------------------------------------------
        renal_prot = pert_out["renal_clearance_frac"]
        est_systemic_tox = ferroptosis_prob * 0.35 * (1.0 - 0.75 * renal_prot)
        sfti_score = ferroptosis_prob - 0.5 * est_systemic_tox

        metab_10_pred = torch.stack(
            [
                m0_hat[:, METABOLITE_TO_IDX["GSH"]],
                m0_hat[:, METABOLITE_TO_IDX["GSSG"]],
                m0_hat[:, METABOLITE_TO_IDX["NADP+"]],
                m0_hat[:, METABOLITE_TO_IDX["putrescine"]],
                m0_hat[:, METABOLITE_TO_IDX["spermidine"]],
                m0_hat[:, METABOLITE_TO_IDX["PC_PUFA2"]],
                m0_hat[:, METABOLITE_TO_IDX["PE_PUFA"]],
                m0_hat[:, METABOLITE_TO_IDX["ePL_PUFA"]],
                m0_hat[:, METABOLITE_TO_IDX["MUFA_PL"]],
                seven_dhc_ss,
            ],
            dim=-1,
        )
        oxylipid_4_pred = torch.stack([pl_ooh, seven_dhc_ss, fe2_lyso, g_rupture], dim=-1)

        out_payload: dict[str, Any] = {
            "fluxes": fluxes,
            "mass_balance_residual": sv_residual,
            "pl_ooh": pl_ooh,
            "fe2_lyso": fe2_lyso,
            "g_rupture": g_rupture,
            "seven_dhc_ss": seven_dhc_ss,
            "sterol_7dhc": seven_dhc_ss,
            "e_gpx4_active": e_gpx4_active,
            "sec_readthrough_eff": sec_out["sec_readthrough_eff"],
            "aphynity_gain": gem_out["aphynity_gain"],
            "aphynity_logit": aphynity_logit,
            "rescue_shield": gem_out["rescue_shield"],
            "rates": gem_out["rates"],
            "h_ferro": h_ferro,
            "h_apop": h_apop,
            "h_necro": h_necro,
            "h_cupro": h_cupro,
            "h_disulf": h_disulf,
            "hazards": hazards_5,
            "rcd_22_hazards": rcd_22_hazards,
            "viability_pred": viability_pred,
            "ferroptosis_prob": ferroptosis_prob,
            "survivorship_factor": survivorship_factor,
            "ablate_survivorship_flag": self.ablate_survivorship,
            "rescue_delta_pred": rescue_delta_pred,
            "synergy_pred": synergy_pred,
            "z_cell": z_cell,
            "z_modules": trunk_out["z_modules"],
            "p_hat": p_hat,
            "p_hat_log": trunk_out["p_hat_log"],
            "m0_hat": m0_hat,
            "metab_anchor_pred": metab_10_pred,
            "oxylipid_pred": oxylipid_4_pred,
            "mgm_recon": mgm_recon_full,
            "delta_expr_pred": delta_expr_pred,
            "post_expr_pred": post_expr_pred,
            "chronos_pred": chronos_pred,
            "immunogenic_pathways": immunogenic_pathways,
            "icb_response_logit": icb_response_logit,
            "sfti_score": sfti_score,
            "pred_pec50": pert_out["pred_pec50"],
        }
        if mgm_eval_mask is not None:
            out_payload["mgm_eval_mask"] = mgm_eval_mask
        if gt_imm_target is not None and gt_imm_mask is not None:
            out_payload["gt_immunogenic_target"] = gt_imm_target
            out_payload["gt_immunogenic_mask"] = gt_imm_mask
        return out_payload

    def compute_sfti(
        self,
        tumor_expr: torch.Tensor,
        normal_panel_expr: torch.Tensor | None = None,
        *,
        lambda_tox: float = 1.0,
        **pert_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute paired Selective Ferroptotic Therapeutic Index across starved TME vs fed normal organs."""
        bsz = tumor_expr.shape[0]
        dev = tumor_expr.device
        dtp = tumor_expr.dtype

        tumor_batch: dict[str, Any] = {
            "gene_expr": tumor_expr,
            "cys2_media": torch.full((bsz,), 0.08, device=dev, dtype=dtp),
            "glc_media": torch.full((bsz,), 0.35, device=dev, dtype=dtp),
            "po2_media": torch.full((bsz,), 0.25, device=dev, dtype=dtp),
        }
        tumor_batch.update(pert_kwargs)
        out_tumor = self.forward(tumor_batch)
        p_tumor = out_tumor["ferroptosis_prob"]

        if normal_panel_expr is None:
            normal_panel_expr = tumor_expr.detach() * 0.85

        n_norm = normal_panel_expr.shape[0]
        norm_batch: dict[str, Any] = {
            "gene_expr": normal_panel_expr,
            "cys2_media": torch.ones(n_norm, device=dev, dtype=dtp),
            "glc_media": torch.ones(n_norm, device=dev, dtype=dtp),
            "po2_media": torch.ones(n_norm, device=dev, dtype=dtp),
        }
        for k, v in pert_kwargs.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == bsz and n_norm != bsz:
                norm_batch[k] = v[:1].expand(n_norm, *v.shape[1:])
            else:
                norm_batch[k] = v
        out_norm = self.forward(norm_batch)
        renal_shield = torch.ones(n_norm, device=dev, dtype=dtp)
        if "np_params" in norm_batch and isinstance(norm_batch["np_params"], torch.Tensor):
            r_h = norm_batch["np_params"][:, 0]
            renal_shield = 1.0 - 0.75 * torch.sigmoid((8.0 - r_h) / 1.25)
        p_norm = out_norm["ferroptosis_prob"] * renal_shield
        max_norm_tox = p_norm.max() if n_norm != bsz else p_norm

        sfti = p_tumor - float(lambda_tox) * max_norm_tox
        return {
            "sfti": sfti,
            "tumor_ferro_prob": p_tumor,
            "max_normal_ferro_prob": max_norm_tox if isinstance(max_norm_tox, torch.Tensor) else torch.tensor(max_norm_tox),
        }

    def predict_spatial_wave(
        self,
        gene_expr: torch.Tensor,
        coords: torch.Tensor | None = None,
        *,
        n_steps: int = 12,
        dt_step: float = 0.5,
        diffusion_coeff: float = 18.0,
        paracrine_length_um: float = 35.0,
        ref_well_deg35: float | None = None,
        seed_spots: torch.Tensor | None = None,
        **pert_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """2D spatial reaction-diffusion trigger-wave PDE rollout over micrometer tissue coordinates `[N, 2]`."""
        if gene_expr.ndim == 1:
            gene_expr = gene_expr.unsqueeze(0)
        n_spots = gene_expr.shape[0]
        device = gene_expr.device
        dtype = gene_expr.dtype

        if coords is None:
            side = max(1, int(n_spots**0.5))
            grid_i = torch.arange(n_spots, device=device, dtype=dtype)
            coords = torch.stack([(grid_i % side) * 25.0, (grid_i // side) * 25.0], dim=-1)
        else:
            coords = torch.nan_to_num(coords.to(device=device, dtype=dtype), nan=0.0)

        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist_sq = (diff.pow(2)).sum(dim=-1)
        dist = (dist_sq + 1e-6).sqrt()
        eye = torch.eye(n_spots, device=device, dtype=dtype)

        lam = max(5.0, float(paracrine_length_um))
        adj_phys = torch.exp(-dist / lam) * (1.0 - eye)
        deg35_vec = adj_phys.sum(dim=-1)
        ref_deg = float(ref_well_deg35) if (ref_well_deg35 is not None and ref_well_deg35 > 1e-4) else 8.0
        paracrine_coupling = (deg35_vec / ref_deg).clamp(0.02, 4.0)

        mean_density = deg35_vec.mean().clamp(min=1e-4)
        yap_density_gate = torch.sigmoid(1.2 * (1.15 - deg35_vec / (mean_density + 4.0)))

        spatial_batch: dict[str, Any] = {"gene_expr": gene_expr}
        spatial_batch.update(pert_kwargs)
        base_out = self.forward(spatial_batch)

        pl_ooh_base = base_out["pl_ooh"] * (0.75 + 0.50 * yap_density_gate)
        g_rup = base_out["g_rupture"]
        gsh = base_out["metab_anchor_pred"][:, 0].clamp(min=0.05)

        w_paracrine = adj_phys / ref_deg

        centroid = coords.mean(dim=0, keepdim=True)
        dist_to_center = ((coords - centroid).pow(2).sum(dim=-1) + 1e-6).sqrt()
        if seed_spots is None:
            seed_profile = torch.exp(-dist_to_center.pow(2) / (2.0 * (lam * 0.75) ** 2))
        else:
            seed_profile = seed_spots.to(device=device, dtype=dtype).reshape(n_spots).clamp(0.0, 1.0)

        u = (pl_ooh_base + 1.25 * seed_profile).clamp(min=1e-5)
        traj = [u]

        propagative_gate = g_rup / (0.35 + gsh)
        d_eff = (float(diffusion_coeff) / (lam + 10.0)) * (0.20 + propagative_gate)

        for _ in range(max(1, int(n_steps))):
            paracrine_influx = torch.matmul(w_paracrine, u * (0.3 + 0.7 * g_rup))
            bistable_react = (
                0.45 * pl_ooh_base
                + 0.85 * (u.pow(2) / (0.60 + u.pow(2))) * paracrine_coupling
                - 0.45 * gsh * u
            )
            du = d_eff * (paracrine_influx - 0.5 * u) + bistable_react
            u = (u + float(dt_step) * du).clamp(min=1e-6, max=50.0)
            traj.append(u)

        wave_traj = torch.stack(traj, dim=0)
        u_final = wave_traj[-1]

        thresh = 0.5 * (wave_traj[0].mean() + u_final.max())
        front_w0 = torch.softmax(5.0 * (wave_traj[0] - thresh), dim=0)
        front_wT = torch.softmax(5.0 * (u_final - thresh), dim=0)
        r_start = (front_w0 * dist_to_center).sum()
        r_end = (front_wT * dist_to_center).sum()
        total_time = max(1e-3, float(n_steps) * float(dt_step))
        c_wave = F.softplus((r_end - r_start) / total_time + 0.65 * u_final.mean() * propagative_gate.mean())

        z_dev = u_final - u_final.mean()
        w_sum = adj_phys.sum().clamp(min=1e-6)
        num = float(n_spots) * torch.matmul(z_dev, torch.matmul(adj_phys, z_dev))
        den = w_sum * (z_dev.pow(2).sum() + 1e-6)
        morans_i = (num / den).clamp(-1.0, 1.0)

        propagative_prob = torch.sigmoid(2.0 * (propagative_gate.mean() * paracrine_coupling.mean() - 0.30))

        return {
            "wave_trajectory": wave_traj,
            "u_final": u_final,
            "c_wave": c_wave,
            "morans_i": morans_i,
            "propagative_prob": propagative_prob,
            "yap_density_gate": yap_density_gate,
            "paracrine_coupling": paracrine_coupling,
        }


__all__ = [
    "PRIMARY_5_HAZARDS",
    "IMMUNOGENIC_6_PATHWAYS",
    "FerroFluxFM",
]
