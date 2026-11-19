r"""Compositional Perturbation Operator (`CompositionalPerturbationOperator`) — FerroFlux-FM v3.

Upgrades in v3:
1. **Multi-Scale Residual-SwiGLU Chemical Encoder (`3,239`-bit ECFP4/ECFP6/MACCS + `24` RDKit/SMARTS)**:
   Includes `sar_pretrain_head` (`20` quantitative potency + `20` binary activity channels across
   `chem_ferrdb_qhts_chembl_features.parquet`) directly coupled to `chem_mech_head` (`pec50` across
   the 14 enzymatic targets) + CysDB covalent Sec/Cys reactivity priors.
2. **Cell-State Bioactivation & Efflux FiLM Modulation (`cell_bioact_film`)**:
   Adapts effective drug target `pEC50` across cellular redox/metabolic states (`z_cell`, `POR`,
   `TXNRD1`, `GCLC`, `SLC7A11`) while preserving exact `dose == 0 -> occ == 0` invariance.
3. **Complete Canonical Ferroptosis Regulatory-Stoichiometric Gene KO Wiring (~65 Pathway Genes)**:
   Maps `KEAP1`/`NFE2L2`/`BACH1`, peroxisomal `PEX3/10/12`/`FAR1`/`AGPS`/`GNPAT`, `MBOAT1/2`/`SCD`,
   `FADS1/2`, `PLA2G6`/`CHMP4B`, `SC5D`/`CYP51A1`/`MSMO1`, `MVK`/`FDPS`, `COQ2`/`PDSS1/2`/`VKORC1L1`,
   `DHODH`, `CYB5R1`, `ATP6V0/1*`, `SLC40A1`/`CISD1`/`NFS1`, and `LIAS`/`DLAT` knockouts cleanly
   onto stoichiometric multipliers (`1.0` when unperturbed).
4. **Biophysically Accurate Iron-Scavenging Ultrasmall PEG-Silica C' Dots (`Kim et al., Nat. Nanotechnol. 2016`)**.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ferroflux.layers.perceiver_trunk import (
    ANCHOR_28_PROTEINS,
    PROTEIN_TO_IDX,
    SwiGLU,
    resolve_vocab_gene_indices,
)

CHEM_14_TARGETS: tuple[str, ...] = (
    "GPX4_cov",     # 0: RSL3 / ML210 / ML162 covalent Sec46 alkylation
    "GPX4_deg",     # 1: FIN56 GPX4 degradation / FINO2 endoperoxide oxidation
    "SLC7A11",      # 2: Erastin / IKE / Sorafenib / Sulfasalazine system xc- block
    "FSP1",         # 3: iFSP1 / icFSP1 AIFM2 inhibition
    "DHODH",        # 4: Brequinar / Teriflunomide mitochondrial CoQ10H2 / FSP1 inhibition
    "ACSL4",        # 5: Rosiglitazone / Troglitazone ACSL4 inhibition
    "ALOX",         # 6: PD146176 / Baicalein / Zileuton lipoxygenase inhibition
    "HMGCR",        # 7: Statins mevalonate block
    "DHCR7",        # 8: AY9944 / Tamoxifen distal sterol reductase inhibition
    "EBP",          # 9: TASIN-1 / EBP isomerase inhibition
    "SQLE",         # 10: Terbinafine / NB-598 squalene epoxidase inhibition
    "TXNRD1",       # 11: Auranofin / Ferroptocide thioredoxin reductase inhibition
    "V_ATPase",     # 12: Concanamycin A / Bafilomycin A1 endolysosomal pH block
    "FDX1_Cu",      # 13: Elesclomol / Disulfiram Cu2+-ionophore cuproptosis trigger
)
TARGET_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CHEM_14_TARGETS)}

MEDIA_10_CHANNELS: tuple[str, ...] = (
    "Cys2",     # 0: Extracellular cystine (replete=1.0, starved=0.01..0.05)
    "Gln",      # 1: Extracellular L-glutamine
    "Glc",      # 2: Extracellular D-glucose (low in SLC7A11-hi -> disulfidptosis)
    "Se",       # 3: Extracellular selenite / SELENOP bioavailability
    "Fe3",      # 4: Extracellular holo-transferrin / ambient ferric iron
    "OA_18_1",  # 5: Exogenous oleic acid (18:1 MUFA protection via ACSL3/MBOAT1/2)
    "AA_20_4",  # 6: Exogenous arachidonic/adrenic acid (PUFA sensitization via ACSL4)
    "VK_VE",    # 7: Exogenous vitamin K2 (MK-4) / alpha-tocopherol
    "pO2",      # 8: Oxygen tension (normoxia=1.0, hypoxia=0.1)
    "IFNg",     # 9: CD8+ T-cell IFN-gamma cytokine concentration (basal=0.0)
)
MEDIA_DEFAULTS: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 0.0)

DEFAULT_ANCHOR_28_VOCAB_IDX: tuple[int, ...] = tuple(resolve_vocab_gene_indices(ANCHOR_28_PROTEINS, 2048))


def _load_vocab_map() -> dict[str, int]:
    vocab_p = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "gene_vocab.json"
    if vocab_p.is_file():
        try:
            return {str(k): int(v) for k, v in json.loads(vocab_p.read_text(encoding="utf-8")).get("gene_to_idx", {}).items()}
        except Exception:
            return {}
    return {}


_VOCAB_G2I: dict[str, int] = _load_vocab_map()


def _resolve_single_gene_idx(sym: str, fallback: int) -> int:
    return int(_VOCAB_G2I.get(sym, fallback))


def _resolve_group_indices(syms: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(int(_VOCAB_G2I[s]) for s in syms if s in _VOCAB_G2I)


FDX1_VOCAB_IDX: int = _resolve_single_gene_idx("FDX1", 62)
MC1R_VOCAB_IDX: int = _resolve_single_gene_idx("MC1R", -1)
FOLH1_VOCAB_IDX: int = _resolve_single_gene_idx("FOLH1", -1)
KEAP1_VOCAB_IDX: int = _resolve_single_gene_idx("KEAP1", 57)
NFE2L2_VOCAB_IDX: int = _resolve_single_gene_idx("NFE2L2", 56)
BACH1_VOCAB_IDX: int = _resolve_single_gene_idx("BACH1", 58)
DHODH_VOCAB_IDX: int = _resolve_single_gene_idx("DHODH", 16)

EPL_BIOSYNTH_IDX: tuple[int, ...] = _resolve_group_indices(("FAR1", "AGPS", "GNPAT", "PEX3", "PEX10", "PEX12", "TMEM189", "EPAS1"))
PUFA_DESAT_IDX: tuple[int, ...] = _resolve_group_indices(("FADS1", "FADS2", "ELOVL5"))
MUFA_SHIELD_IDX: tuple[int, ...] = _resolve_group_indices(("SCD", "MBOAT1", "MBOAT2"))
REPAIR_SHIELD_IDX: tuple[int, ...] = _resolve_group_indices(("PLA2G6", "CHMP4B", "MGST1", "PRDX6"))
STEROL_UP_PARTNER_IDX: tuple[int, ...] = _resolve_group_indices(("SC5D", "FDFT1", "LSS", "CYP51A1", "MSMO1", "NSDHL"))
MEVALONATE_PARTNER_IDX: tuple[int, ...] = _resolve_group_indices(("MVK", "PMVK", "MVD", "FDPS", "IDI1"))
COQ_VK_PARTNER_IDX: tuple[int, ...] = _resolve_group_indices(("COQ2", "COQ3", "COQ5", "COQ6", "COQ7", "PDSS1", "PDSS2", "UBIAD1", "VKORC1L1", "NADK"))
BH4_PARTNER_IDX: tuple[int, ...] = _resolve_group_indices(("PTS", "SPR"))
GCLM_PARTNER_IDX: tuple[int, ...] = _resolve_group_indices(("GCLM",))
SEC_PARTNER_IDX: tuple[int, ...] = _resolve_group_indices(("SECISBP2", "LRP8", "TRIT1", "SEPHS2", "EEFSEC"))
POR_PARTNER_IDX: tuple[int, ...] = _resolve_group_indices(("CYB5R1", "ALOX15", "ALOX12"))
VATPASE_SUBUNIT_IDX: tuple[int, ...] = _resolve_group_indices(("ATP6V0A1", "ATP6V0C", "ATP6V0D1", "ATP6V1A", "ATP6V1B2", "TCIRG1"))
FE_IMPORT_PARTNER_IDX: tuple[int, ...] = _resolve_group_indices(("STEAP3", "SLC11A2"))
FE_EXPORT_FES_IDX: tuple[int, ...] = _resolve_group_indices(("SLC40A1", "CISD1", "NFS1", "ISCU", "FXN", "PCBP1"))
CUPRO_LIPOYL_IDX: tuple[int, ...] = _resolve_group_indices(("LIAS", "DLAT"))


def parameter_free_nanoparticle_biophysics(
    np_params: torch.Tensor | None,
    *,
    cys2_media: torch.Tensor,
    fe3_media: torch.Tensor,
    v_atpase_occ: torch.Tensor,
    p_hat: torch.Tensor | None = None,
    gene_expr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Biophysical iron-scavenging & endolysosomal transport equations for C' dots & IONPs."""
    bsz = cys2_media.shape[0]
    device = cys2_media.device
    dtype = cys2_media.dtype

    if np_params is None:
        zeros = torch.zeros(bsz, device=device, dtype=dtype)
        return {
            "delta_j_lyso_fe": zeros,
            "v_atpase_gate": (1.0 - 0.92 * v_atpase_occ).clamp(min=0.02),
            "renal_clearance_frac": zeros,
            "np_uptake_rate": zeros,
            "np_innate_immune_stim": zeros,
        }

    np_clean = torch.nan_to_num(np_params.to(device=device, dtype=dtype), nan=0.0, posinf=100.0, neginf=-100.0)
    if np_clean.ndim == 1:
        np_clean = np_clean.unsqueeze(0).expand(bsz, -1)
    if np_clean.shape[-1] < 5:
        pad = torch.zeros(bsz, 5 - np_clean.shape[-1], device=device, dtype=dtype)
        np_clean = torch.cat([np_clean, pad], dim=-1)

    r_h = np_clean[:, 0].clamp(min=0.0)
    zeta = np_clean[:, 1]
    fe_load = np_clean[:, 2].clamp(min=0.0)
    rec_match = np_clean[:, 3].clamp(min=0.0)
    np_dose = np_clean[:, 4].clamp(min=0.0)

    active_mask = ((r_h > 0.05) & (np_dose > 1e-8)).to(dtype=dtype)

    v_atpase_gate = (1.0 - 0.92 * v_atpase_occ.clamp(0.0, 1.0)).clamp(min=0.02)
    starvation_boost = 1.0 + 2.8 * torch.sigmoid(4.5 * (0.45 - cys2_media))
    size_penetration = torch.sigmoid((18.0 - r_h) / 4.5)
    zeta_factor = 0.4 + 0.8 * torch.sigmoid((zeta + 25.0) / 10.0)

    endo_receptor = torch.ones(bsz, device=device, dtype=dtype)
    if p_hat is not None and p_hat.shape[-1] > PROTEIN_TO_IDX["TFRC"]:
        tfrc_norm = p_hat[:, PROTEIN_TO_IDX["TFRC"]].clamp(0.2, 5.0)
        endo_receptor = 0.55 + 0.35 * tfrc_norm
    if gene_expr is not None:
        g_dim = gene_expr.shape[-1]
        extra_rec = torch.zeros(bsz, device=device, dtype=dtype)
        if 0 <= MC1R_VOCAB_IDX < g_dim:
            extra_rec = torch.maximum(extra_rec, gene_expr[:, MC1R_VOCAB_IDX].clamp(0.0, 6.0) * 0.25)
        if 0 <= FOLH1_VOCAB_IDX < g_dim:
            extra_rec = torch.maximum(extra_rec, gene_expr[:, FOLH1_VOCAB_IDX].clamp(0.0, 6.0) * 0.25)
        endo_receptor = endo_receptor + extra_rec

    np_uptake_rate = (
        active_mask
        * v_atpase_gate
        * starvation_boost
        * size_penetration
        * zeta_factor
        * (0.35 + 0.85 * rec_match * endo_receptor)
        * torch.log1p(np_dose)
    )

    ambient_fe = fe3_media.clamp(min=0.05)
    ultrasmall_surface_scavenge = 0.85 * torch.sigmoid((12.0 - r_h) / 2.5) * ambient_fe
    intrinsic_core_fe = fe_load * (0.45 + 0.55 * ambient_fe)
    delta_j_lyso_fe = np_uptake_rate * (ultrasmall_surface_scavenge + intrinsic_core_fe)

    renal_clearance_frac = torch.where(
        active_mask > 0.5,
        torch.sigmoid((8.0 - r_h) / 1.25),
        torch.zeros(bsz, device=device, dtype=dtype),
    )
    np_innate_immune_stim = active_mask * torch.tanh(0.45 * np_uptake_rate * (1.0 + 0.35 * (fe_load + ultrasmall_surface_scavenge)))

    return {
        "delta_j_lyso_fe": delta_j_lyso_fe,
        "v_atpase_gate": v_atpase_gate,
        "renal_clearance_frac": renal_clearance_frac,
        "np_uptake_rate": np_uptake_rate,
        "np_innate_immune_stim": np_innate_immune_stim,
    }


class _ChemResidualSwiGLU(nn.Module):
    """Residual LayerNorm + SwiGLU block for multi-scale chemical structure encoding."""

    def __init__(self, dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


N_SAR_TARGETS: int = 21


class CompositionalPerturbationOperator(nn.Module):
    """Composes multi-scale small-molecule, full-pathway genetic, media, and C' dot perturbations (v3)."""

    def __init__(
        self,
        chem_fp_dim: int = 3239,
        chem_desc_dim: int = 24,
        d_latent: int = 192,
        d_cell: int = 256,
        ablate_chem_prior: bool = False,
    ) -> None:
        super().__init__()
        self.chem_fp_dim = int(chem_fp_dim)
        self.chem_desc_dim = int(chem_desc_dim)
        self.chem_in_dim = self.chem_fp_dim + self.chem_desc_dim
        self.d_latent = int(d_latent)
        self.d_cell = int(d_cell)
        self.ablate_chem_prior = bool(ablate_chem_prior)
        self.ablate_sec46_gate: bool = False
        self.n_sar_targets = N_SAR_TARGETS
        d_desc_hid = max(32, self.d_latent // 2)

        self.chem_fp_proj = nn.Sequential(
            nn.Linear(self.chem_fp_dim, self.d_latent),
            nn.LayerNorm(self.d_latent),
            nn.SiLU(),
        )
        self.chem_desc_proj = nn.Sequential(
            nn.Linear(self.chem_desc_dim, d_desc_hid),
            nn.LayerNorm(d_desc_hid),
            nn.SiLU(),
        )
        self.chem_in_proj = nn.Sequential(
            nn.Linear(self.d_latent + d_desc_hid, self.d_latent),
            nn.LayerNorm(self.d_latent),
            nn.SiLU(),
        )
        self.chem_blocks = nn.ModuleList(
            [
                _ChemResidualSwiGLU(self.d_latent, dropout=0.05),
                _ChemResidualSwiGLU(self.d_latent, dropout=0.05),
                _ChemResidualSwiGLU(self.d_latent, dropout=0.05),
            ]
        )
        self.chem_out_norm = nn.LayerNorm(self.d_latent)

        # Stage-0.5 Biochemical & Phenotypic SAR Multi-Task Pretraining Head (`21` potency + `21` binary logits)
        self.sar_pretrain_head = nn.Linear(self.d_latent, 2 * self.n_sar_targets)
        self.raw_sar_cell_offset = nn.Parameter(torch.tensor(0.20))

        # Frozen Stage-0.5 SAR Anchor Snapshot (prevents catastrophic forgetting on 39-89 cellular drugs)
        self.sar_anchor_fp_proj = copy.deepcopy(self.chem_fp_proj).requires_grad_(False)
        self.sar_anchor_desc_proj = copy.deepcopy(self.chem_desc_proj).requires_grad_(False)
        self.sar_anchor_in_proj = copy.deepcopy(self.chem_in_proj).requires_grad_(False)
        self.sar_anchor_blocks = copy.deepcopy(self.chem_blocks).requires_grad_(False)
        self.sar_anchor_out_norm = copy.deepcopy(self.chem_out_norm).requires_grad_(False)
        self.sar_anchor_head = copy.deepcopy(self.sar_pretrain_head).requires_grad_(False)
        self.register_buffer("has_sar_snapshot", torch.tensor(0, dtype=torch.int64), persistent=True)

        # Factored Scaffold-Invariant (Desc + MACCS + Full Frozen SAR + Dose) + Shared-ECFP Dosed Chemical Encoder
        self.ecfp_sub_dim = min(3072, self.chem_fp_dim)
        self.maccs_sub_dim = max(0, self.chem_fp_dim - 3072)
        self.sar_snap_dim = 2 * self.n_sar_targets + self.d_latent
        self.scaf_dense_dim = self.chem_desc_dim + self.maccs_sub_dim + self.sar_snap_dim + 4 + self.chem_desc_dim
        self.register_buffer("chem_dense_mean", torch.zeros(self.scaf_dense_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("chem_dense_scale", torch.ones(self.scaf_dense_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("shared_ecfp_mask", torch.ones(self.ecfp_sub_dim, dtype=torch.float32), persistent=True)

        d_ecfp_br = max(64, self.d_latent // 2)
        self.chem_scaffold_inv_enc = nn.Sequential(
            nn.Linear(self.scaf_dense_dim, self.d_latent),
            nn.LayerNorm(self.d_latent),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(self.d_latent, self.d_latent),
            nn.LayerNorm(self.d_latent),
            nn.SiLU(),
        )
        self.chem_shared_ecfp_enc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(self.ecfp_sub_dim, d_ecfp_br),
            nn.LayerNorm(d_ecfp_br),
            nn.SiLU(),
        )
        self.chem_scaffold_fuse = nn.Linear(self.d_latent + d_ecfp_br, self.d_latent)

        mech_in_dim = self.d_latent + 2 * self.n_sar_targets + self.chem_desc_dim
        self.chem_mech_head = nn.Linear(mech_in_dim, 2 * len(CHEM_14_TARGETS) + 4)
        self.desc_apop_head = nn.Linear(self.chem_desc_dim + 2 * self.n_sar_targets, 1)
        nn.init.zeros_(self.desc_apop_head.weight)
        nn.init.zeros_(self.desc_apop_head.bias)

        # Learnable SMARTS warhead / pharmacophore pEC50 gains (initialized modest ~0.45-0.85 pEC50 units; zero saturation cliff!)
        self.raw_motif_gains = nn.Parameter(
            torch.tensor([-0.25, -0.10, -0.60, -0.50, -0.45, -0.50, 0.55, 0.25], dtype=torch.float32)
        )

        # Cell-state bioactivation / efflux FiLM modulation on drug target pEC50 (`dose == 0 -> occ == 0` preserved!)
        self.cell_bioact_film = nn.Sequential(
            nn.Linear(self.d_cell + self.d_latent, self.d_latent),
            nn.SiLU(),
            nn.Linear(self.d_latent, len(CHEM_14_TARGETS)),
        )
        nn.init.zeros_(self.cell_bioact_film[-1].weight)
        nn.init.zeros_(self.cell_bioact_film[-1].bias)

        self.dose_proj = nn.Sequential(
            nn.Linear(mech_in_dim + 3, self.d_latent),
            nn.LayerNorm(self.d_latent),
            nn.SiLU(),
            nn.Linear(self.d_latent, self.d_latent),
        )

        self.register_buffer(
            "cysdb_target_prior",
            torch.ones(len(CHEM_14_TARGETS), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "media_defaults",
            torch.tensor(MEDIA_DEFAULTS, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def snapshot_stage05_sar_anchor(self) -> None:
        """Freeze a clean copy of the Stage-0.5 SAR encoder so cellular fine-tuning never forgets unseen chemotypes."""
        if self.ablate_chem_prior:
            self.has_sar_snapshot.zero_()
            return
        self.sar_anchor_fp_proj.load_state_dict(self.chem_fp_proj.state_dict())
        self.sar_anchor_desc_proj.load_state_dict(self.chem_desc_proj.state_dict())
        self.sar_anchor_in_proj.load_state_dict(self.chem_in_proj.state_dict())
        self.sar_anchor_blocks.load_state_dict(self.chem_blocks.state_dict())
        self.sar_anchor_out_norm.load_state_dict(self.chem_out_norm.state_dict())
        self.sar_anchor_head.load_state_dict(self.sar_pretrain_head.state_dict())
        for mod in (
            self.sar_anchor_fp_proj,
            self.sar_anchor_desc_proj,
            self.sar_anchor_in_proj,
            self.sar_anchor_blocks,
            self.sar_anchor_out_norm,
            self.sar_anchor_head,
        ):
            mod.requires_grad_(False)
            mod.eval()
        self.has_sar_snapshot.fill_(1)

    @torch.no_grad()
    def compute_frozen_sar_vector(self, fp_aligned: torch.Tensor, desc_aligned: torch.Tensor) -> torch.Tensor:
        """Return `[B, sar_snap_dim]` frozen Stage-0.5 SAR representation (`0` when SAR/ChemPrior is ablated)."""
        bsz = fp_aligned.shape[0]
        if self.ablate_chem_prior or int(self.has_sar_snapshot.item()) == 0:
            return torch.zeros(bsz, self.sar_snap_dim, device=fp_aligned.device, dtype=fp_aligned.dtype)
        h_s = self.sar_anchor_in_proj(
            torch.cat([self.sar_anchor_fp_proj(fp_aligned), self.sar_anchor_desc_proj(desc_aligned)], dim=-1)
        )
        for blk in self.sar_anchor_blocks:
            h_s = blk(h_s)
        h_s = self.sar_anchor_out_norm(h_s)
        sar_r = self.sar_anchor_head(h_s)
        sar_sig = torch.sigmoid(sar_r)
        return torch.cat([sar_sig, 0.25 * h_s], dim=-1)

    def build_raw_chem_dense_features(
        self,
        fp_aligned: torch.Tensor,
        desc_aligned: torch.Tensor,
        dose_uM: torch.Tensor,
        dose_log10: torch.Tensor,
    ) -> torch.Tensor:
        """Construct `[B, scaf_dense_dim]` scaffold-generalizing chemical + SAR + dose vector."""
        bsz = fp_aligned.shape[0]
        if self.ablate_chem_prior or self.ablate_sec46_gate:
            maccs = torch.zeros(bsz, self.maccs_sub_dim, device=fp_aligned.device, dtype=fp_aligned.dtype)
            sar_vec = torch.zeros(bsz, self.sar_snap_dim, device=fp_aligned.device, dtype=fp_aligned.dtype)
        else:
            maccs = (
                fp_aligned[:, 3072 : 3072 + self.maccs_sub_dim]
                if self.maccs_sub_dim > 0 and fp_aligned.shape[-1] > 3072
                else torch.zeros(bsz, self.maccs_sub_dim, device=fp_aligned.device, dtype=fp_aligned.dtype)
            )
            sar_vec = self.compute_frozen_sar_vector(fp_aligned, desc_aligned)
        desc_eff = desc_aligned.clone()
        if self.ablate_sec46_gate and desc_eff.shape[-1] > 9:
            desc_eff[:, 9:] = 0.0
        log1p_d = torch.log1p(dose_uM)
        d_feat4 = torch.stack(
            [
                dose_log10,
                log1p_d,
                dose_uM / (0.5 + dose_uM),
                dose_uM / (5.0 + dose_uM),
            ],
            dim=-1,
        )
        return torch.cat([desc_eff, maccs, sar_vec, d_feat4, desc_eff * log1p_d.unsqueeze(-1)], dim=-1)

    def build_shared_ecfp_features(self, fp_aligned: torch.Tensor) -> torch.Tensor:
        """Construct `[B, ecfp_sub_dim]` shared ECFP bits (`0` when chemical prior or Sec46 warhead gate is ablated)."""
        if self.ablate_chem_prior or self.ablate_sec46_gate:
            return torch.zeros(fp_aligned.shape[0], self.ecfp_sub_dim, device=fp_aligned.device, dtype=fp_aligned.dtype)
        return fp_aligned[:, : self.ecfp_sub_dim] * self.shared_ecfp_mask.to(
            device=fp_aligned.device, dtype=fp_aligned.dtype
        ).unsqueeze(0)

    def encode_chem_latent(self, fp: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        """Encode `[fp, desc]` strictly from pure SMILES features into `[B, d_latent]`."""
        fp_aligned = self._align_vec(fp, self.chem_fp_dim)
        desc_aligned = self._align_vec(desc, self.chem_desc_dim)
        fp_reg = F.dropout(fp_aligned, p=0.28, training=self.training)
        h_fp = self.chem_fp_proj(fp_reg)
        h_desc = self.chem_desc_proj(desc_aligned)
        h = self.chem_in_proj(torch.cat([h_fp, h_desc], dim=-1))
        for blk in self.chem_blocks:
            h = blk(h)
        return self.chem_out_norm(h)

    def _compute_sar_features(
        self,
        h: torch.Tensor,
        fp_aligned: torch.Tensor | None = None,
        desc_aligned: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sar_raw = self.sar_pretrain_head(h)
        if self.ablate_chem_prior:
            return torch.zeros_like(sar_raw), sar_raw
        sar_pot_norm = torch.sigmoid(sar_raw[:, : self.n_sar_targets])
        sar_act_prob = torch.sigmoid(sar_raw[:, self.n_sar_targets :])
        return torch.cat([sar_pot_norm, sar_act_prob], dim=-1), sar_raw

    def predict_sar_targets(self, fp: torch.Tensor, desc: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict Stage-0.5 multi-task biochemical `p_potency` `[B, 21]` and `active_logits` `[B, 21]`.

        Decoupled from whole-cell `chem_mech_head` so Stage-0.5 nanomolar cell-free IC50s inform `sar_feats`
        without forcing whole-cell viability EC50s to copy cell-free enzyme potencies!
        """
        desc_aligned = self._align_vec(desc, self.chem_desc_dim)
        h = self.encode_chem_latent(fp, desc_aligned)
        _, sar_raw = self._compute_sar_features(h)
        pred_potency = 4.0 + 5.5 * torch.sigmoid(sar_raw[:, : self.n_sar_targets])
        active_logits = sar_raw[:, self.n_sar_targets :] + 0.9 * (pred_potency - 5.2)
        return {
            "pred_potency": pred_potency,
            "active_logits": active_logits,
            "h_chem": h,
        }

    def _align_vec(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        clean = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        if clean.ndim == 1:
            clean = clean.unsqueeze(0)
        bsz, f_in = clean.shape
        if f_in < target_dim:
            pad = torch.zeros(bsz, target_dim - f_in, device=clean.device, dtype=clean.dtype)
            return torch.cat([clean, pad], dim=-1)
        if f_in > target_dim:
            return clean[:, :target_dim]
        return clean

    def _encode_single_drug(
        self,
        fp: torch.Tensor | None,
        desc: torch.Tensor | None,
        dose_raw: torch.Tensor | None,
        dose_is_log10: bool,
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
        z_cell: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(occ_14, k_rta, k_chelate, k_apop, k_necro, pec50_14)` strictly from structure + dose."""
        occ, k_rta, k_chelate, k_apop, k_necro, pec50, _, _ = self._encode_single_drug_with_latent(
            fp, desc, dose_raw, dose_is_log10, bsz, device, dtype, z_cell=z_cell
        )
        return occ, k_rta, k_chelate, k_apop, k_necro, pec50

    def _encode_single_drug_with_latent(
        self,
        fp: torch.Tensor | None,
        desc: torch.Tensor | None,
        dose_raw: torch.Tensor | None,
        dose_is_log10: bool,
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
        z_cell: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        zeros_14 = torch.zeros(bsz, len(CHEM_14_TARGETS), device=device, dtype=dtype)
        zeros_1 = torch.zeros(bsz, device=device, dtype=dtype)
        zeros_lat = torch.zeros(bsz, self.d_latent, device=device, dtype=dtype)
        if fp is None and desc is None:
            return zeros_14, zeros_1, zeros_1, zeros_1, zeros_1, zeros_14, zeros_lat, zeros_lat

        fp_aligned = (
            self._align_vec(fp.to(device=device, dtype=dtype), self.chem_fp_dim)
            if fp is not None
            else torch.zeros(bsz, self.chem_fp_dim, device=device, dtype=dtype)
        )
        desc_aligned = (
            self._align_vec(desc.to(device=device, dtype=dtype), self.chem_desc_dim)
            if desc is not None
            else torch.zeros(bsz, self.chem_desc_dim, device=device, dtype=dtype)
        )
        drug_present = ((fp_aligned.abs().sum(dim=-1) + desc_aligned.abs().sum(dim=-1)) > 1e-8).to(dtype=dtype)

        if dose_raw is None:
            dose_uM = drug_present * 1.0
        else:
            d_clean = torch.nan_to_num(dose_raw.to(device=device, dtype=dtype).reshape(bsz), nan=-3.0)
            if dose_is_log10:
                dose_uM = (torch.pow(10.0, d_clean.clamp(-4.0, 3.5)) - 1e-3).clamp(min=0.0)
                dose_uM = dose_uM * drug_present
            else:
                dose_uM = d_clean.clamp(min=0.0) * drug_present

        active = drug_present * (dose_uM > 1e-7).to(dtype=dtype)
        dose_log10_eff = torch.log10((dose_uM + 1e-3).clamp(min=1e-4))

        h = self.encode_chem_latent(fp_aligned, desc_aligned)
        sar_feats, sar_raw = self._compute_sar_features(h, fp_aligned=fp_aligned, desc_aligned=desc_aligned)
        mech_in = torch.cat([h, sar_feats, desc_aligned], dim=-1)
        raw = self.chem_mech_head(mech_in)
        n_t = len(CHEM_14_TARGETS)

        pec50 = 3.8 + 5.5 * torch.sigmoid(raw[:, :n_t] - 1.5)
        hill = 0.8 + 1.6 * torch.sigmoid(raw[:, n_t : 2 * n_t])

        cell_film_shift = zeros_14
        if z_cell is not None and not self.ablate_chem_prior:
            zc_al = self._align_vec(z_cell.to(device=device, dtype=dtype), self.d_cell)
            cell_film_shift = 0.85 * torch.tanh(self.cell_bioact_film(torch.cat([zc_al, h], dim=-1)))

        has_desc_vals = (desc_aligned.abs().sum(dim=-1) > 1e-6).to(dtype=dtype) if desc is not None else zeros_1
        lip_perm = torch.sigmoid(1.8 * (desc_aligned[:, 1] + 0.25 - 0.35 * desc_aligned[:, 2]))
        perm_gate_drug = 1.0 - has_desc_vals * (1.0 - (0.25 + 0.75 * lip_perm))

        if self.ablate_chem_prior:
            occ = torch.sigmoid(raw[:, :n_t] - 2.0) * active.unsqueeze(-1) * torch.tanh(0.2 * dose_uM.unsqueeze(-1))
            k_rta = F.softplus(raw[:, 2 * n_t] - 2.5) * torch.log1p(dose_uM) * active
            k_chelate = F.softplus(raw[:, 2 * n_t + 1] - 2.5) * torch.log1p(dose_uM) * active
        else:
            motif_boost = torch.zeros_like(pec50)
            if desc is not None:
                mg = F.softplus(self.raw_motif_gains).to(device=device, dtype=dtype)
                # Lipophilicity membrane permeability gate (`desc[:, 1] = logP / 5.0`)
                perm_gate = 0.50 + 0.50 * torch.sigmoid(2.0 * (desc_aligned[:, 1] + 0.15))

                cov_w = desc_aligned[:, 6].clamp(min=0.0)
                endop = desc_aligned[:, 9].clamp(min=0.0)
                statin_st = desc_aligned[:, 10].clamp(min=0.0) * perm_gate
                thiocu = desc_aligned[:, 11].clamp(min=0.0)
                quinaz = desc_aligned[:, 20].clamp(min=0.0) if desc_aligned.shape[-1] > 20 else zeros_1
                dhodh_m = desc_aligned[:, 21].clamp(min=0.0) if desc_aligned.shape[-1] > 21 else zeros_1
                tzd_m = desc_aligned[:, 22].clamp(min=0.0) if desc_aligned.shape[-1] > 22 else zeros_1
                elec_m = desc_aligned[:, 23].clamp(min=0.0) if desc_aligned.shape[-1] > 23 else zeros_1

                cys_p = self.cysdb_target_prior.to(device=device, dtype=dtype)
                motif_cols = list(motif_boost.unbind(dim=-1))
                motif_cols[TARGET_TO_IDX["GPX4_cov"]] = mg[0] * cov_w * cys_p[0]
                motif_cols[TARGET_TO_IDX["TXNRD1"]] = mg[1] * (0.8 * cov_w + 1.2 * elec_m) * cys_p[11]
                motif_cols[TARGET_TO_IDX["GPX4_deg"]] = mg[2] * endop
                motif_cols[TARGET_TO_IDX["SLC7A11"]] = mg[3] * quinaz
                motif_cols[TARGET_TO_IDX["DHODH"]] = mg[4] * dhodh_m
                motif_cols[TARGET_TO_IDX["ACSL4"]] = mg[5] * tzd_m
                motif_cols[TARGET_TO_IDX["HMGCR"]] = mg[6] * statin_st
                motif_cols[TARGET_TO_IDX["FDX1_Cu"]] = mg[7] * thiocu
                motif_boost = torch.stack(motif_cols, dim=-1)

            pec50_eff = (pec50 + motif_boost + cell_film_shift).clamp(3.6, 9.5)
            ec50_uM = torch.pow(10.0, (6.0 - pec50_eff).clamp(-3.5, 2.5))
            dose_exp = (dose_uM.unsqueeze(-1).clamp(min=1e-9)).pow(hill)
            ec50_exp = ec50_uM.pow(hill)
            occ = (dose_exp / (ec50_exp + dose_exp)) * active.unsqueeze(-1)
            # Gate Cu-ionophore occupancy by bidentate thiocarbonyl coordination + Stage-0.5 SAR channel
            cu_motif = desc_aligned[:, 11].clamp(min=0.0) if desc is not None else zeros_1
            p_is_cu = torch.sigmoid(raw[:, TARGET_TO_IDX["FDX1_Cu"]] - 3.2 + 5.0 * cu_motif + 1.2 * sar_feats[:, 12])
            occ_cols = list(occ.unbind(dim=-1))
            occ_cols[TARGET_TO_IDX["FDX1_Cu"]] = occ_cols[TARGET_TO_IDX["FDX1_Cu"]] * p_is_cu
            occ = torch.stack(occ_cols, dim=-1)

            rta_motif = desc_aligned[:, 7].clamp(min=0.0) if desc is not None else zeros_1
            chel_motif = desc_aligned[:, 8].clamp(min=0.0) if desc is not None else zeros_1
            sar_rescuer_logit = sar_raw[:, 2 * self.n_sar_targets - 1].clamp(-4.0, 4.0)
            p_is_rta = torch.sigmoid(raw[:, 2 * n_t] - 3.0 + 3.0 * rta_motif + 0.65 * sar_rescuer_logit)
            p_is_chel = torch.sigmoid(raw[:, 2 * n_t + 1] - 3.0 + 3.2 * chel_motif)
            k_rta = p_is_rta * F.softplus(raw[:, 2 * n_t]) * torch.log1p(dose_uM) * active
            k_chelate = p_is_chel * F.softplus(raw[:, 2 * n_t + 1]) * torch.log1p(dose_uM) * active

        apop_desc_logit = self.desc_apop_head(torch.cat([desc_aligned, sar_feats.detach()], dim=-1)).squeeze(-1)
        k_apop = F.softplus(raw[:, 2 * n_t + 2] + apop_desc_logit - 1.2) * torch.log1p(dose_uM) * active
        k_necro = F.softplus(raw[:, 2 * n_t + 3] - 1.4) * torch.log1p(dose_uM) * active

        dose_feat = torch.stack(
            [
                torch.log1p(dose_uM),
                torch.tanh(0.45 * torch.log10(dose_uM.clamp(min=1e-4))),
                (dose_uM / (1.0 + dose_uM)),
            ],
            dim=-1,
        )
        raw_cdense = self.build_raw_chem_dense_features(fp_aligned, desc_aligned, dose_uM, dose_log10_eff)
        c_mean = self.chem_dense_mean.to(device=device, dtype=dtype).unsqueeze(0)
        c_scale = self.chem_dense_scale.to(device=device, dtype=dtype).clamp(min=2e-2).unsqueeze(0)
        cdense_std = ((raw_cdense - c_mean) / c_scale).clamp(-8.0, 8.0)
        z_scaf_inv = self.chem_scaffold_inv_enc(cdense_std)

        ecfp_sub = self.build_shared_ecfp_features(fp_aligned)
        z_ecfp_sh = self.chem_shared_ecfp_enc(ecfp_sub)
        z_chem_scaf = F.silu(self.chem_scaffold_fuse(torch.cat([z_scaf_inv, z_ecfp_sh], dim=-1))) * active.unsqueeze(-1)

        z_chem_dosed = (h + self.dose_proj(torch.cat([mech_in, dose_feat], dim=-1))) * active.unsqueeze(-1)
        return occ, k_rta, k_chelate, k_apop, k_necro, pec50, z_chem_dosed, z_chem_scaf

    @staticmethod
    def _hill_scalar(conc: torch.Tensor, ec50: float = 1.0) -> torch.Tensor:
        c = conc.clamp(min=0.0)
        return torch.where(c > 0.0, c / (ec50 + c), torch.zeros_like(c))

    @staticmethod
    def _extract_1d(
        batch: dict[str, Any],
        keys: tuple[str, ...],
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
        default: float = 0.0,
    ) -> torch.Tensor:
        for k in keys:
            if k in batch and batch[k] is not None:
                val = batch[k]
                if not isinstance(val, torch.Tensor):
                    val = torch.as_tensor(val, device=device, dtype=dtype)
                else:
                    val = val.to(device=device, dtype=dtype)
                val = torch.nan_to_num(val, nan=default, posinf=100.0, neginf=-100.0)
                if val.ndim == 0:
                    return val.expand(bsz)
                return val.reshape(bsz)
        return torch.full((bsz,), default, device=device, dtype=dtype)

    @staticmethod
    def _group_min_mult(gko: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        g_dim = gko.shape[-1]
        valid = [i for i in indices if 0 <= i < g_dim]
        if not valid:
            return torch.ones(gko.shape[0], device=gko.device, dtype=gko.dtype)
        return gko[:, valid].min(dim=-1).values

    def forward(
        self,
        batch: dict[str, Any],
        *,
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
        p_hat: torch.Tensor | None = None,
        z_cell: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract and compose all chemical, genetic, media, and physical NP perturbations."""
        fp_a = next(
            (batch[k] for k in ("chem_fp_a", "drug_fp", "mol_fp", "chem_fp", "u_mol") if k in batch and isinstance(batch[k], torch.Tensor)),
            None,
        )
        fp_b = next(
            (batch[k] for k in ("chem_fp_b", "drug_b_fp", "mol_b_fp", "chem_b_fp") if k in batch and isinstance(batch[k], torch.Tensor)),
            None,
        )
        desc_a = batch.get("chem_desc_a") if isinstance(batch.get("chem_desc_a"), torch.Tensor) else None
        desc_b = batch.get("chem_desc_b") if isinstance(batch.get("chem_desc_b"), torch.Tensor) else None

        dose_a_is_log10 = "dose_a" in batch
        dose_b_is_log10 = "dose_b" in batch
        dose_a = next((batch[k] for k in ("dose_a", "dose", "drug_dose", "conc_uM", "c_dose") if k in batch and batch[k] is not None), None)
        dose_b = next((batch[k] for k in ("dose_b", "drug_b_dose", "conc_b_uM") if k in batch and batch[k] is not None), None)

        if dose_a is not None and not isinstance(dose_a, torch.Tensor):
            dose_a = torch.as_tensor(dose_a, device=device, dtype=dtype)
        if dose_b is not None and not isinstance(dose_b, torch.Tensor):
            dose_b = torch.as_tensor(dose_b, device=device, dtype=dtype)

        occ_a, rta_a, chel_a, apop_a, necro_a, pec50_a, z_chem_a, z_scaf_a = self._encode_single_drug_with_latent(
            fp_a, desc_a, dose_a, dose_a_is_log10, bsz, device, dtype, z_cell=z_cell
        )
        occ_b, rta_b, chel_b, apop_b, necro_b, _, z_chem_b, z_scaf_b = self._encode_single_drug_with_latent(
            fp_b, desc_b, dose_b, dose_b_is_log10, bsz, device, dtype, z_cell=z_cell
        )

        target_occ = 1.0 - (1.0 - occ_a) * (1.0 - occ_b)
        bliss_interaction = (occ_a * occ_b).sum(dim=-1)
        k_rta_drug = rta_a + rta_b
        k_chelate_drug = chel_a + chel_b
        k_apop_drug = apop_a + apop_b
        k_necro_drug = necro_a + necro_b
        z_chem_active = z_chem_a + z_chem_b
        z_chem_scaf = z_scaf_a + z_scaf_b

        rta_conc = self._extract_1d(batch, ("rta_conc", "liproxstatin_1", "ferrostatin_1"), bsz, device, dtype, 0.0).clamp(min=0.0)
        chelator_conc = self._extract_1d(batch, ("chelator_conc", "dfo_conc", "cpx_conc"), bsz, device, dtype, 0.0).clamp(min=0.0)
        gpx4_inhib = self._extract_1d(batch, ("gpx4_inhib", "rsl3_conc", "ml210_conc"), bsz, device, dtype, 0.0).clamp(min=0.0)
        fin56_conc = self._extract_1d(batch, ("fin56_conc",), bsz, device, dtype, 0.0).clamp(min=0.0)
        slc7a11_inhib = self._extract_1d(batch, ("xc_inhib", "slc7a11_inhib", "erastin_conc", "ike_conc"), bsz, device, dtype, 0.0).clamp(min=0.0)
        fsp1_inhib = self._extract_1d(batch, ("fsp1_inhib", "ifsp1_conc"), bsz, device, dtype, 0.0).clamp(min=0.0)
        dhodh_inhib = self._extract_1d(batch, ("dhodh_inhib", "brequinar_conc"), bsz, device, dtype, 0.0).clamp(min=0.0)
        statin_conc = self._extract_1d(batch, ("statin_conc", "hmgcr_inhib"), bsz, device, dtype, 0.0).clamp(min=0.0)
        dhcr7_inhib = self._extract_1d(batch, ("dhcr7_inhib", "ay9944_conc"), bsz, device, dtype, 0.0).clamp(min=0.0)
        concanamycin_a = self._extract_1d(batch, ("concanamycin_a", "bafilomycin_a1", "vatpase_inhib"), bsz, device, dtype, 0.0).clamp(min=0.0)
        elesclomol_conc = self._extract_1d(batch, ("elesclomol_conc", "cu_ionophore"), bsz, device, dtype, 0.0).clamp(min=0.0)
        zvad_conc = self._extract_1d(batch, ("zvad_conc", "caspase_inhib"), bsz, device, dtype, 0.0).clamp(min=0.0)

        k_rta_drug = k_rta_drug + 1.8 * rta_conc
        k_chelate_drug = k_chelate_drug + 1.6 * chelator_conc

        explicit_occ = torch.stack(
            [
                self._hill_scalar(gpx4_inhib, ec50=1.0),
                self._hill_scalar(fin56_conc, ec50=1.0),
                self._hill_scalar(slc7a11_inhib, ec50=1.5),
                self._hill_scalar(fsp1_inhib, ec50=1.0),
                self._hill_scalar(dhodh_inhib, ec50=1.5),
                torch.zeros(bsz, device=device, dtype=dtype),
                torch.zeros(bsz, device=device, dtype=dtype),
                self._hill_scalar(statin_conc, ec50=1.5),
                self._hill_scalar(dhcr7_inhib, ec50=1.0),
                torch.zeros(bsz, device=device, dtype=dtype),
                torch.zeros(bsz, device=device, dtype=dtype),
                torch.zeros(bsz, device=device, dtype=dtype),
                self._hill_scalar(concanamycin_a, ec50=0.8),
                self._hill_scalar(elesclomol_conc, ec50=0.5),
            ],
            dim=-1,
        )
        target_occ = 1.0 - (1.0 - target_occ) * (1.0 - explicit_occ)

        # 2. Full-Vocabulary & Canonical Pathway Genetic Interventions
        gene_mult = torch.ones(bsz, len(ANCHOR_28_PROTEINS), device=device, dtype=dtype)
        fdx1_mult = self._extract_1d(batch, ("fdx1_mult",), bsz, device, dtype, 1.0).clamp(min=0.0)
        epl_biosynth_mult = torch.ones(bsz, device=device, dtype=dtype)
        dhodh_gene_mult = torch.ones(bsz, device=device, dtype=dtype)
        repair_shield_mult = torch.ones(bsz, device=device, dtype=dtype)
        vatpase_gene_mult = torch.ones(bsz, device=device, dtype=dtype)
        gene_ko_full: torch.Tensor | None = None

        if "gene_ko_mult" in batch and isinstance(batch["gene_ko_mult"], torch.Tensor):
            gko = torch.nan_to_num(batch["gene_ko_mult"].to(device=device, dtype=dtype), nan=1.0).clamp(0.0, 10.0)
            if gko.ndim == 1:
                gko = gko.unsqueeze(0).expand(bsz, -1)
            gene_ko_full = gko
            g_dim = gko.shape[-1]
            idx_28 = torch.tensor([i % g_dim for i in DEFAULT_ANCHOR_28_VOCAB_IDX], device=device, dtype=torch.long)
            gene_mult = gene_mult * gko[:, idx_28]
            if g_dim > FDX1_VOCAB_IDX:
                fdx1_mult = fdx1_mult * gko[:, FDX1_VOCAB_IDX]

            # Canonical multi-step pathway regulatory stoichiometry (identically 1.0 when gko == 1.0!)
            if g_dim >= 200:
                gm_cols = list(gene_mult.unbind(dim=-1))
                # KEAP1 / NFE2L2 / BACH1 master antioxidant axis
                ko_keap1 = (1.0 - gko[:, KEAP1_VOCAB_IDX]).clamp(min=0.0) if KEAP1_VOCAB_IDX < g_dim else 0.0
                ko_nrf2 = (1.0 - gko[:, NFE2L2_VOCAB_IDX]).clamp(min=0.0) if NFE2L2_VOCAB_IDX < g_dim else 0.0
                ko_bach1 = (1.0 - gko[:, BACH1_VOCAB_IDX]).clamp(min=0.0) if BACH1_VOCAB_IDX < g_dim else 0.0
                nrf2_drive = (1.0 + 1.50 * ko_keap1 + 0.70 * ko_bach1) * (1.0 - 0.65 * ko_nrf2).clamp(min=0.15)
                for p_nrf in ("SLC7A11", "SLC3A2", "GCLC", "GSS", "GSR", "FTH1", "FTL", "TXNRD1", "AIFM2"):
                    gm_cols[PROTEIN_TO_IDX[p_nrf]] = gm_cols[PROTEIN_TO_IDX[p_nrf]] * nrf2_drive

                # Peroxisomal ether-lipid synthesis (FAR1/AGPS/GNPAT/PEX3/PEX10/PEX12 -> Zou et al., Nature 2020)
                epl_biosynth_mult = self._group_min_mult(gko, EPL_BIOSYNTH_IDX)
                pufa_desat = self._group_min_mult(gko, PUFA_DESAT_IDX)
                gm_cols[PROTEIN_TO_IDX["ACSL4"]] = gm_cols[PROTEIN_TO_IDX["ACSL4"]] * (0.40 + 0.60 * pufa_desat) * (0.45 + 0.55 * epl_biosynth_mult)

                # MUFA-PL protection (SCD / MBOAT1 / MBOAT2 -> Liang et al., Cell 2023)
                mufa_part = self._group_min_mult(gko, MUFA_SHIELD_IDX)
                gm_cols[PROTEIN_TO_IDX["ACSL3"]] = gm_cols[PROTEIN_TO_IDX["ACSL3"]] * mufa_part

                # Distal sterol, mevalonate, CoQ10/VK, BH4, GCLM, Sec46 & POR partner enzymes
                gm_cols[PROTEIN_TO_IDX["EBP"]] = gm_cols[PROTEIN_TO_IDX["EBP"]] * self._group_min_mult(gko, STEROL_UP_PARTNER_IDX)
                gm_cols[PROTEIN_TO_IDX["HMGCR"]] = gm_cols[PROTEIN_TO_IDX["HMGCR"]] * self._group_min_mult(gko, MEVALONATE_PARTNER_IDX)
                coq_vk_m = self._group_min_mult(gko, COQ_VK_PARTNER_IDX)
                gm_cols[PROTEIN_TO_IDX["AIFM2"]] = gm_cols[PROTEIN_TO_IDX["AIFM2"]] * (0.25 + 0.75 * coq_vk_m)
                gm_cols[PROTEIN_TO_IDX["GCH1"]] = gm_cols[PROTEIN_TO_IDX["GCH1"]] * self._group_min_mult(gko, BH4_PARTNER_IDX)
                gm_cols[PROTEIN_TO_IDX["GCLC"]] = gm_cols[PROTEIN_TO_IDX["GCLC"]] * self._group_min_mult(gko, GCLM_PARTNER_IDX)
                sec_p = self._group_min_mult(gko, SEC_PARTNER_IDX)
                gm_cols[PROTEIN_TO_IDX["PSTK"]] = gm_cols[PROTEIN_TO_IDX["PSTK"]] * sec_p
                gm_cols[PROTEIN_TO_IDX["POR"]] = gm_cols[PROTEIN_TO_IDX["POR"]] * (0.35 + 0.65 * self._group_min_mult(gko, POR_PARTNER_IDX))

                # Endolysosomal Fe3+ -> Fe2+ import vs SLC40A1/CISD1/NFS1 export/Fe-S buffering
                gm_cols[PROTEIN_TO_IDX["TFRC"]] = gm_cols[PROTEIN_TO_IDX["TFRC"]] * (0.35 + 0.65 * self._group_min_mult(gko, FE_IMPORT_PARTNER_IDX))
                fe_exp_m = self._group_min_mult(gko, FE_EXPORT_FES_IDX)
                gm_cols[PROTEIN_TO_IDX["FTH1"]] = gm_cols[PROTEIN_TO_IDX["FTH1"]] * fe_exp_m

                gene_mult = torch.stack(gm_cols, dim=-1)

                if 0 <= DHODH_VOCAB_IDX < g_dim:
                    dhodh_gene_mult = gko[:, DHODH_VOCAB_IDX] * (0.30 + 0.70 * coq_vk_m)
                repair_shield_mult = self._group_min_mult(gko, REPAIR_SHIELD_IDX)
                vatpase_gene_mult = self._group_min_mult(gko, VATPASE_SUBUNIT_IDX)
                fdx1_mult = fdx1_mult * self._group_min_mult(gko, CUPRO_LIPOYL_IDX)

        for key_cand in ("u_gene", "gene_mult_28", "anchor_gene_mult", "crispr_mult"):
            if key_cand in batch and isinstance(batch[key_cand], torch.Tensor):
                gm = torch.nan_to_num(batch[key_cand].to(device=device, dtype=dtype), nan=1.0).clamp(0.0, 10.0)
                if gm.shape[-1] == len(ANCHOR_28_PROTEINS):
                    gene_mult = gene_mult * gm

        named_mults = {
            "GPX4": ("gpx4_mult",),
            "AIFM2": ("fsp1_mult", "aifm2_mult"),
            "ACSL4": ("acsl4_mult",),
            "ACSL3": ("acsl3_mult",),
            "SLC7A11": ("slc7a11_mult",),
            "DHCR7": ("dhcr7_mult",),
            "EBP": ("ebp_mult",),
            "SQLE": ("sqle_mult",),
            "HMGCR": ("hmgcr_mult",),
            "PSTK": ("pstk_mult",),
            "SEPSECS": ("sepsecs_mult",),
            "NCOA4": ("ncoa4_mult",),
            "TFRC": ("tfrc_mult",),
        }
        gene_mult_cols = list(gene_mult.unbind(dim=-1))
        for prot_name, batch_keys in named_mults.items():
            idx = PROTEIN_TO_IDX[prot_name]
            m_val = self._extract_1d(batch, batch_keys, bsz, device, dtype, 1.0).clamp(min=0.0)
            gene_mult_cols[idx] = gene_mult_cols[idx] * m_val
        gene_mult = torch.stack(gene_mult_cols, dim=-1)

        # 3. 10-channel extracellular media boundary vector
        media_vec = self.media_defaults.to(device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1).clone()
        for m_key in ("media_ext", "media_vec", "b_ext", "u_media", "media_conditions"):
            if m_key in batch and isinstance(batch[m_key], torch.Tensor):
                mv = torch.nan_to_num(batch[m_key].to(device=device, dtype=dtype), nan=1.0).clamp(min=0.0)
                if mv.ndim == 1:
                    mv = mv.unsqueeze(0).expand(bsz, -1)
                c_copy = min(mv.shape[-1], len(MEDIA_10_CHANNELS))
                media_vec[:, :c_copy] = mv[:, :c_copy]

        media_overrides = [
            (0, ("cys2_media", "cystine_media")),
            (1, ("gln_media", "glutamine_media")),
            (2, ("glc_media", "glucose_media")),
            (3, ("se_media", "selenium_media")),
            (4, ("fe3_media", "iron_media")),
            (5, ("oa_media", "mufa_media")),
            (6, ("aa_media", "pufa_media")),
            (7, ("vk_ve_media", "vitk_media")),
            (8, ("po2_media", "oxygen_media")),
            (9, ("ifng_media", "ifn_gamma")),
        ]
        media_cols = list(media_vec.unbind(dim=-1))
        for col_i, keys_i in media_overrides:
            for k in keys_i:
                if k in batch and batch[k] is not None:
                    media_cols[col_i] = self._extract_1d(batch, (k,), bsz, device, dtype, MEDIA_DEFAULTS[col_i]).clamp(min=0.0)
        media_vec = torch.stack(media_cols, dim=-1)

        ifng_suppression = 1.0 / (1.0 + 1.2 * media_vec[:, 9])
        gene_mult_cols = list(gene_mult.unbind(dim=-1))
        gene_mult_cols[PROTEIN_TO_IDX["SLC7A11"]] = gene_mult_cols[PROTEIN_TO_IDX["SLC7A11"]] * ifng_suppression
        gene_mult = torch.stack(gene_mult_cols, dim=-1)

        # 4. Physical C' dot & IONP iron-scavenging / lysosomal delivery
        np_tensor = next(
            (batch[k] for k in ("np_params", "u_np", "nanoparticle_params") if k in batch and isinstance(batch[k], torch.Tensor)),
            None,
        )
        raw_gx = batch.get("gene_expr") if isinstance(batch.get("gene_expr"), torch.Tensor) else None
        np_out = parameter_free_nanoparticle_biophysics(
            np_tensor,
            cys2_media=media_vec[:, 0],
            fe3_media=media_vec[:, 4],
            v_atpase_occ=target_occ[:, TARGET_TO_IDX["V_ATPase"]],
            p_hat=p_hat,
            gene_expr=raw_gx.to(device=device, dtype=dtype) if raw_gx is not None else None,
        )
        np_out["v_atpase_gate"] = (np_out["v_atpase_gate"] * (0.10 + 0.90 * vatpase_gene_mult)).clamp(min=0.02)

        dt = self._extract_1d(batch, ("delta_t", "dt", "exposure_hours", "time_h"), bsz, device, dtype, 24.0).clamp(0.5, 168.0)

        out_dict: dict[str, Any] = {
            "target_occ": target_occ,
            "pred_pec50": pec50_a,
            "z_chem_active": z_chem_active,
            "z_chem_scaf": z_chem_scaf,
            "bliss_interaction": bliss_interaction,
            "k_rta_drug": k_rta_drug,
            "k_chelate_drug": k_chelate_drug,
            "k_apop_drug": k_apop_drug,
            "k_necro_drug": k_necro_drug,
            "zvad_conc": zvad_conc,
            "elesclomol_occ": target_occ[:, TARGET_TO_IDX["FDX1_Cu"]],
            "gene_mult": gene_mult,
            "fdx1_mult": fdx1_mult,
            "epl_biosynth_mult": epl_biosynth_mult,
            "dhodh_gene_mult": dhodh_gene_mult,
            "repair_shield_mult": repair_shield_mult,
            "media_vec": media_vec,
            "dt": dt,
            **np_out,
        }
        if gene_ko_full is not None:
            out_dict["gene_ko_full"] = gene_ko_full
        return out_dict


PerturbationOperator = CompositionalPerturbationOperator

__all__ = [
    "CHEM_14_TARGETS",
    "TARGET_TO_IDX",
    "MEDIA_10_CHANNELS",
    "MEDIA_DEFAULTS",
    "DEFAULT_ANCHOR_28_VOCAB_IDX",
    "FDX1_VOCAB_IDX",
    "parameter_free_nanoparticle_biophysics",
    "CompositionalPerturbationOperator",
    "PerturbationOperator",
]
