"""Sec46 Selenoprotein Translational Readthrough & Membrane-Docking Gate (`Sec46Gate`).

Resolves a foundational failure mode of transcriptomic ferroptosis models: *GPX4* is a Sec46
selenoprotein whose UGA stop-codon recoding via the 3'-UTR SECIS stem-loop strictly depends on:
1. Selenocysteine synthesis & elongation machinery (`PSTK`, `SEPSECS`, `SEPHS2`, `EEFSEC`,
   `SECISBP2`, and `LRP8`/`PRDX6` selenium delivery).
2. Mevalonate-derived isopentenyl pyrophosphate (`IPP` via `HMGCR`/`TRIT1`) required for i6A37
   modification of tRNA^[Ser]Sec (explaining statin & FIN56 translational sensitization even at
   elevated *GPX4* mRNA).
3. Extracellular selenium bioavailability (`[Se]`).
4. GPX4 cationic fin-loop membrane docking efficiency & absence of covalent warhead alkylation
   (`RSL3`, `ML210`, `ML162`) or GPX4 protein degradation (`FIN56`).

All translational efficiency weights are parameterized with strictly positive `softplus` gains,
guaranteeing deterministically that depleting `PSTK`/`SEPSECS` or mevalonate `IPP` strictly
lowers active catalytic `[GPX4_active]` at fixed *GPX4* mRNA expression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SEC_MACHINERY_GENES: tuple[str, ...] = (
    "PSTK",
    "SEPSECS",
    "SEPHS2",
    "EEFSEC",
    "SECISBP2",
    "LRP8",
)
IPP_MEVALONATE_GENES: tuple[str, ...] = ("HMGCR", "SQLE", "TRIT1")


class Sec46Gate(nn.Module):
    """Multiplicative UGA-SECIS selenoprotein translational & covalent warhead gate for GPX4."""

    def __init__(
        self,
        in_features: int = 6,
        ablate_sec46_gate: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.ablate_sec46_gate = bool(ablate_sec46_gate)

        # Strictly positive monotonic gains (via softplus + floor)
        self.raw_w_gpx4 = nn.Parameter(torch.tensor(0.65))
        self.b_gpx4 = nn.Parameter(torch.tensor(0.15))

        self.raw_w_sec = nn.Parameter(torch.tensor(0.85))
        self.b_sec = nn.Parameter(torch.tensor(0.25))

        self.raw_w_ipp = nn.Parameter(torch.tensor(0.80))
        self.b_ipp = nn.Parameter(torch.tensor(0.25))

        self.raw_km_se = nn.Parameter(torch.tensor(-1.20))  # K_m,Se ~ 0.27 normalized units
        self.raw_w_dock = nn.Parameter(torch.tensor(0.50))
        self.b_dock = nn.Parameter(torch.tensor(1.10))

    @staticmethod
    def _reduce_feature(feat: torch.Tensor) -> torch.Tensor:
        """Reduce `[B]` or `[B, K]` tensor cleanly to `[B]` preserving strict monotonicity."""
        if feat.ndim == 1:
            return feat
        if feat.ndim >= 2:
            # Smooth minimum + mean captures rate-limiting bottleneck in PSTK/SEPSECS/SEPHS2/EEFSEC
            mean_val = feat.mean(dim=-1)
            soft_min = -0.5 * torch.logsumexp(-2.0 * feat, dim=-1)
            return 0.5 * mean_val + 0.5 * soft_min
        return feat.reshape(-1)

    def compute_active_gpx4(
        self,
        x: torch.Tensor | None = None,
        *,
        gpx4_expr: torch.Tensor | None = None,
        sec_machinery: torch.Tensor | None = None,
        ipp_level: torch.Tensor | None = None,
        se_media: torch.Tensor | None = None,
        membrane_docking: torch.Tensor | None = None,
        gpx4_covalent_occ: torch.Tensor | None = None,
        gpx4_degrader_occ: torch.Tensor | None = None,
        ablate: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute active catalytic Sec46-GPX4 enzyme pool and intermediate gate efficiencies."""
        if x is not None:
            x_clean = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            if x_clean.ndim == 1:
                x_clean = x_clean.unsqueeze(0)
            if gpx4_expr is None:
                gpx4_expr = x_clean[:, 0]
            if sec_machinery is None and x_clean.shape[-1] >= 3:
                sec_machinery = x_clean[:, 1:-1]
            elif sec_machinery is None and x_clean.shape[-1] == 2:
                sec_machinery = x_clean[:, 1]
            if ipp_level is None and x_clean.shape[-1] >= 3:
                ipp_level = x_clean[:, -1]

        if gpx4_expr is None:
            raise ValueError("Sec46Gate requires either positional tensor `x` or `gpx4_expr`.")

        gpx4_clean = self._reduce_feature(torch.nan_to_num(gpx4_expr, nan=0.0, posinf=10.0, neginf=-10.0))
        device = gpx4_clean.device
        dtype = gpx4_clean.dtype
        bsz = gpx4_clean.shape[0]

        w_gpx4 = F.softplus(self.raw_w_gpx4).to(dtype=dtype) + 0.20
        e_apo = F.softplus(w_gpx4 * gpx4_clean + self.b_gpx4.to(dtype=dtype)) + 1e-4

        is_ablated = self.ablate_sec46_gate if ablate is None else bool(ablate)

        if is_ablated:
            sec_eff = torch.ones(bsz, device=device, dtype=dtype)
            ipp_eff = torch.ones(bsz, device=device, dtype=dtype)
            se_sat = torch.ones(bsz, device=device, dtype=dtype)
            dock_eff = torch.ones(bsz, device=device, dtype=dtype)
        else:
            if sec_machinery is None:
                sec_val = torch.ones(bsz, device=device, dtype=dtype)
            else:
                sec_val = self._reduce_feature(
                    torch.nan_to_num(sec_machinery.to(device=device, dtype=dtype), nan=0.0, posinf=10.0, neginf=-10.0)
                )
            if ipp_level is None:
                ipp_val = torch.ones(bsz, device=device, dtype=dtype)
            else:
                ipp_val = self._reduce_feature(
                    torch.nan_to_num(ipp_level.to(device=device, dtype=dtype), nan=0.0, posinf=10.0, neginf=-10.0)
                )

            w_sec = F.softplus(self.raw_w_sec).to(dtype=dtype) + 0.25
            w_ipp = F.softplus(self.raw_w_ipp).to(dtype=dtype) + 0.25
            sec_eff = torch.sigmoid(w_sec * sec_val + self.b_sec.to(dtype=dtype))
            ipp_eff = torch.sigmoid(w_ipp * ipp_val + self.b_ipp.to(dtype=dtype))

            if se_media is None:
                se_conc = torch.ones(bsz, device=device, dtype=dtype)
            else:
                se_conc = torch.nan_to_num(se_media.to(device=device, dtype=dtype), nan=1.0).clamp(min=1e-4)
            km_se = F.softplus(self.raw_km_se).to(dtype=dtype) + 0.05
            se_sat = se_conc / (km_se + se_conc)

            if membrane_docking is None:
                dock_val = torch.ones(bsz, device=device, dtype=dtype)
            else:
                dock_val = self._reduce_feature(
                    torch.nan_to_num(membrane_docking.to(device=device, dtype=dtype), nan=1.0)
                )
            w_dock = F.softplus(self.raw_w_dock).to(dtype=dtype) + 0.15
            dock_eff = torch.sigmoid(w_dock * dock_val + self.b_dock.to(dtype=dtype))

        # Covalent Sec46 warhead alkylation (RSL3/ML210/ML162) & FIN56 GPX4 degradation
        cov_eff_gain = 0.98
        deg_eff_gain = 0.95
        if gpx4_covalent_occ is None:
            cov_survival = torch.ones(bsz, device=device, dtype=dtype)
        else:
            occ = torch.nan_to_num(gpx4_covalent_occ.to(device=device, dtype=dtype), nan=0.0).clamp(0.0, 0.995)
            cov_survival = 1.0 - cov_eff_gain * occ

        if gpx4_degrader_occ is None:
            deg_survival = torch.ones(bsz, device=device, dtype=dtype)
        else:
            deg = torch.nan_to_num(gpx4_degrader_occ.to(device=device, dtype=dtype), nan=0.0).clamp(0.0, 0.995)
            deg_survival = 1.0 - deg_eff_gain * deg

        e_active = (e_apo * sec_eff * ipp_eff * se_sat * dock_eff * cov_survival * deg_survival).clamp(min=1e-6)

        return {
            "e_gpx4_active": e_active,
            "active_gpx4": e_active,
            "e_gpx4_apo": e_apo,
            "sec_readthrough_eff": sec_eff,
            "ipp_trna_mod_eff": ipp_eff,
            "se_saturation": se_sat,
            "docking_eff": dock_eff,
        }

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        gpx4_expr: torch.Tensor | None = None,
        sec_machinery: torch.Tensor | None = None,
        ipp_level: torch.Tensor | None = None,
        se_media: torch.Tensor | None = None,
        membrane_docking: torch.Tensor | None = None,
        gpx4_covalent_occ: torch.Tensor | None = None,
        gpx4_degrader_occ: torch.Tensor | None = None,
        ablate: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.compute_active_gpx4(
            x,
            gpx4_expr=gpx4_expr,
            sec_machinery=sec_machinery,
            ipp_level=ipp_level,
            se_media=se_media,
            membrane_docking=membrane_docking,
            gpx4_covalent_occ=gpx4_covalent_occ,
            gpx4_degrader_occ=gpx4_degrader_occ,
            ablate=ablate,
        )


Sec46TranslationalGate = Sec46Gate

__all__ = [
    "SEC_MACHINERY_GENES",
    "IPP_MEVALONATE_GENES",
    "Sec46Gate",
    "Sec46TranslationalGate",
]
