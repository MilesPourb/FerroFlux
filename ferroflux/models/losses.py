"""Mechanistic Curriculum + Differentiable Ranking & APHYNITY Loss Suite (`FerroFluxCurriculumLoss`) — v3.

Upgrades in v3 over v2:
- `DifferentiableRankAndCorrelationLoss`: combines calibrated Masked Huber loss (preserving ECE & Brier
  probability calibration) with Centered Cosine / Pearson Correlation Loss (`1 - r(pred, target)`) and
  Pairwise Logistic Margin Ranking Loss (`log(1 + exp(-s * (pred_i - pred_j) * sign(y_i - y_j)))`) across
  valid pairs with `|y_i - y_j| > 0.06`, directly optimizing Spearman `rho` and `AUROC`.
- APHYNITY Residual Norm Penalty (`1e-3 * ||aphynity_logit||_2^2`) forcing white-box `DiffFerroGEM`
  stoichiometric kinetics to explain maximal variance before activating the grey-box residual.
- Gene-wise / cell-wise MGM correlation loss in Stage 1 for cross-pathway transcriptomic reconstruction.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor | None,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    delta: float = 1.0,
) -> tuple[torch.Tensor, bool]:
    """NaN-safe masked Huber loss returning `(loss_scalar, had_valid_entries)`."""
    if target is None:
        return pred.new_zeros(()), False
    tgt = target.to(device=pred.device, dtype=pred.dtype)
    if tgt.shape != pred.shape:
        try:
            tgt = tgt.view_as(pred)
        except RuntimeError:
            min_d = min(pred.shape[-1], tgt.shape[-1])
            pred = pred[..., :min_d]
            tgt = tgt[..., :min_d]

    finite_mask = torch.isfinite(tgt).to(dtype=pred.dtype)
    tgt_clean = torch.nan_to_num(tgt, nan=0.0, posinf=20.0, neginf=-20.0)

    if mask is not None:
        m = torch.nan_to_num(mask.to(device=pred.device, dtype=pred.dtype), nan=0.0).clamp(0.0, 1.0)
        if m.shape != pred.shape:
            while m.ndim < pred.ndim:
                m = m.unsqueeze(-1)
            m = m.expand_as(pred)
        valid = finite_mask * m
    else:
        valid = finite_mask

    elem = F.huber_loss(pred, tgt_clean, reduction="none", delta=delta)
    if weights is not None:
        w = torch.nan_to_num(weights.to(device=pred.device, dtype=pred.dtype), nan=1.0).clamp(0.1, 20.0)
        while w.ndim < elem.ndim:
            w = w.unsqueeze(-1)
        elem = elem * w

    denom = valid.sum()
    if denom.item() < 1e-6:
        return pred.new_zeros(()), False
    return (elem * valid).sum() / denom.clamp(min=1.0), True


def _masked_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor | None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool]:
    if target is None:
        return logits.new_zeros(()), False
    tgt = target.to(device=logits.device, dtype=logits.dtype).reshape_as(logits)
    finite_mask = torch.isfinite(tgt).to(dtype=logits.dtype)
    tgt_clean = torch.nan_to_num(tgt, nan=0.0).clamp(0.0, 1.0)
    if mask is not None:
        m = torch.nan_to_num(mask.to(device=logits.device, dtype=logits.dtype).reshape_as(logits), nan=0.0).clamp(0.0, 1.0)
        valid = finite_mask * m
    else:
        valid = finite_mask
    denom = valid.sum()
    if denom.item() < 1e-6:
        return logits.new_zeros(()), False
    elem = F.binary_cross_entropy_with_logits(logits.float(), tgt_clean.float(), reduction="none").to(logits.dtype)
    return (elem * valid).sum() / denom.clamp(min=1.0), True


class DifferentiableRankAndCorrelationLoss(nn.Module):
    """Joint Huber + Centered Pearson Correlation + Pairwise Logistic Ranking Loss for 1D masked targets."""

    def __init__(
        self,
        huber_delta: float = 0.5,
        corr_weight: float = 0.35,
        rank_weight: float = 0.35,
        margin_min: float = 0.06,
        rank_scale: float = 6.0,
    ) -> None:
        super().__init__()
        self.huber_delta = float(huber_delta)
        self.corr_weight = float(corr_weight)
        self.rank_weight = float(rank_weight)
        self.margin_min = float(margin_min)
        self.rank_scale = float(rank_scale)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        huber_val, has_valid = _masked_huber(pred, target, mask=mask, delta=self.huber_delta)
        if not has_valid or target is None:
            return huber_val, False

        p_f = pred.float().reshape(-1)
        t_f = torch.nan_to_num(target.to(device=pred.device).float().reshape(-1), nan=0.0)
        if mask is not None:
            m_f = (torch.nan_to_num(mask.to(device=pred.device).float().reshape(-1), nan=0.0) > 0.5) & torch.isfinite(target.to(device=pred.device).reshape(-1))
        else:
            m_f = torch.isfinite(target.to(device=pred.device).reshape(-1))

        if int(m_f.sum().item()) < 6:
            return huber_val, True

        pv = p_f[m_f]
        tv = t_f[m_f]
        if pv.numel() > 256:
            pv = pv[:256]
            tv = tv[:256]

        t_centered = tv - tv.mean()
        t_norm = t_centered.norm()
        if t_norm.item() < 1e-5:
            return huber_val, True

        p_centered = pv - pv.mean()
        p_norm = (p_centered.pow(2).sum() + 1e-6).sqrt()
        pearson_r = (p_centered * t_centered).sum() / (p_norm * t_norm + 1e-6)
        corr_loss = (1.0 - pearson_r.clamp(-1.0, 1.0)).to(pred.dtype)

        diff_t = tv.unsqueeze(1) - tv.unsqueeze(0)
        pair_valid = diff_t.abs() > self.margin_min
        if bool(pair_valid.any().item()):
            diff_p = pv.unsqueeze(1) - pv.unsqueeze(0)
            sign_t = torch.sign(diff_t)
            pair_elem = F.softplus(-self.rank_scale * diff_p * sign_t)
            rank_loss = ((pair_elem * pair_valid.float()).sum() / pair_valid.float().sum().clamp(min=1.0)).to(pred.dtype)
        else:
            rank_loss = huber_val.new_zeros(())

        total = huber_val + self.corr_weight * corr_loss + self.rank_weight * rank_loss
        return total, True


class FerroFluxCurriculumLoss(nn.Module):
    """Unified 4-stage mechanistic + differentiable ranking & APHYNITY curriculum loss (v3)."""

    def __init__(self, huber_delta: float = 1.0, aphynity_lambda: float = 1e-3) -> None:
        super().__init__()
        self.huber_delta = float(huber_delta)
        self.aphynity_lambda = float(aphynity_lambda)
        self.viab_rank_loss = DifferentiableRankAndCorrelationLoss(
            huber_delta=0.25, corr_weight=0.35, rank_weight=0.40, margin_min=0.05, rank_scale=8.0
        )
        self.chronos_rank_loss = DifferentiableRankAndCorrelationLoss(
            huber_delta=0.75, corr_weight=0.40, rank_weight=0.35, margin_min=0.08, rank_scale=4.0
        )

    def forward(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        stage: int = 3,
    ) -> dict[str, torch.Tensor]:
        if stage not in (1, 2, 3, 4):
            raise ValueError(f"Curriculum stage must be in (1, 2, 3, 4), got {stage!r}")

        gene_expr = torch.nan_to_num(batch["gene_expr"], nan=0.0, posinf=20.0, neginf=-20.0)
        mgm_mask = outputs.get("mgm_eval_mask", batch.get("gene_mask", batch.get("expr_mask")))

        mgm_huber, has_mgm = _masked_huber(
            outputs["mgm_recon"],
            gene_expr,
            mask=mgm_mask,
            delta=self.huber_delta,
        )
        mgm_corr = mgm_huber.new_zeros(())
        if has_mgm and outputs["mgm_recon"].shape[0] >= 4:
            # Across-sample gene co-expression correlation on masked genes (drives Stage-0 pathway reconstruction!)
            rec_f = outputs["mgm_recon"].float()
            tgt_f = gene_expr.to(device=rec_f.device).float()
            if rec_f.shape == tgt_f.shape:
                rc = rec_f - rec_f.mean(dim=0, keepdim=True)
                tc = tgt_f - tgt_f.mean(dim=0, keepdim=True)
                t_std = tc.pow(2).sum(dim=0)
                active_g = t_std > 1e-4
                if bool(active_g.any().item()):
                    r_g = (rc[:, active_g] * tc[:, active_g]).sum(dim=0) / (
                        (rc[:, active_g].pow(2).sum(dim=0) + 1e-6).sqrt() * (t_std[active_g] + 1e-6).sqrt()
                    )
                    mgm_corr = (1.0 - r_g.mean()).to(mgm_huber.dtype) * 0.35
        mgm_loss = mgm_huber + mgm_corr

        flux_reg = 1e-4 * outputs["fluxes"].pow(2).mean()
        aphynity_reg = (
            self.aphynity_lambda * outputs["aphynity_logit"].pow(2).mean()
            if "aphynity_logit" in outputs and isinstance(outputs["aphynity_logit"], torch.Tensor)
            else mgm_loss.new_zeros(())
        )
        phys_reg = (
            aphynity_reg
            + 1e-3 * (outputs["pl_ooh"].log1p().pow(2).mean() + outputs["seven_dhc_ss"].log1p().pow(2).mean())
            + 1e-6 * outputs["rcd_22_hazards"].pow(2).mean()
            + 1e-6 * outputs["delta_expr_pred"].pow(2).mean()
            + 1e-6 * outputs["immunogenic_pathways"].pow(2).mean()
            + 1e-6 * outputs["chronos_pred"].pow(2).mean()
            + 1e-6 * outputs["synergy_pred"].pow(2).mean()
        )

        components: dict[str, torch.Tensor] = {
            "mgm_loss": mgm_loss,
            "flux_reg": flux_reg,
            "aphynity_reg": aphynity_reg,
        }

        if stage == 1:
            l_prot, _ = _masked_huber(
                outputs["p_hat_log"],
                batch.get("prot_anchor_target", batch.get("proteomics")),
                mask=batch.get("prot_anchor_mask", batch.get("proteomics_mask")),
                delta=self.huber_delta,
            )
            l_metab, _ = _masked_huber(
                torch.log(outputs["metab_anchor_pred"].clamp(min=1e-5)),
                batch.get("metab_anchor_target", batch.get("metabolomics")),
                mask=batch.get("metab_anchor_mask", batch.get("metabolomics_mask")),
                delta=self.huber_delta,
            )
            total = 1.0 * mgm_loss + 1.0 * l_prot + 1.5 * l_metab + 0.1 * flux_reg + phys_reg
            components.update({"prot_anchor_loss": l_prot, "metab_anchor_loss": l_metab})

        elif stage == 2:
            surv_w = batch.get("survivorship_weight")
            l_chronos, _ = self.chronos_rank_loss(
                outputs["chronos_pred"],
                batch.get("chronos_target"),
                mask=batch.get("chronos_mask"),
            )
            l_de, has_de = _masked_huber(
                outputs["delta_expr_pred"],
                batch.get("delta_expr_target"),
                mask=batch.get("delta_expr_mask"),
                weights=surv_w,
                delta=self.huber_delta,
            )
            cos_de = mgm_loss.new_zeros(())
            if has_de and isinstance(batch.get("delta_expr_target"), torch.Tensor):
                tgt_de = torch.nan_to_num(
                    batch["delta_expr_target"].to(device=gene_expr.device, dtype=gene_expr.dtype), nan=0.0
                )
                cos_sim = F.cosine_similarity(outputs["delta_expr_pred"], tgt_de, dim=-1, eps=1e-6)
                cos_de = (1.0 - cos_sim).mean() * 0.25

            l_sec, _ = _masked_huber(
                outputs["sec_readthrough_eff"],
                batch.get("sec46_readthrough_target"),
                mask=batch.get("sec46_mask"),
                delta=0.5,
            )
            total = 0.30 * mgm_loss + 1.2 * l_chronos + 0.8 * (l_de + cos_de) + 1.0 * l_sec + 0.1 * flux_reg + phys_reg
            components.update({"chronos_loss": l_chronos, "delta_expr_loss": l_de, "sec46_loss": l_sec})

        elif stage == 3:
            l_viab, _ = self.viab_rank_loss(
                outputs["viability_pred"],
                batch.get("viability_target", batch.get("viability")),
                mask=batch.get("viability_mask"),
            )
            l_haz, _ = _masked_huber(
                outputs["hazards"],
                batch.get("hazard_target"),
                mask=batch.get("hazard_mask"),
                delta=self.huber_delta,
            )
            l_rescue, _ = _masked_huber(
                outputs["rescue_delta_pred"],
                batch.get("rescue_delta_target"),
                mask=batch.get("rescue_mask"),
                delta=0.5,
            )
            l_syn, _ = _masked_huber(
                outputs["synergy_pred"],
                batch.get("synergy_target"),
                mask=batch.get("synergy_mask"),
                delta=self.huber_delta,
            )
            l_chr3, _ = self.chronos_rank_loss(
                outputs["chronos_pred"],
                batch.get("chronos_target"),
                mask=batch.get("chronos_mask"),
            )
            total = (
                0.15 * mgm_loss
                + 1.5 * l_viab
                + 1.2 * l_haz
                + 2.0 * l_rescue
                + 0.8 * l_syn
                + 1.0 * l_chr3
                + 0.1 * flux_reg
                + phys_reg
            )
            components.update(
                {
                    "viability_loss": l_viab,
                    "hazard_loss": l_haz,
                    "rescue_loss": l_rescue,
                    "synergy_loss": l_syn,
                }
            )

        else:  # stage == 4
            l_oxy, _ = _masked_huber(
                torch.log1p(outputs["oxylipid_pred"].clamp(min=0.0)),
                batch.get("oxylipid_target"),
                mask=batch.get("oxylipid_mask"),
                delta=self.huber_delta,
            )
            l_icb, _ = _masked_bce_logits(
                outputs["icb_response_logit"],
                batch.get("icb_response_target"),
                mask=batch.get("icb_mask"),
            )
            l_viab4, _ = self.viab_rank_loss(
                outputs["viability_pred"],
                batch.get("viability_target"),
                mask=batch.get("viability_mask"),
            )
            l_chr4, _ = self.chronos_rank_loss(
                outputs["chronos_pred"],
                batch.get("chronos_target"),
                mask=batch.get("chronos_mask"),
            )
            total = (
                0.35 * mgm_loss
                + 1.2 * l_oxy
                + 0.8 * l_icb
                + 1.4 * l_viab4
                + 1.0 * l_chr4
                + 0.1 * flux_reg
                + phys_reg
            )
            components.update({"oxylipid_loss": l_oxy, "icb_loss": l_icb, "viability_loss": l_viab4})

        components["loss"] = total
        return components


FerroFluxLoss = FerroFluxCurriculumLoss


def compute_curriculum_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    stage: int = 3,
) -> dict[str, torch.Tensor]:
    return FerroFluxCurriculumLoss()(outputs, batch, stage=stage)


__all__ = [
    "DifferentiableRankAndCorrelationLoss",
    "FerroFluxCurriculumLoss",
    "FerroFluxLoss",
    "compute_curriculum_loss",
]
