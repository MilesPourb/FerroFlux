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
        group_ids: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        huber_val, has_valid = _masked_huber(pred, target, mask=mask, weights=sample_weights, delta=self.huber_delta)
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
        wv = (
            torch.nan_to_num(sample_weights.to(device=pred.device).float().reshape(-1)[m_f], nan=1.0).clamp(0.2, 5.0)
            if isinstance(sample_weights, torch.Tensor) and sample_weights.numel() == p_f.numel()
            else None
        )
        gv = group_ids.to(device=pred.device).reshape(-1)[m_f] if isinstance(group_ids, torch.Tensor) and group_ids.numel() == p_f.numel() else None
        if pv.numel() > 256:
            pv = pv[:256]
            tv = tv[:256]
            if wv is not None:
                wv = wv[:256]
            if gv is not None:
                gv = gv[:256]

        if wv is not None:
            w_sum = wv.sum().clamp(min=1e-4)
            t_mean = (tv * wv).sum() / w_sum
            p_mean = (pv * wv).sum() / w_sum
            w_sqrt = wv.sqrt()
            t_centered = (tv - t_mean) * w_sqrt
            p_centered = (pv - p_mean) * w_sqrt
        else:
            t_centered = tv - tv.mean()
            p_centered = pv - pv.mean()
        t_norm = t_centered.norm()
        if t_norm.item() < 1e-5:
            return huber_val, True

        p_norm = (p_centered.pow(2).sum() + 1e-6).sqrt()
        pearson_r = (p_centered * t_centered).sum() / (p_norm * t_norm + 1e-6)
        corr_loss = (1.0 - pearson_r.clamp(-1.0, 1.0)).to(pred.dtype)

        wg_corr_loss = huber_val.new_zeros(())
        if gv is not None:
            uniq_g, counts_g = torch.unique(gv, return_counts=True)
            valid_g = uniq_g[(counts_g >= 5) & (uniq_g >= 0)]
            if valid_g.numel() > 0:
                wg_terms: list[torch.Tensor] = []
                for gid in valid_g[:16].tolist():
                    mg = gv == gid
                    tg = tv[mg]
                    tgc = tg - tg.mean()
                    tgn = tgc.norm()
                    if tgn.item() > 1e-4:
                        pg = pv[mg]
                        pgc = pg - pg.mean()
                        rg = (pgc * tgc).sum() / ((pgc.pow(2).sum() + 1e-6).sqrt() * tgn + 1e-6)
                        wg_terms.append(1.0 - rg.clamp(-1.0, 1.0))
                if wg_terms:
                    wg_corr_loss = torch.stack(wg_terms).mean().to(pred.dtype)

        diff_t = tv.unsqueeze(1) - tv.unsqueeze(0)
        pair_valid = diff_t.abs() > self.margin_min
        if bool(pair_valid.any().item()):
            diff_p = pv.unsqueeze(1) - pv.unsqueeze(0)
            sign_t = torch.sign(diff_t)
            pair_elem = F.softplus(-self.rank_scale * diff_p * sign_t)
            rank_loss = ((pair_elem * pair_valid.float()).sum() / pair_valid.float().sum().clamp(min=1.0)).to(pred.dtype)
        else:
            rank_loss = huber_val.new_zeros(())

        total = huber_val + self.corr_weight * corr_loss + 0.25 * wg_corr_loss + self.rank_weight * rank_loss
        return total, True


class FerroFluxCurriculumLoss(nn.Module):
    """Unified 4-stage mechanistic + differentiable ranking & APHYNITY curriculum loss (v3)."""

    def __init__(self, huber_delta: float = 1.0, aphynity_lambda: float = 3e-3) -> None:
        super().__init__()
        self.huber_delta = float(huber_delta)
        self.aphynity_lambda = float(aphynity_lambda)
        self.viab_rank_loss = DifferentiableRankAndCorrelationLoss(
            huber_delta=0.20, corr_weight=0.45, rank_weight=0.45, margin_min=0.04, rank_scale=10.0
        )
        self.chronos_rank_loss = DifferentiableRankAndCorrelationLoss(
            huber_delta=0.65, corr_weight=0.50, rank_weight=0.40, margin_min=0.06, rank_scale=5.0
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
        mgm_mask = outputs.get("mgm_eval_mask")

        if mgm_mask is not None:
            mgm_huber, has_mgm = _masked_huber(
                outputs["mgm_recon"],
                gene_expr,
                mask=mgm_mask,
                delta=self.huber_delta,
            )
        else:
            mgm_huber, has_mgm = gene_expr.new_zeros(()), False
        mgm_corr = mgm_huber.new_zeros(())
        if has_mgm and outputs["mgm_recon"].shape[0] >= 4:
            # Across-sample gene co-expression correlation on masked genes (drives Stage-0 pathway reconstruction!)
            rec_f = outputs["mgm_recon"].float()
            tgt_f = gene_expr.to(device=rec_f.device).float()
            if rec_f.shape == tgt_f.shape:
                if isinstance(mgm_mask, torch.Tensor) and mgm_mask.shape == rec_f.shape:
                    m_w = (mgm_mask.to(device=rec_f.device).float() > 0.5).float()
                    col_n = m_w.sum(dim=0, keepdim=True)
                    valid_cols = col_n.squeeze(0) >= 4
                    if bool(valid_cols.any().item()):
                        rm = (rec_f[:, valid_cols] * m_w[:, valid_cols]).sum(dim=0, keepdim=True) / col_n[:, valid_cols]
                        tm = (tgt_f[:, valid_cols] * m_w[:, valid_cols]).sum(dim=0, keepdim=True) / col_n[:, valid_cols]
                        rc = (rec_f[:, valid_cols] - rm) * m_w[:, valid_cols]
                        tc = (tgt_f[:, valid_cols] - tm) * m_w[:, valid_cols]
                        t_std = tc.pow(2).sum(dim=0)
                        act_c = t_std > 1e-4
                        if bool(act_c.any().item()):
                            r_g = (rc[:, act_c] * tc[:, act_c]).sum(dim=0) / (
                                (rc[:, act_c].pow(2).sum(dim=0) + 1e-6).sqrt() * (t_std[act_c] + 1e-6).sqrt()
                            )
                            mgm_corr = (1.0 - r_g.mean()).to(mgm_huber.dtype) * 0.45
                else:
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

        imm_loss = mgm_huber.new_zeros(())
        if "gt_immunogenic_target" in outputs and "immunogenic_pathways" in outputs:
            gt_it = outputs["gt_immunogenic_target"]
            gt_im = outputs.get("gt_immunogenic_mask")
            imm_huber, has_imm = _masked_huber(outputs["immunogenic_pathways"], gt_it, mask=gt_im, delta=0.75)
            imm_corr = mgm_huber.new_zeros(())
            if has_imm and outputs["immunogenic_pathways"].shape[0] >= 6:
                ip_f = outputs["immunogenic_pathways"].float()
                it_f = gt_it.float()
                im_f = (gt_im.float() > 0.5).float() if isinstance(gt_im, torch.Tensor) else torch.ones_like(ip_f)
                corr_list: list[torch.Tensor] = []
                for c_k in range(ip_f.shape[-1]):
                    mk = im_f[:, c_k] > 0.5
                    if int(mk.sum().item()) >= 5:
                        tk = it_f[mk, c_k]
                        tkc = tk - tk.mean()
                        if float(tkc.pow(2).sum().item()) > 1e-4:
                            pk = ip_f[mk, c_k]
                            pkc = pk - pk.mean()
                            rk = (pkc * tkc).sum() / ((pkc.pow(2).sum() + 1e-6).sqrt() * (tkc.pow(2).sum() + 1e-6).sqrt())
                            corr_list.append(1.0 - rk.clamp(-1.0, 1.0))
                if corr_list:
                    imm_corr = 0.35 * torch.stack(corr_list).mean().to(mgm_huber.dtype)
            imm_loss = imm_huber + imm_corr

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
            total = 1.0 * mgm_loss + 0.65 * imm_loss + 1.0 * l_prot + 1.5 * l_metab + 0.1 * flux_reg + phys_reg
            components.update({"prot_anchor_loss": l_prot, "metab_anchor_loss": l_metab, "imm_loss": imm_loss})

        elif stage == 2:
            raw_sw = batch.get("survivorship_weight")
            if bool(outputs.get("ablate_survivorship_flag", False)) or not isinstance(raw_sw, torch.Tensor):
                surv_w = None
            else:
                sw_c = torch.nan_to_num(raw_sw.float(), nan=1.0).clamp(1.0, 6.0)
                surv_w = sw_c / sw_c.mean().clamp(min=1e-4)
            l_chronos, _ = self.chronos_rank_loss(
                outputs["chronos_pred"],
                batch.get("chronos_target"),
                mask=batch.get("chronos_mask"),
                group_ids=batch.get("group_id_int"),
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
                group_ids=batch.get("group_id_int"),
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
                group_ids=batch.get("group_id_int"),
            )
            total = (
                0.15 * mgm_loss
                + 1.6 * l_viab
                + 0.75 * l_haz
                + 0.8 * l_rescue
                + 0.6 * l_syn
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
                group_ids=batch.get("group_id_int"),
            )
            l_chr4, _ = self.chronos_rank_loss(
                outputs["chronos_pred"],
                batch.get("chronos_target"),
                mask=batch.get("chronos_mask"),
                group_ids=batch.get("group_id_int"),
            )
            total = (
                0.45 * mgm_loss
                + 0.65 * imm_loss
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
