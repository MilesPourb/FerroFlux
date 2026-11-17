"""Pillar 3 Forensic Audit: Stage-0 Blind Validation Circularity, mRNA Tautology & Spatial Decomposition."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
import torch

from ferroflux.audit.resource_guard import enforce_login_node_politeness, get_peak_rss_mb
from ferroflux.models.ferroflux_fm import FerroFluxFM
from ferroflux.validation.stage0_bradbury_blind_eval import (
    CGAS_STING_GENES,
    _classify_melanoma_arm,
    _classify_prostate_regimen_tier,
    _parse_geo_series_matrix_metadata,
)

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
LOCKBOX_DIR = ROOT_DIR / "data" / "stage0_blind_lockbox"


def _load_local_geo_cohort(gse: str, g2i: dict[str, int], n_genes: int = 2048) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[dict[str, Any]]]:
    gdir = LOCKBOX_DIR / gse
    meta = pd.DataFrame(_parse_geo_series_matrix_metadata(gdir / f"{gse}_series_matrix.txt.gz"))
    cnt_path = next(gdir.glob("*counts*.gz"))
    df = pd.read_csv(cnt_path, sep="\t", index_col=0)
    for sc in ("gene_name", "gene_symbol", "Symbol", "SYMBOL", "gene"):
        if sc in df.columns:
            syms = df[sc].astype(str).str.strip()
            valid = (syms != "") & (syms.str.lower() != "nan")
            df = df.loc[valid].set_index(syms.loc[valid].str.upper())
            break
    drop_meta = {"start", "end", "length", "entrez_id", "chromosome"}
    keep = [c for c in df.select_dtypes(include=[np.number]).columns if str(c).lower() not in drop_meta]
    num = df[keep]
    vals = num.to_numpy(dtype=np.float64)
    if np.nanmax(vals) > 50.0:
        vals = np.log2(vals / np.maximum(vals.sum(axis=0, keepdims=True), 1.0) * 1e6 + 1.0)
    norm_df = pd.DataFrame(vals, index=num.index.astype(str).str.upper(), columns=[str(c) for c in num.columns]).groupby(level=0).mean()

    # Audit column alignment exact vs positional fallback!
    col_upper = {c.upper(): c for c in norm_df.columns}
    align_audit: list[dict[str, Any]] = []
    aligned_cols: list[str] = []
    for idx_m, mrow in meta.iterrows():
        title_s = str(mrow.get("title", "")).strip()
        gsm_s = str(mrow.get("gsm_id", "")).strip()
        desc_s = str(mrow.get("description", ""))
        tokens = [title_s, gsm_s] + [re.sub(r"(?i)^library\s*name\s*:\s*", "", t).strip() for t in desc_s.split(";") if t.strip()]
        matched = None
        match_method = "exact_token"
        for tok in tokens:
            if tok.upper() in col_upper:
                matched = col_upper[tok.upper()]
                break
        if matched is None:
            sub_hit = next((c for c in norm_df.columns if gsm_s in c or c in title_s), None)
            if sub_hit:
                matched = sub_hit
                match_method = "substring"
            else:
                matched = norm_df.columns[min(int(idx_m), norm_df.shape[1] - 1)]
                match_method = "POSITIONAL_FALLBACK"
        aligned_cols.append(matched)
        align_audit.append({"gsm_id": gsm_s, "title": title_s, "matched_col": matched, "method": match_method})

    sub = norm_df[aligned_cols]
    X = np.zeros((len(meta), n_genes), dtype=np.float32)
    for gsym, rvals in sub.iterrows():
        if gsym in g2i:
            X[:, g2i[gsym]] = rvals.to_numpy(dtype=np.float32)
    return meta, X, sub, align_audit


def _load_ensemble(sub_dir: str, max_seeds: int = 5) -> list[FerroFluxFM]:
    pts = sorted((ROOT_DIR / "checkpoints" / sub_dir).glob("*.pt"))[:max_seeds]
    models = []
    for p in pts:
        pay = torch.load(p, map_location="cpu", weights_only=False)
        m = FerroFluxFM(**pay["model_init_kwargs"])
        m.load_state_dict(pay["model_state_dict"])
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def _ens_pred(models: list[FerroFluxFM], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    acc: dict[str, list[np.ndarray]] = {}
    for m in models:
        out = m(batch)
        for k in ("h_ferro", "seven_dhc_ss", "g_rupture", "pl_ooh", "icb_response_logit", "immunogenic_pathways"):
            acc.setdefault(k, []).append(out[k].float().numpy())
    return {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in acc.items()}


def run_pillar3_audit(output_dir: Path, max_threads: int = 6) -> dict[str, Any]:
    enforce_login_node_politeness(max_threads=max_threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings: list[dict[str, Any]] = []

    vocab = json.loads((ROOT_DIR / "data" / "processed" / "gene_vocab.json").read_text(encoding="utf-8"))
    g2i = {g.upper(): i for i, g in enumerate(vocab["genes"])}

    m_mel, x_mel, sub_mel, al_mel = _load_local_geo_cohort("GSE310954", g2i)
    m_p1, x_p1, sub_p1, al_p1 = _load_local_geo_cohort("GSE319397", g2i)
    m_p2, x_p2, sub_p2, al_p2 = _load_local_geo_cohort("GSE319419", g2i)

    # Check column alignment integrity across all 32 GEO samples
    pos_fallbacks = [x for x in (al_mel + al_p1 + al_p2) if x["method"] == "POSITIONAL_FALLBACK"]
    findings.append(
        {
            "id": "P3-GEO-COL-ALIGNMENT",
            "severity": "MAJOR" if pos_fallbacks else "VERIFIED-CLEAN",
            "title": "Audit of GSM-to-expression-column alignment across GSE310954, GSE319397, and GSE319419",
            "details": {
                "total_samples": len(al_mel) + len(al_p1) + len(al_p2),
                "positional_fallbacks_count": len(pos_fallbacks),
                "sample_alignments_gse319397": al_p1[:5],
                "fallback_rows": pos_fallbacks[:10],
            },
        }
    )

    mel_arms = np.array([_classify_melanoma_arm(f"{r['title']} {r['characteristics']}") for _, r in m_mel.iterrows()])
    sting_genes = [g for g in CGAS_STING_GENES if g in sub_mel.index]
    obs_sting = sub_mel.loc[sting_genes].mean(axis=0).to_numpy(dtype=np.float32)

    pc_meta = pd.concat([m_p1.assign(acc="GSE319397"), m_p2.assign(acc="GSE319419")], ignore_index=True)
    x_pc = np.vstack([x_p1, x_p2])
    pc_tiers = np.array([_classify_prostate_regimen_tier(f"{r['title']} {r['characteristics']}") for _, r in pc_meta.iterrows()])
    st_p1 = sub_p1.loc[[g for g in ("EBP", "SC5D") if g in sub_p1.index]].mean(axis=0).to_numpy() - sub_p1.loc["DHCR7"].to_numpy()

    deconf_rows: list[dict[str, Any]] = []
    for ens_name, sub_dir in [("BASE_academic", "academic"), ("XL_academic", "xl_academic"), ("XL_open", "xl_open")]:
        models = _load_ensemble(sub_dir, max_seeds=5)

        # Cohort 1 Regime A: 100% Unconditioned RNA-Only
        p_mel_uncond = _ens_pred(
            models,
            {
                "gene_expr": torch.from_numpy(x_mel),
                "np_params": torch.zeros((len(x_mel), 5)),
                "cys2_media": torch.ones(len(x_mel)),
            },
        )
        auc_mel = float(roc_auc_score(mel_arms, p_mel_uncond["h_ferro"]))
        rho_mel = float(spearmanr(obs_sting, p_mel_uncond["h_ferro"]).statistic)
        mwu_mel = float(mannwhitneyu(p_mel_uncond["h_ferro"][mel_arms == 1], p_mel_uncond["h_ferro"][mel_arms == 0]).pvalue)

        # Cohort 1 Regime C: Forward Counterfactual from Untreated CT Baseline ONLY (B16-GM vs B16-F10)
        ct_gm_idx = [i for i, r in m_mel.iterrows() if "GM" in str(r["title"]) and mel_arms[i] == 0]
        ct_f10_idx = [i for i, r in m_mel.iterrows() if "F10" in str(r["title"]) and mel_arms[i] == 0]
        cd_gm_idx = [i for i, r in m_mel.iterrows() if "GM" in str(r["title"]) and mel_arms[i] == 1]
        cd_f10_idx = [i for i, r in m_mel.iterrows() if "F10" in str(r["title"]) and mel_arms[i] == 1]
        obs_delta_gm = float(obs_sting[cd_gm_idx].mean() - obs_sting[ct_gm_idx].mean())
        obs_delta_f10 = float(obs_sting[cd_f10_idx].mean() - obs_sting[ct_f10_idx].mean())
        x_ct_only = np.stack([x_mel[ct_gm_idx].mean(axis=0), x_mel[ct_f10_idx].mean(axis=0)], axis=0)
        p_ct_base = _ens_pred(models, {"gene_expr": torch.from_numpy(x_ct_only), "np_params": torch.zeros((2, 5))})
        p_ct_cdot = _ens_pred(
            models,
            {
                "gene_expr": torch.from_numpy(x_ct_only),
                "np_params": torch.tensor([[6.0, -5.0, 1.0, 1.0, 8.0], [6.0, -5.0, 1.0, 1.0, 8.0]]),
                "cys2_media": torch.tensor([0.15, 0.15]),
            },
        )
        pred_delta_gm = float(p_ct_cdot["h_ferro"][0] - p_ct_base["h_ferro"][0])
        pred_delta_f10 = float(p_ct_cdot["h_ferro"][1] - p_ct_base["h_ferro"][1])

        # Cohort 2 Regime A (unconditioned) vs Regime B (EBP, SC5D, DHCR7 masked out in input!)
        p_pc_uncond = _ens_pred(
            models,
            {"gene_expr": torch.from_numpy(x_pc), "np_params": torch.zeros((len(x_pc), 5)), "media_ext": torch.ones((len(x_pc), 10))},
        )
        mask_no_sterol = np.ones_like(x_p1)
        x_p1_masked = x_p1.copy()
        for sg in ("EBP", "SC5D", "DHCR7"):
            if sg in g2i:
                mask_no_sterol[:, g2i[sg]] = 0.0
                x_p1_masked[:, g2i[sg]] = 0.0
        p_p1_masked = _ens_pred(
            models,
            {
                "gene_expr": torch.from_numpy(x_p1_masked),
                "gene_mask": torch.from_numpy(mask_no_sterol),
                "np_params": torch.zeros((len(x_p1), 5)),
            },
        )
        rho_sterol_unmasked = float(spearmanr(st_p1, p_pc_uncond["seven_dhc_ss"][: len(st_p1)]).statistic)
        rho_sterol_masked = float(spearmanr(st_p1, p_p1_masked["seven_dhc_ss"]).statistic)
        rho_tiers_uncond = float(spearmanr(pc_tiers, p_pc_uncond["h_ferro"]).statistic)

        deconf_rows.append(
            {
                "ensemble": ens_name,
                "gse310954_uncond_auroc": round(auc_mel, 4),
                "gse310954_uncond_mwu_p": round(mwu_mel, 4),
                "gse310954_uncond_sting_rho": round(rho_mel, 4),
                "gse310954_fwd_ct_only_gm_gt_f10_match": bool((obs_delta_gm > obs_delta_f10) == (pred_delta_gm > pred_delta_f10)),
                "gse319397_sterol_rho_with_direct_mrna": round(rho_sterol_unmasked, 4),
                "gse319397_sterol_rho_when_ebp_sc5d_dhcr7_masked": round(rho_sterol_masked, 4),
                "prostate_tiers_uncond_rho": round(rho_tiers_uncond, 4),
            }
        )

    # Cohort 3 Zenodo 18271042 Spatial Wave Factor Attribution
    z_zip = LOCKBOX_DIR / "zenodo_18271042" / "CellDeathSpreading-v1.0.0.zip"
    zen_decomp: dict[str, Any] = {}
    if z_zip.is_file():
        xl_m = _load_ensemble("xl_academic", max_seeds=1)[0]
        with zipfile.ZipFile(z_zip) as zf:
            names = zf.namelist()
            spi_entry = next(n for n in names if n.endswith("SPI_calculations.csv"))
            spi_df = pd.read_csv(zf.open(spi_entry))
            well_deg35_map: dict[str, float] = {}
            for nc in names:
                m_w = re.search(r"death_annotations/(sample_\d+)\.csv$", nc)
                if m_w:
                    wxy = pd.read_csv(zf.open(nc), usecols=["cell_x", "cell_y"]).to_numpy(dtype=np.float32)
                    d = np.sqrt(((wxy[:, None, :] - wxy[None, :, :]) ** 2).sum(-1) + 1e-6)
                    np.fill_diagonal(d, 1e9)
                    well_deg35_map[m_w.group(1)] = float(np.exp(-d / 35.0).sum(-1).mean())

            obs_spi, u_means, morans, prop_probs, para_factors = [], [], [], [], []
            base_vec = torch.ones(1, 2048) * 1.4
            for _, srow in spi_df.iterrows():
                exp_fn = str(srow["Experiment_name"])
                ann_e = next(n for n in names if n.endswith(f"death_annotations/{exp_fn}"))
                xy = torch.tensor(pd.read_csv(zf.open(ann_e), usecols=["cell_x", "cell_y"]).to_numpy(dtype=np.float32))
                n_c = xy.shape[0]
                ctr = xy.mean(0, keepdim=True)
                focal = torch.exp(-((xy - ctr).pow(2).sum(-1)) / (2.0 * 35.0**2))
                dij = ((xy[:, None, :] - xy[None, :, :]).pow(2).sum(-1) + 1e-6).sqrt()
                deg35 = float((torch.exp(-dij / 35.0) * (1.0 - torch.eye(n_c))).sum(-1).mean().item())
                m_p = re.search(r"(sample_\d+)", exp_fn)
                w_d = well_deg35_map.get(m_p.group(1), 10.0) if m_p else 10.0
                pf = deg35 / w_d

                with torch.no_grad():
                    wout = xl_m.predict_spatial_wave(
                        base_vec.expand(n_c, 2048),
                        coords=xy,
                        n_steps=14,
                        dt_step=0.5,
                        seed_spots=focal,
                        cys2_media=torch.full((n_c,), 0.15),
                        gpx4_inhib=torch.full((n_c,), 3.5),
                    )
                obs_spi.append(float(srow["SPI"]))
                u_means.append(float(wout["u_final"].mean().item()))
                morans.append(float(wout["morans_i"].item()))
                prop_probs.append(float(wout["propagative_prob"].item()))
                para_factors.append(float(pf))

            zen_decomp = {
                "n_fovs": len(obs_spi),
                "rho_u_final_mean_alone": round(float(spearmanr(obs_spi, u_means).statistic), 4),
                "rho_morans_i_alone": round(float(spearmanr(obs_spi, morans).statistic), 4),
                "rho_propagative_prob_alone": round(float(spearmanr(obs_spi, prop_probs).statistic) if np.std(prop_probs) > 1e-9 else 0.0, 4),
                "rho_external_paracrine_factor_alone": round(float(spearmanr(obs_spi, para_factors).statistic), 4),
                "rho_combined_product_reported": round(float(spearmanr(obs_spi, np.array(u_means) * np.array(para_factors)).statistic), 4),
            }

    pd.DataFrame(deconf_rows).to_csv(output_dir / "pillar3_stage0_deconfounded.csv", index=False)

    findings.extend(
        [
            {
                "id": "P3-TREATMENT-INPUT-CIRCULARITY",
                "severity": "CRITICAL",
                "title": "Cohort 1 (AUROC=1.00) and Cohort 2 regimen ladder (rho=0.59) fed ground-truth treatment arms into np_params/media",
                "details": deconf_rows,
            },
            {
                "id": "P3-MRNA-STEROL-TAUTOLOGY",
                "severity": "CRITICAL",
                "title": "Cohort 2 [7-DHC] validation compares model EBP/DHCR7 formula against input EBP/SC5D-DHCR7 mRNA; correlation collapses when those 3 genes are masked",
                "details": {r["ensemble"]: {"unmasked_rho": r["gse319397_sterol_rho_with_direct_mrna"], "masked_rho": r["gse319397_sterol_rho_when_ebp_sc5d_dhcr7_masked"]} for r in deconf_rows},
            },
            {
                "id": "P3-ZENODO-WAVE-DECOMPOSITION",
                "severity": "MAJOR",
                "title": "Decomposition of Cohort 3 (Zenodo 18271042) rho=0.9286 between internal PDE u_final/Moran's I vs external geometric paracrine_factor",
                "details": zen_decomp,
            },
        ]
    )

    report = {
        "pillar": 3,
        "name": "Stage-0 Blind Validation Circularity, mRNA Tautology & Spatial Decomposition",
        "findings": findings,
        "peak_rss_mb": get_peak_rss_mb(),
    }
    (output_dir / "pillar3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_pillar3_audit(ROOT_DIR / "results" / "audit")
    print(json.dumps({"pillar": 3, "n_findings": len(res["findings"]), "peak_rss_mb": res["peak_rss_mb"]}, indent=2))
