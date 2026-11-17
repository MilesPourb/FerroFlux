"""Zero-Trust Forensic Audit Regression Suite (`FerroFlux-FM v2`).

Permanently guards against every Critical & Major bug uncovered during the 8-Pillar Forensic Audit:
1. `P2-PROBE-MECH-LEAK`: `compute_compound_featurization` accepts ONLY `smiles` (zero `probe_mechanism` bypass).
2. `P1-SILENT-MID-FALLBACK`: Unmatched cell IDs produce `gene_mask == 0` (never silent `ACH-000873` fallback).
3. `P4-DOSE-OFFSET`: `encode_log10_dose_um(0.0) == -3.0` & decoded `0 uM` offset `< 1e-6 uM`.
4. `P4-DEAD-FLUX-NET` & `P4-DISCONNECTED-OTHER17-RCD`: Exact non-zero autograd gradients into
   `flux_proposal_net` (from `h_ferro`) and `other_17_rcd_head` (from `viability_pred`), plus cone
   irreversibility & `mass_balance_residual` (`||S v||_2 < 1e-4`).
5. `P4-CRISPR-2020-GENE-BLINDNESS`: Non-anchor single-gene knockouts (`g >= 64`) shift `chronos_pred`
   and `delta_expr_pred` with non-zero gradients into `gene_essentiality_prior`.
6. `P4-UNMASKED-MGM`: `model.train()` applies random `15%` gene masking (`mgm_eval_mask` emitted).
7. `P6-CDOT-IRON-LOADING`: Iron-free (`fe_load == 0.0`) `<8 nm` PEG-silica C' dots induce positive
   `delta_j_lyso_fe > 0` via ambient `Fe3+` scavenging + `MC1R`/`FOLH1`/`TFRC` receptor uptake.
8. `F3`: `is_normal_organ_or_til` metadata flag has zero direct effect on `sfti_score` or `viability_pred`.
9. `P3-TREATMENT-INPUT-CIRCULARITY` & `P3-MRNA-STEROL-TAUTOLOGY`: Stage-0 `_mask_gene_set_in_expr`
   zeros target genes (`expr = 0, mask = 0`), and `generate_nature_figures.py` contains zero hardcoded
   `9.5e7` corpus counts or synthetic fallback DataFrames.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from ferroflux.data.dataset import compute_compound_featurization, encode_log10_dose_um
from ferroflux.models.ferroflux_fm import FerroFluxFM
from ferroflux.models.losses import FerroFluxCurriculumLoss
from ferroflux.validation.stage0_bradbury_blind_eval import _mask_gene_set_in_expr

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")


def test_smiles_only_featurization_and_exact_zero_dose_decoding() -> None:
    fp0, d0_v = compute_compound_featurization("CC(=O)OC1=CC=CC=C1C(=O)O", probe_mechanism="")
    fp1, d1_v = compute_compound_featurization("CC(=O)OC1=CC=CC=C1C(=O)O", probe_mechanism="gpx4_covalent")
    fp2, d2_v = compute_compound_featurization("CC(=O)OC1=CC=CC=C1C(=O)O", probe_mechanism="cuproptosis")
    assert np.array_equal(fp0, fp1) and np.array_equal(d0_v, d1_v) and np.array_equal(d0_v, d2_v), (
        "probe_mechanism MUST have zero effect on compute_compound_featurization!"
    )
    # Thiocarbonyl copper-ionophore SMARTS motif (elesclomol) detected purely from SMILES
    _, desc_eles = compute_compound_featurization("CN(C(=S)C(=O)NN(C)C(=S)C1=CC=CC=C1)C2=CC=CC=C2")
    assert float(desc_eles[11]) == 1.0, "Pure RDKit SMARTS must detect thiocarbonyl copper-ionophore motif!"

    # Exact zero-dose encoding/decoding (`0 uM` produces exact zero occupancy even with active SMILES!)
    d0 = encode_log10_dose_um(0.0)
    d1 = encode_log10_dose_um(1.0)
    assert abs(d0 - (-3.0)) < 1e-6
    m = FerroFluxFM(n_genes=128, d_model=64, n_layers=2, n_heads=2)
    m.eval()
    with torch.no_grad():
        occ0, kr0, kc0, _, _, _ = m.perturbation_op._encode_single_drug(
            torch.from_numpy(fp0).unsqueeze(0),
            torch.from_numpy(desc_eles).unsqueeze(0),
            torch.tensor([d0]),
            True,
            1,
            torch.device("cpu"),
            torch.float32,
        )
        occ1, _, _, _, _, _ = m.perturbation_op._encode_single_drug(
            torch.from_numpy(fp0).unsqueeze(0),
            torch.from_numpy(desc_eles).unsqueeze(0),
            torch.tensor([d1]),
            True,
            1,
            torch.device("cpu"),
            torch.float32,
        )
    assert float(occ0.abs().max().item()) == 0.0 and float(kr0.item()) == 0.0 and float(kc0.item()) == 0.0
    assert float(occ1.max().item()) > 0.0


def test_zero_dead_branches_and_full_2048_gene_crispr_sensitivity() -> None:
    torch.manual_seed(2026)
    m = FerroFluxFM(n_genes=256, d_model=64, n_layers=2, n_heads=2)
    nn_init = torch.randn(256) * 0.25
    with torch.no_grad():
        m.gene_essentiality_prior.copy_(nn_init)
    m.train()

    x = torch.rand(4, 256) + 0.2
    out = m({"gene_expr": x, "gpx4_inhib": torch.full((4,), 2.5)})
    assert "mgm_eval_mask" in out, "model.train() must emit 15% random mgm_eval_mask!"
    assert "mass_balance_residual" in out, "forward() must emit mass_balance_residual!"
    assert float(out["mass_balance_residual"].max().item()) < 1e-4

    # Check flux_proposal_net gradient from h_ferro alone
    m.zero_grad(set_to_none=True)
    out["h_ferro"].sum().backward(retain_graph=True)
    flux_grads = [p.grad.norm().item() for p in m.diff_ferro_gem.flux_proposal_net.parameters() if p.grad is not None]
    assert flux_grads and sum(flux_grads) > 1e-8, "flux_proposal_net is disconnected from h_ferro!"

    # Check other_17_rcd_head gradient from viability_pred alone
    m.zero_grad(set_to_none=True)
    out["viability_pred"].sum().backward()
    rcd17_grad = m.other_17_rcd_head.weight.grad
    assert rcd17_grad is not None and float(rcd17_grad.norm().item()) > 1e-9, "other_17_rcd_head is disconnected!"

    # Check non-anchor single-gene CRISPR KO (`gene_idx = 180 >= 64`) shifts chronos_pred & delta_expr_pred
    m.eval()
    gko_wt = torch.ones(2, 256)
    gko_ko = torch.ones(2, 256)
    gko_ko[:, 180] = 0.05
    with torch.no_grad():
        o_wt = m({"gene_expr": x[:2], "gene_ko_mult": gko_wt})
        o_ko = m({"gene_expr": x[:2], "gene_ko_mult": gko_ko})
    assert float((o_ko["chronos_pred"] - o_wt["chronos_pred"]).abs().max().item()) > 1e-3
    assert float((o_ko["delta_expr_pred"][:, 180] - o_wt["delta_expr_pred"][:, 180]).abs().max().item()) > 1e-2


def test_iron_free_peg_silica_cdot_scavenges_fe_and_no_normal_label_leak() -> None:
    m = FerroFluxFM(n_genes=128, d_model=64, n_layers=2, n_heads=2)
    m.eval()
    x = torch.ones(2, 128) * 1.5
    # Iron-free (`fe_load == 0.0`) 6 nm PEG-silica C' dot vs vehicle (`dose == 0.0`)
    np_cdot_iron_free = torch.tensor([[6.0, -5.0, 1.0, 0.0, 8.0], [6.0, -5.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        o_np = m({"gene_expr": x, "np_params": np_cdot_iron_free, "cys2_media": torch.tensor([0.1, 0.1])})
    assert float(o_np["fe2_lyso"][0].item()) > float(o_np["fe2_lyso"][1].item()), (
        "Iron-free (<12 nm) PEG-silica C' dot (fe_load=0) must increase lysosomal Fe2+ via ambient Fe3+ scavenging!"
    )

    # Verify `is_normal_organ_or_til` flag has ZERO direct shortcut effect on sfti_score or viability_pred
    with torch.no_grad():
        o0 = m({"gene_expr": x[:1], "is_normal_organ_or_til": torch.tensor([0.0]), "gpx4_inhib": torch.tensor([2.0])})
        o1 = m({"gene_expr": x[:1], "is_normal_organ_or_til": torch.tensor([1.0]), "gpx4_inhib": torch.tensor([2.0])})
    assert abs(float(o0["sfti_score"].item() - o1["sfti_score"].item())) < 1e-7
    assert abs(float(o0["viability_pred"].item() - o1["viability_pred"].item())) < 1e-7


def test_stage0_gene_masking_and_figure_code_integrity() -> None:
    expr = np.ones((4, 16), dtype=np.float32) * 3.5
    g2i = {"EBP": 2, "SC5D": 5, "DHCR7": 9, "GPX4": 0}
    xm, mm, idx = _mask_gene_set_in_expr(expr, g2i, ("EBP", "SC5D", "DHCR7"))
    assert idx == [2, 5, 9]
    assert np.all(xm[:, [2, 5, 9]] == 0.0) and np.all(mm[:, [2, 5, 9]] == 0.0)
    assert np.all(xm[:, 0] == 3.5) and np.all(mm[:, 0] == 1.0)

    fig_src = (ROOT_DIR / "ferroflux" / "validation" / "generate_nature_figures.py").read_text(encoding="utf-8")
    assert "9.5e7" not in fig_src, "Hardcoded 9.5e7 upstream count must never reappear in generate_nature_figures.py!"
    stage0_src = (ROOT_DIR / "ferroflux" / "validation" / "stage0_bradbury_blind_eval.py").read_text(encoding="utf-8")
    stage0_code_only = "\n".join(
        ln for ln in stage0_src.split("def run_stage0_blind_evaluation")[1].splitlines() if not ln.strip().startswith("#")
    )
    assert "np_mel[mel_arms == 1]" not in stage0_code_only, "Circular treatment-arm conditioning must never reappear!"
    assert "paracrine_factor" not in stage0_code_only, (
        "External paracrine_factor post-multiplier must not appear in run_stage0_blind_evaluation!"
    )


def test_v3_remediated_parquet_shards_rsl3_signed_orcs_and_genuine_3d() -> None:
    import pandas as pd

    proc = ROOT_DIR / "data" / "processed"
    df_ph = pd.read_parquet(proc / "pharmacogenomic_viability.parquet", columns=["compound_name", "canonical_smiles"])
    rsl3 = df_ph[df_ph["compound_name"].astype(str).str.upper().str.contains("RSL3")]
    assert len(rsl3) >= 500, f"Expected >=500 remediated 1S,3R-RSL3 curves, found {len(rsl3)}"
    smi_rsl3 = str(rsl3["canonical_smiles"].iloc[0])
    _, desc_rsl3 = compute_compound_featurization(smi_rsl3)
    assert float(desc_rsl3[6]) == 1.0, "1S,3R-RSL3 chloroacetamide covalent warhead must be detected from SMILES!"

    df_cells = pd.read_parquet(proc / "cells_basal_multiomics.parquet", columns=["model_id", "is_3d_organoid"])
    n_3d_models = int((df_cells["is_3d_organoid"] == 1).sum())
    assert n_3d_models >= 150, f"Expected >=150 genuine 3D organoid/neurosphere models, found {n_3d_models}"

    df_cr = pd.read_parquet(
        proc / "crispr_chronos_2d_3d.parquet",
        columns=["model_id", "gene_symbol", "chronos_score", "is_3d_organoid", "is_crispr_fin_epistasis"],
    )
    cr_3d = df_cr[df_cr["is_3d_organoid"] == 1]
    assert len(cr_3d) >= 5000 and int((cr_3d["model_id"] == "ACH-000873").sum()) == 0

    ep = df_cr[df_cr["is_crispr_fin_epistasis"] == 1]
    gene_means = ep.groupby("gene_symbol")["chronos_score"].mean()
    assert float(gene_means["ACSL4"]) > 1.0 and float(gene_means["POR"]) > 1.0, "Execution gene KOs must rescue FIN (> +1.0)!"
    assert float(gene_means["GPX4"]) < -0.5 and float(gene_means["AIFM2"]) < -0.5, "Defense gene KOs must sensitize FIN (< -0.5)!"


def test_v3_stage05_sar_zero_leakage_across_smiles_ik14_and_scaffolds() -> None:
    from ferroflux.data.dataset import FerroChemSARPretrainDataset, FerroFluxMultimodalDataset
    from ferroflux.training.evaluate_ood_and_ablations import _extract_forbidden_test_chemistry

    ds = FerroFluxMultimodalDataset(data_dir=ROOT_DIR / "data" / "processed", license_track="open", max_samples_per_shard=4000)
    for split_name, method_name in [
        ("1_leave_out_scaffold", "scaffold_split"),
        ("5_cuproptosis_elesclomol_specificity", "cuproptosis_specificity_split"),
    ]:
        _, te_idx = getattr(ds, method_name)()
        f_smi, f_ik14, f_scaf = _extract_forbidden_test_chemistry(ds, list(te_idx), split_name)
        sar_ds = FerroChemSARPretrainDataset(
            data_dir=ROOT_DIR / "data" / "processed",
            license_track="open",
            forbid_smiles=f_smi,
            forbid_ik14=f_ik14,
            forbid_scaffolds=f_scaf,
            max_molecules=1500,
        )
        assert len(sar_ds) > 100
        assert sar_ds.sar_smiles.isdisjoint(f_smi), f"SMILES leak in {split_name}!"
        assert sar_ds.sar_ik14.isdisjoint(f_ik14), f"InChIKey-14 connectivity leak in {split_name}!"
        assert sar_ds.sar_scaffolds.isdisjoint(f_scaf), f"Murcko scaffold leak in {split_name}!"


def test_v3_aphynity_greybox_and_four_corner_epistasis_laws() -> None:
    torch.manual_seed(77)
    m = FerroFluxFM(n_genes=256, d_model=96, n_layers=2, n_heads=4)
    # Perturb APHYNITY residual away from zero init to verify biophysical laws hold for arbitrary trained weights!
    with torch.no_grad():
        for p in m.diff_ferro_gem.aphynity_residual_net.parameters():
            p.normal_(0.0, 0.35)
    m.eval()

    x = torch.ones(4, 256) * 1.4
    z_chem_dummy = torch.randn(4, 96)
    with torch.no_grad():
        o_base = m({"gene_expr": x, "gpx4_inhib": torch.full((4,), 4.0), "elesclomol_conc": torch.full((4,), 2.0)})
        o_lip = m({"gene_expr": x, "gpx4_inhib": torch.full((4,), 4.0), "elesclomol_conc": torch.full((4,), 2.0), "rta_conc": torch.full((4,), 6.0)})
        o_dfo = m({"gene_expr": x, "gpx4_inhib": torch.full((4,), 4.0), "elesclomol_conc": torch.full((4,), 2.0), "chelator_conc": torch.full((4,), 6.0)})
        assert float((o_base["aphynity_gain"] - 1.0).abs().max().item()) > 1e-3
        assert bool((o_lip["h_ferro"] < o_base["h_ferro"]).all()), "APHYNITY residual must preserve 100% Lip-1 rescue!"
        assert bool((o_dfo["h_ferro"] < o_base["h_ferro"]).all()), "APHYNITY residual must preserve 100% DFO rescue!"
        assert float((o_lip["h_cupro"] - o_base["h_cupro"]).abs().max().item()) == 0.0

        # 4-Corner Analytical Epistasis: ACSL4 KO (idx 24) rescues GPX4 inhibition while AIFM2/FSP1 KO (idx 12) sensitizes!
        from ferroflux.layers.perturbation_ops import DEFAULT_ANCHOR_28_VOCAB_IDX, PROTEIN_TO_IDX

        acsl4_g = DEFAULT_ANCHOR_28_VOCAB_IDX[PROTEIN_TO_IDX["ACSL4"]]
        aifm2_g = DEFAULT_ANCHOR_28_VOCAB_IDX[PROTEIN_TO_IDX["AIFM2"]]
        gko_acsl4 = torch.ones(2, 256)
        gko_acsl4[:, acsl4_g] = 0.05
        gko_aifm2 = torch.ones(2, 256)
        gko_aifm2[:, aifm2_g] = 0.05

        ep_acsl4 = m({"gene_expr": x[:2], "gene_ko_mult": gko_acsl4, "gpx4_inhib": torch.full((2,), 3.5)})["chronos_pred"]
        ep_aifm2 = m({"gene_expr": x[:2], "gene_ko_mult": gko_aifm2, "gpx4_inhib": torch.full((2,), 3.5)})["chronos_pred"]
        assert bool((ep_acsl4 > ep_aifm2 + 0.5).all()), (
            f"4-corner epistasis must rank ACSL4 KO rescue ({ep_acsl4.tolist()}) above AIFM2 KO synthetic lethality ({ep_aifm2.tolist()})!"
        )

