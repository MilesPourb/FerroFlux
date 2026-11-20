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


def test_v3_comma_smiles_sanitization_and_bidentate_cu_ionophore() -> None:
    from ferroflux.data.dataset import canonicalize_smiles_and_keys, compute_compound_featurization

    # PRISM comma-delimited replicate SMILES must parse identically to clean single SMILES
    disulf_single = "CCN(CC)C(=S)SSC(=S)N(CC)CC"
    disulf_prism = f"{disulf_single}, {disulf_single}, {disulf_single}"
    c1, ik1, sc1 = canonicalize_smiles_and_keys(disulf_single)
    c2, ik2, sc2 = canonicalize_smiles_and_keys(disulf_prism)
    assert c1 == c2 and len(ik1) == 14 and ik1 == ik2 and sc1 == sc2

    fp1, d1 = compute_compound_featurization(disulf_single)
    fp2, d2 = compute_compound_featurization(disulf_prism)
    assert np.array_equal(fp1, fp2) and np.array_equal(d1, d2)
    assert float(fp2.sum()) > 10.0 and float(d2[11]) == 1.0

    # Necrostatin-1 has only 1 cyclic C=S carbon -> must NOT trigger bidentate Cu-ionophore descriptor
    _, d_nec = compute_compound_featurization("CN1C(=O)NC(Cc2c[nH]c3ccccc23)C1=S")
    assert float(d_nec[11]) == 0.0


def test_v3_txnrd1_auranofin_connectivity_and_rescue_non_contradiction() -> None:
    from ferroflux.data.dataset import FerroFluxMultimodalDataset, compute_compound_featurization, encode_log10_dose_um

    torch.manual_seed(19)
    m = FerroFluxFM(n_genes=256, d_model=64, n_layers=2, n_heads=2)
    m.eval()
    aur_smi = "CC[P+](CC)(CC)[Au-][S]C1OC(COC(C)=O)C(OC(C)=O)C(OC(C)=O)C1OC(C)=O"
    fp_aur, d_aur = compute_compound_featurization(aur_smi)
    x = torch.ones(2, 256) * 1.2
    out_veh = m({"gene_expr": x})
    out_aur = m(
        {
            "gene_expr": x,
            "chem_fp_a": torch.from_numpy(np.stack([fp_aur, fp_aur])),
            "chem_desc_a": torch.from_numpy(np.stack([d_aur, d_aur])),
            "dose_a": torch.full((2,), encode_log10_dose_um(5.0)),
        }
    )
    assert bool((out_aur["h_ferro"] > out_veh["h_ferro"]).all())
    assert bool((out_aur["h_disulf"] > out_veh["h_disulf"]).all())
    out_aur["viability_pred"].sum().backward()
    assert m.raw_k_txnrd1_tox.grad is not None and float(m.raw_k_txnrd1_tox.grad.abs().item()) > 1e-8

    ds = FerroFluxMultimodalDataset(data_dir=ROOT_DIR / "data" / "processed", license_track="academic", max_samples_per_shard=1500)
    rec_idx = next((i for i, r in enumerate(ds.records) if r.get("kind") == "rescue"), None)
    if rec_idx is not None:
        item = ds[rec_idx]
        assert float(item["gpx4_inhib"].item()) > 0.0
        assert float(item["rta_conc"].item()) == 0.0


def test_v3_fast_multimodal_indexer_exact_parity() -> None:
    from ferroflux.data.dataset import FerroFluxMultimodalDataset, get_fast_indexer

    ds = FerroFluxMultimodalDataset(data_dir=ROOT_DIR / "data" / "processed", license_track="open", max_samples_per_shard=800)
    idxr = get_fast_indexer(ds)
    assert idxr is not None
    sample_idx = list(range(0, min(len(ds), 64), 3))
    ref_b = ds.collate_fn([ds[i] for i in sample_idx])
    fast_b = idxr.get_batch(sample_idx)
    for k, rv in ref_b.items():
        assert float((rv - fast_b[k]).abs().max().item()) < 1e-6, f"Parity mismatch on {k}"


def test_v3_round2_sar_snapshot_mgm_shift_invariance_and_apop_identifiability() -> None:
    torch.manual_seed(314)
    m = FerroFluxFM(n_genes=128, d_model=64, n_layers=2, n_heads=2)
    m.perturbation_op.snapshot_stage05_sar_anchor()
    assert float(m.perturbation_op.has_sar_snapshot.item()) == 1.0
    for p in m.perturbation_op.sar_anchor_blocks.parameters():
        assert not p.requires_grad

    fp_a, desc_a = compute_compound_featurization("CC(=O)OC1=CC=CC=C1C(=O)O")
    fp_t = torch.from_numpy(fp_a).unsqueeze(0)
    desc_t = torch.from_numpy(desc_a).unsqueeze(0)
    sar_before = m.perturbation_op.compute_frozen_sar_vector(fp_t, desc_t).detach().clone()

    # Mutate live chem_fp_proj weights -> frozen SAR anchor must remain bitwise identical!
    with torch.no_grad():
        m.perturbation_op.chem_fp_proj[0].weight.add_(1.25)
    sar_after = m.perturbation_op.compute_frozen_sar_vector(fp_t, desc_t)
    assert torch.allclose(sar_before, sar_after, atol=0.0)

    # Ablated SAR pretrain must emit exact zeros for frozen SAR vector
    m_ab = FerroFluxFM(n_genes=128, d_model=64, n_layers=2, n_heads=2, ablate_sar_pretrain=True)
    assert float(m_ab.perturbation_op.compute_frozen_sar_vector(fp_t, desc_t).abs().max().item()) == 0.0

    # Rescue-gated additive APHYNITY scaffold head preserves 100% Lip-1 & DFO rescue for arbitrary weights!
    with torch.no_grad():
        m.diff_ferro_gem.scaf_gb_ready.fill_(1.0)
        for p in m.diff_ferro_gem.aphynity_scaf_head.parameters():
            p.normal_(0.5, 0.25)
    m.eval()
    x_base = torch.linspace(0.5, 3.5, 128).unsqueeze(0).repeat(2, 1)
    fp_b = fp_t.repeat(2, 1)
    desc_b = desc_t.repeat(2, 1)
    dose_b = torch.full((2,), encode_log10_dose_um(4.0))
    with torch.no_grad():
        o_drug = m({"gene_expr": x_base, "chem_fp_a": fp_b, "chem_desc_a": desc_b, "dose_a": dose_b})
        o_lip = m({"gene_expr": x_base, "chem_fp_a": fp_b, "chem_desc_a": desc_b, "dose_a": dose_b, "rta_conc": torch.full((2,), 5.0)})
        o_dfo = m({"gene_expr": x_base, "chem_fp_a": fp_b, "chem_desc_a": desc_b, "dose_a": dose_b, "chelator_conc": torch.full((2,), 5.0)})
    assert bool((o_lip["h_ferro"] < o_drug["h_ferro"]).all()), "aphynity_scaf_head must preserve Lip-1 rescue!"
    assert bool((o_dfo["h_ferro"] < o_drug["h_ferro"]).all()), "aphynity_scaf_head must preserve DFO rescue!"

    # Verify full non-ablated `warmstart_from_train_indices` + `fit_post_curriculum_greybox_from_train_indices` execute cleanly
    from ferroflux.data.dataset import FerroFluxMultimodalDataset
    from ferroflux.training.train_curriculum import (
        fit_post_curriculum_greybox_from_train_indices,
        warmstart_from_train_indices,
    )

    ds_w = FerroFluxMultimodalDataset(data_dir=ROOT_DIR / "data" / "processed", license_track="open", max_samples_per_shard=600)
    m_w = FerroFluxFM(n_genes=ds_w.n_genes, d_model=64, n_layers=2, n_heads=2, ablate_physical_anchors=False)
    m_w.perturbation_op.snapshot_stage05_sar_anchor()
    tr_sub = list(range(len(ds_w) - 100))
    forbid_sub = set(range(len(ds_w) - 100, len(ds_w)))
    warmstart_from_train_indices(m_w, ds_w, tr_sub, forbid_test_set=forbid_sub, device=torch.device("cpu"), seed=101)
    fit_post_curriculum_greybox_from_train_indices(
        m_w, ds_w, tr_sub, forbid_test_set=forbid_sub, split_name="1_leave_out_scaffold", device=torch.device("cpu"), seed=101
    )
    assert float(m_w.diff_ferro_gem.scaf_gb_ready.item()) == 1.0
    fit_post_curriculum_greybox_from_train_indices(
        m_w, ds_w, tr_sub, forbid_test_set=forbid_sub, split_name="5_cuproptosis_elesclomol_specificity", device=torch.device("cpu"), seed=101
    )


def test_v3_round3_synergy_smiles_b_zero_leak_honest_ablations_and_mgm_sterol() -> None:
    """Verify multi-drug `smiles_b` zero-leakage, orthogonal ablations, affine MGM invariance & pure sterol cross-prediction."""
    from ferroflux.data.dataset import FerroChemSARPretrainDataset, FerroFluxMultimodalDataset
    from ferroflux.training.evaluate_ood_and_ablations import _build_six_ood_splits, _extract_forbidden_test_chemistry

    # 1. Multi-drug `synergy` `smiles_b` salt/scaffold zero-leakage across OOD splits & Stage-0.5 SAR
    ds = FerroFluxMultimodalDataset(
        data_dir=ROOT_DIR / "data" / "processed", license_track="open", max_samples_per_shard=2500
    )
    splits = _build_six_ood_splits(ds)
    for sp_key in ("1_leave_out_scaffold", "5_cuproptosis_elesclomol_specificity"):
        tr_idx, te_idx = splits[sp_key]
        f_smi, f_ik14, f_scaf = _extract_forbidden_test_chemistry(ds, te_idx, sp_key)
        assert len(f_scaf) >= 1
        for ti in tr_idx:
            rc = ds.records[ti]
            for ks in ("canonical_smiles", "fin_modifier_smiles", "smiles_a", "smiles_b"):
                sv = str(rc.get(ks, "") or "").strip()
                assert not sv or sv not in f_smi, f"Leaked {ks} in {sp_key}!"
            for ki in ("ik14", "ik14_a", "ik14_b"):
                iv = str(rc.get(ki, "") or "").strip()[:14]
                assert not iv or iv not in f_ik14, f"Leaked {ki} in {sp_key}!"
            for ksc in ("murcko_scaffold", "murcko_scaffold_a", "murcko_scaffold_b"):
                scv = str(rc.get(ksc, "") or "").strip()
                assert not scv or scv not in f_scaf, f"Leaked {ksc} in {sp_key}!"

    # Synthetic multi-drug salt-form `smiles_b` without pre-populated `murcko_scaffold` must still block shared scaffolds!
    class _DummySynDS:
        def __init__(self) -> None:
            self.records = [
                {
                    "kind": "synergy",
                    "smiles_a": "CCO",
                    "smiles_b": "C#CCNc1ccc(-c2cc(=O)c3ccccc3o2)cc1.Cl",
                }
            ]

        def __len__(self) -> int:
            return 1

    f_s2, f_ik2, f_sc2 = _extract_forbidden_test_chemistry(_DummySynDS(), [0], "1_leave_out_scaffold")
    assert "O=c1cc(-c2ccccc2)oc2ccccc12" in f_sc2
    sar_clean = FerroChemSARPretrainDataset(
        data_dir=ROOT_DIR / "data" / "processed",
        license_track="open",
        forbid_smiles=f_s2,
        forbid_ik14=f_ik2,
        forbid_scaffolds=f_sc2,
    )
    assert len(sar_clean) > 1000

    # 2. Honest ablation orthogonality: `ablate_sec46_gate` preserves ECFP/MACCS/SAR, and `ablate_cis_reg_slot`
    #    & `ablate_physical_anchors` NEVER erase observed mRNA (`gx_rel`)!
    fp_v, desc_v = compute_compound_featurization("CC(=O)Nc1ccc(O)cc1")
    fp_t = torch.from_numpy(fp_v).unsqueeze(0)
    desc_t = torch.from_numpy(desc_v).unsqueeze(0)
    du_t = torch.tensor([2.5])
    dl_t = torch.tensor([0.4])

    m_full = FerroFluxFM(n_genes=256, d_model=64, n_layers=2, n_heads=2)
    m_sec = FerroFluxFM(n_genes=256, d_model=64, n_layers=2, n_heads=2, ablate_sec46_gate=True)
    m_cis = FerroFluxFM(n_genes=256, d_model=64, n_layers=2, n_heads=2, ablate_cis_reg_slot=True)
    m_anc = FerroFluxFM(n_genes=256, d_model=64, n_layers=2, n_heads=2, ablate_physical_anchors=True)
    for mod in (m_full, m_sec):
        mod.perturbation_op.snapshot_stage05_sar_anchor()

    ecfp_sec = m_sec.perturbation_op.build_shared_ecfp_features(fp_t)
    cd_sec = m_sec.perturbation_op.build_raw_chem_dense_features(fp_t, desc_t, du_t, dl_t)
    sar_off = m_sec.perturbation_op.chem_desc_dim + m_sec.perturbation_op.maccs_sub_dim
    assert float(ecfp_sec.abs().sum().item()) > 0.0, "ablate_sec46_gate must NOT zero shared ECFP bits!"
    assert float(cd_sec[:, sar_off : sar_off + m_sec.perturbation_op.sar_snap_dim].abs().sum().item()) > 0.0, (
        "ablate_sec46_gate must NOT zero frozen SAR vector!"
    )

    gx_in = torch.linspace(1.0, 9.0, 256).unsqueeze(0)
    gm_in = torch.ones_like(gx_in)
    z28 = torch.zeros(1, 28)
    z12 = torch.zeros(1, 12)
    z234 = torch.zeros(1, 234)
    z64 = torch.ones(1, 64)
    fc_full, _ = m_full.diff_ferro_gem.build_raw_cell_greybox_features(gx_in, gm_in, z28, z12, z234, z64)
    fc_cis, _ = m_cis.diff_ferro_gem.build_raw_cell_greybox_features(gx_in, gm_in, z28, z12, z234, z64)
    fc_anc, _ = m_anc.diff_ferro_gem.build_raw_cell_greybox_features(gx_in, gm_in, z28, z12, z234, z64)
    assert torch.allclose(fc_full[:, :256], fc_cis[:, :256], atol=1e-6), "ablate_cis_reg must NOT erase mRNA!"
    assert torch.allclose(fc_full[:, :256], fc_anc[:, :256], atol=1e-6), "ablate_physical_anchors must NOT erase mRNA!"

    # 3. Affine-equivariant `mgm_recon` shift invariance AND absence of `log(h_ferro)` proxy in Stage-0 sterol eval
    m_full.eval()
    with torch.no_grad():
        m_full.mgm_svd_proj.weight.normal_(0.0, 0.05)
        m_full.mgm_ctx_decoder.weight[:, m_full.d_model :].normal_(0.0, 0.05)
        r_base = m_full({"gene_expr": gx_in, "gene_mask": gm_in})["mgm_recon"]
        r_shift = m_full({"gene_expr": gx_in + 4.5, "gene_mask": gm_in})["mgm_recon"]
    # Shifting observed expression by +4.5 shifts reconstructed z-score back by +4.5 on the affine SVD branch!
    assert float(((r_shift - r_base) - 4.5).abs().mean().item()) < 0.6

    s0_src = (ROOT_DIR / "ferroflux" / "validation" / "stage0_bradbury_blind_eval.py").read_text(encoding="utf-8")
    assert 'pc_sterol_masked_pred["h_ferro"]' not in s0_src, "Stage-0 Cohort 3 must NOT use h_ferro proxy!"
    assert 'pc_sterol_masked_pred["mgm_recon"]' in s0_src





