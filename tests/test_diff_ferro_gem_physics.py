"""Biophysical Conservation, Nullspace Stoichiometry, Sign Laws & Zero-Bypass Skip Tests.

Verifies Reviewer #2 mathematical & biophysical identifiability guarantees for FerroFlux-FM:
1. 94-reaction Human2 subnetwork (ferro_gem_94rxn.json) satisfies ||S @ N_basis||_F < 1e-10.
2. DiffFerroGEM fluxes satisfy exact mass balance ||S v||_2 < 1e-5 with strictly positive kinetic rates.
3. Monotonic rescue specificity: increasing [Liproxstatin-1] or [DFO] strictly lowers [PL-OOH] and H_ferro,
   while leaving Elesclomol cuproptosis hazard H_cupro invariant.
4. Opposing distal sterol flux law ([7-DHC]_ss = alpha * v_EBP_SC5D / (k_0 + v_DHCR7)):
   DHCR7 suppression elevates [7-DHC]_ss & protects (lowers H_ferro), whereas upstream EBP suppression
   depletes [7-DHC]_ss & sensitizes (raises H_ferro).
5. Sec46 translational gate: depleting PSTK/SEPSECS or mevalonate IPP lowers active GPX4 even at fixed GPX4 mRNA.
6. Parameter-free physical nanoparticle factorization: <8 nm C' dot + cystine starvation elevates
   lysosomal [Fe2+]_lyso, Gal3 rupture probability g_rupture, and H_ferro, rescued by Concanamycin A, DFO, or Lip-1.
7. Zero neural skip-connection bypasses DiffFerroGEM ([PL-OOH] / g_rupture) to H_ferro.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
GEM_JSON = ROOT_DIR / "configs" / "ferro_gem_94rxn.json"


def _extract_tensor(out: Any, candidate_keys: list[str]) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        lower_map = {k.lower(): v for k, v in out.items()}
        for ck in candidate_keys:
            if ck.lower() in lower_map and isinstance(lower_map[ck.lower()], torch.Tensor):
                return lower_map[ck.lower()]
        for k, v in out.items():
            if any(ck.lower() in k.lower() for ck in candidate_keys) and isinstance(v, torch.Tensor):
                return v
    for ck in candidate_keys:
        if hasattr(out, ck) and isinstance(getattr(out, ck), torch.Tensor):
            return getattr(out, ck)
    raise KeyError(f"Could not find any of {candidate_keys} in output: {type(out)} / {getattr(out, 'keys', lambda: dir(out))()}")


def test_ferro_gem_94rxn_json_stoichiometry_and_nullspace() -> None:
    assert GEM_JSON.is_file(), f"Missing stoichiometric config {GEM_JSON}"
    gem = json.loads(GEM_JSON.read_text(encoding="utf-8"))

    S = np.asarray(gem.get("S", gem.get("stoichiometric_matrix")), dtype=np.float64)
    N = np.asarray(gem.get("N_basis", gem.get("nullspace_basis")), dtype=np.float64)
    rxns = gem.get("reactions", gem.get("reaction_ids", []))

    assert S.shape[1] == 94, f"Expected 94 reactions in S, got shape {S.shape}"
    assert len(rxns) == 94, f"Expected 94 reaction entries, got {len(rxns)}"
    assert N.shape[0] == 94 and N.shape[1] >= 1, f"Invalid nullspace basis shape {N.shape}"

    residual = np.linalg.norm(S @ N, ord="fro")
    assert residual < 1e-10, f"Nullspace basis violates S @ N = 0: ||S @ N||_F = {residual:.3e}"

    gram_err = np.linalg.norm(N.T @ N - np.eye(N.shape[1]), ord="fro")
    assert gram_err < 1e-6, f"Nullspace basis is not orthonormal: ||N^T N - I||_F = {gram_err:.3e}"


def test_diff_ferro_gem_mass_balance_and_positive_kinetics() -> None:
    from ferroflux.layers import diff_ferro_gem

    gem_cls = getattr(diff_ferro_gem, "DiffFerroGEM")
    sig = inspect.signature(gem_cls.__init__)
    kwargs: dict[str, Any] = {}
    if "gem_json_path" in sig.parameters:
        kwargs["gem_json_path"] = str(GEM_JSON)
    elif "config_path" in sig.parameters:
        kwargs["config_path"] = str(GEM_JSON)
    if "d_model" in sig.parameters:
        kwargs["d_model"] = 128

    layer = gem_cls(**kwargs)
    layer.eval()

    torch.manual_seed(42)
    B = 32
    d_in = getattr(layer, "d_model", getattr(layer, "hidden_dim", 128))
    z_c = torch.randn(B, d_in) * 3.0

    out = layer(z_c)
    v = _extract_tensor(out, ["fluxes", "v", "v_ss", "steady_state_flux"])
    pl_ooh = _extract_tensor(out, ["pl_ooh", "PL_OOH", "lipid_peroxide"])
    fe2_lyso = _extract_tensor(out, ["fe2_lyso", "lyso_fe2", "labile_fe2", "fe_lyso"])
    g_rup = _extract_tensor(out, ["g_rupture", "lysosome_rupture", "rupture_prob"])
    sterol_7dhc = _extract_tensor(out, ["seven_dhc_ss", "dhc7_ss", "7dhc_ss", "sterol_7dhc", "c_7dhc"])

    gem_data = json.loads(GEM_JSON.read_text(encoding="utf-8"))
    S = torch.tensor(gem_data.get("S", gem_data.get("stoichiometric_matrix")), dtype=v.dtype)
    sv_norm = torch.linalg.vector_norm(v.double() @ S.double().T, ord=2, dim=-1)
    assert torch.all(sv_norm < 1e-5), f"Stoichiometric conservation violated: max ||S v||_2 = {sv_norm.max().item():.3e}"

    for name, tensor in [
        ("pl_ooh", pl_ooh),
        ("fe2_lyso", fe2_lyso),
        ("g_rupture", g_rup),
        ("7dhc_ss", sterol_7dhc),
    ]:
        assert torch.isfinite(tensor).all(), f"Non-finite values in {name}"
        assert (tensor >= 0.0).all(), f"Negative physical concentration/probability in {name}: min={tensor.min().item()}"
    assert (g_rup <= 1.0 + 1e-6).all(), f"Probability g_rupture > 1: max={g_rup.max().item()}"

    if isinstance(out, dict) and "rates" in out and isinstance(out["rates"], dict):
        for rname, rval in out["rates"].items():
            assert (rval > 0).all(), f"Rate constant {rname} is not strictly positive!"


def test_sec46_gate_monotonicity_at_fixed_gpx4_mrna() -> None:
    from ferroflux.layers import sec46_gate

    gate_cls = getattr(sec46_gate, "Sec46Gate")
    gate = gate_cls()
    gate.eval()

    # Construct baseline replete state vs PSTK/SEPSECS-depleted vs IPP-depleted state at identical GPX4 mRNA
    if hasattr(gate, "compute_active_gpx4"):
        fn = gate.compute_active_gpx4
    else:
        fn = gate.forward

    sig = inspect.signature(fn)
    B = 8
    gpx4_mrna = torch.full((B,), 2.5)
    high = torch.full((B,), 2.0)
    low = torch.full((B,), -3.0)

    if {"gpx4_expr", "sec_machinery", "ipp_level"}.issubset(sig.parameters):
        e_base = _extract_tensor(fn(gpx4_expr=gpx4_mrna, sec_machinery=high, ipp_level=high), ["e_gpx4_active", "active_gpx4"])
        e_low_sec = _extract_tensor(fn(gpx4_expr=gpx4_mrna, sec_machinery=low, ipp_level=high), ["e_gpx4_active", "active_gpx4"])
        e_low_ipp = _extract_tensor(fn(gpx4_expr=gpx4_mrna, sec_machinery=high, ipp_level=low), ["e_gpx4_active", "active_gpx4"])
    else:
        # Generic tensor input convention: column 0 = GPX4, subsequent columns = PSTK, SEPSECS, SEPHS2, EEFSEC, IPP
        in_dim = getattr(gate, "in_features", 6)
        x_base = torch.full((B, in_dim), 2.0)
        x_base[:, 0] = gpx4_mrna
        x_low_sec = x_base.clone()
        x_low_sec[:, 1:-1] = -3.0
        x_low_ipp = x_base.clone()
        x_low_ipp[:, -1] = -3.0
        e_base = _extract_tensor(fn(x_base), ["e_gpx4_active", "active_gpx4"])
        e_low_sec = _extract_tensor(fn(x_low_sec), ["e_gpx4_active", "active_gpx4"])
        e_low_ipp = _extract_tensor(fn(x_low_ipp), ["e_gpx4_active", "active_gpx4"])

    assert torch.all(e_low_sec < e_base), "PSTK/SEPSECS depletion must strictly reduce active Sec46-GPX4 at fixed GPX4 mRNA!"
    assert torch.all(e_low_ipp < e_base), "Mevalonate IPP depletion must strictly reduce active Sec46-GPX4 at fixed GPX4 mRNA!"


def test_monotonic_physical_laws_sterol_rta_chelator_and_c_dot_rescue() -> None:
    from ferroflux.models.ferroflux_fm import FerroFluxFM

    model = FerroFluxFM()
    model.eval()

    # Check if model exposes a clean biophysical probe/counterfactual helper or standard batch dict forward
    assert hasattr(model, "forward")
    G = getattr(model, "n_genes", getattr(model.config, "n_genes", 512) if hasattr(model, "config") else 512)
    B = 4
    base_expr = torch.ones(B, G) * 1.5

    def make_batch(**overrides: Any) -> dict[str, Any]:
        batch: dict[str, Any] = {"gene_expr": base_expr.clone()}
        batch.update(overrides)
        return batch

    # 1. Liproxstatin-1 (RTA) & DFO (Fe chelator) strictly decrease [PL-OOH] and H_ferro, leaving H_cupro unchanged
    out_challenged = model(make_batch(rta_conc=torch.zeros(B), chelator_conc=torch.zeros(B), gpx4_inhib=torch.full((B,), 5.0), elesclomol_conc=torch.full((B,), 2.0)))
    out_lip1 = model(make_batch(rta_conc=torch.full((B,), 5.0), chelator_conc=torch.zeros(B), gpx4_inhib=torch.full((B,), 5.0), elesclomol_conc=torch.full((B,), 2.0)))
    out_dfo = model(make_batch(rta_conc=torch.zeros(B), chelator_conc=torch.full((B,), 5.0), gpx4_inhib=torch.full((B,), 5.0), elesclomol_conc=torch.full((B,), 2.0)))

    plooh_0 = _extract_tensor(out_challenged, ["pl_ooh", "PL_OOH"])
    plooh_lip1 = _extract_tensor(out_lip1, ["pl_ooh", "PL_OOH"])
    plooh_dfo = _extract_tensor(out_dfo, ["pl_ooh", "PL_OOH"])
    h_ferro_0 = _extract_tensor(out_challenged, ["h_ferro", "H_ferro", "ferroptosis_hazard"])
    h_ferro_lip1 = _extract_tensor(out_lip1, ["h_ferro", "H_ferro", "ferroptosis_hazard"])
    h_ferro_dfo = _extract_tensor(out_dfo, ["h_ferro", "H_ferro", "ferroptosis_hazard"])

    assert torch.all(plooh_lip1 < plooh_0), "Liproxstatin-1 must strictly lower [PL-OOH]!"
    assert torch.all(h_ferro_lip1 < h_ferro_0), "Liproxstatin-1 must strictly lower H_ferro!"
    assert torch.all(plooh_dfo < plooh_0), "DFO iron chelation must strictly lower [PL-OOH]!"
    assert torch.all(h_ferro_dfo < h_ferro_0), "DFO iron chelation must strictly lower H_ferro!"

    h_cupro_0 = _extract_tensor(out_challenged, ["h_cupro", "H_cupro", "cuproptosis_hazard"])
    h_cupro_lip1 = _extract_tensor(out_lip1, ["h_cupro", "H_cupro", "cuproptosis_hazard"])
    h_cupro_dfo = _extract_tensor(out_dfo, ["h_cupro", "H_cupro", "cuproptosis_hazard"])
    assert torch.allclose(h_cupro_0, h_cupro_lip1, atol=1e-5), "Liproxstatin-1 must NOT alter Elesclomol cuproptosis hazard!"
    assert torch.allclose(h_cupro_0, h_cupro_dfo, atol=1e-5), "DFO must NOT alter Elesclomol cuproptosis hazard!"

    # 2. Opposing Sterol Flux Law: DHCR7 KO raises [7-DHC]_ss & lowers H_ferro; EBP KO lowers [7-DHC]_ss & raises H_ferro
    out_wt = model(make_batch(gpx4_inhib=torch.full((B,), 3.0)))
    out_dhcr7_ko = model(make_batch(gpx4_inhib=torch.full((B,), 3.0), dhcr7_mult=torch.full((B,), 0.05)))
    out_ebp_ko = model(make_batch(gpx4_inhib=torch.full((B,), 3.0), ebp_mult=torch.full((B,), 0.05)))

    dhc_wt = _extract_tensor(out_wt, ["seven_dhc_ss", "dhc7_ss", "7dhc_ss", "sterol_7dhc"])
    dhc_dhcr7 = _extract_tensor(out_dhcr7_ko, ["seven_dhc_ss", "dhc7_ss", "7dhc_ss", "sterol_7dhc"])
    dhc_ebp = _extract_tensor(out_ebp_ko, ["seven_dhc_ss", "dhc7_ss", "7dhc_ss", "sterol_7dhc"])
    hf_wt = _extract_tensor(out_wt, ["h_ferro", "H_ferro", "ferroptosis_hazard"])
    hf_dhcr7 = _extract_tensor(out_dhcr7_ko, ["h_ferro", "H_ferro", "ferroptosis_hazard"])
    hf_ebp = _extract_tensor(out_ebp_ko, ["h_ferro", "H_ferro", "ferroptosis_hazard"])

    assert torch.all(dhc_dhcr7 > dhc_wt) and torch.all(hf_dhcr7 < hf_wt), (
        "DHCR7 suppression must accumulate [7-DHC]_ss and lower H_ferro!"
    )
    assert torch.all(dhc_ebp < dhc_wt) and torch.all(hf_ebp > hf_wt), (
        "Upstream EBP suppression must deplete [7-DHC]_ss and elevate H_ferro!"
    )

    # 3. Ultrasmall (<8 nm) C' dot + cystine starvation elevates [Fe2+]_lyso, g_rupture, & H_ferro,
    #    and is rescued by Concanamycin A (V-ATPase block), DFO, or Liproxstatin-1!
    np_cdot = torch.tensor([[6.0, -5.0, 1.0, 1.0, 10.0]] * B)  # R_h=6nm, zeta, Fe_load, receptor_match, dose
    out_replete = model(make_batch(cys2_media=torch.ones(B)))
    out_cdot_starve = model(make_batch(cys2_media=torch.full((B,), 0.02), np_params=np_cdot))
    out_cdot_cona = model(make_batch(cys2_media=torch.full((B,), 0.02), np_params=np_cdot, concanamycin_a=torch.ones(B) * 5.0))
    out_cdot_dfo = model(make_batch(cys2_media=torch.full((B,), 0.02), np_params=np_cdot, chelator_conc=torch.ones(B) * 5.0))
    out_cdot_lip1 = model(make_batch(cys2_media=torch.full((B,), 0.02), np_params=np_cdot, rta_conc=torch.ones(B) * 5.0))

    fe_rep = _extract_tensor(out_replete, ["fe2_lyso", "lyso_fe2"])
    fe_cdot = _extract_tensor(out_cdot_starve, ["fe2_lyso", "lyso_fe2"])
    grup_rep = _extract_tensor(out_replete, ["g_rupture", "lysosome_rupture"])
    grup_cdot = _extract_tensor(out_cdot_starve, ["g_rupture", "lysosome_rupture"])
    hf_rep = _extract_tensor(out_replete, ["h_ferro", "H_ferro"])
    hf_cdot = _extract_tensor(out_cdot_starve, ["h_ferro", "H_ferro"])

    assert torch.all(fe_cdot > fe_rep), "C' dot + cystine starvation must elevate [Fe2+]_lyso!"
    assert torch.all(grup_cdot > grup_rep), "C' dot + cystine starvation must elevate Gal3 lysosome rupture g_rupture!"
    assert torch.all(hf_cdot > hf_rep), "C' dot + cystine starvation must elevate H_ferro!"

    assert torch.all(_extract_tensor(out_cdot_cona, ["h_ferro", "H_ferro"]) < hf_cdot), "Concanamycin A must rescue C' dot ferroptosis!"
    assert torch.all(_extract_tensor(out_cdot_dfo, ["h_ferro", "H_ferro"]) < hf_cdot), "DFO must rescue C' dot ferroptosis!"
    assert torch.all(_extract_tensor(out_cdot_lip1, ["h_ferro", "H_ferro"]) < hf_cdot), "Liproxstatin-1 must rescue C' dot ferroptosis!"


def test_zero_bypass_gradient_from_h_ferro_around_diff_ferro_gem() -> None:
    from ferroflux.models.ferroflux_fm import FerroFluxFM

    model = FerroFluxFM()
    model.train()
    G = getattr(model, "n_genes", getattr(model.config, "n_genes", 512) if hasattr(model, "config") else 512)
    x = torch.randn(4, G, requires_grad=True)

    # 1. Normal forward/backward: d(H_ferro)/dx must be finite and non-zero
    out_normal = model({"gene_expr": x, "gpx4_inhib": torch.ones(4) * 2.0})
    h_ferro = _extract_tensor(out_normal, ["h_ferro", "H_ferro", "ferroptosis_hazard"])
    h_ferro.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0, "Normal gradient through DiffFerroGEM to gene_expr should be non-zero!"

    # 2. Detach DiffFerroGEM output via forward hook: d(H_ferro)/dx MUST be identically ZERO!
    gem_submodules = [
        m for name, m in model.named_modules() if "differrogem" in type(m).__name__.lower() or name.endswith("diff_ferro_gem")
    ]
    assert gem_submodules, "FerroFluxFM must contain a DiffFerroGEM submodule!"

    def _detach_all(module: Any, inputs: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output.detach()
        if isinstance(output, dict):
            return {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in output.items()}
        if isinstance(output, tuple):
            return tuple(v.detach() if isinstance(v, torch.Tensor) else v for v in output)
        return output

    x2 = torch.randn(4, G, requires_grad=True)
    handle = gem_submodules[0].register_forward_hook(_detach_all)
    try:
        out_detached = model({"gene_expr": x2, "gpx4_inhib": torch.ones(4) * 2.0})
        h_ferro_det = _extract_tensor(out_detached, ["h_ferro", "H_ferro", "ferroptosis_hazard"])
        if h_ferro_det.requires_grad:
            h_ferro_det.sum().backward()
            grad_norm = 0.0 if x2.grad is None else x2.grad.abs().max().item()
            assert grad_norm == 0.0, (
                f"SKIP-CONNECTION LEAK DETECTED: H_ferro received gradient {grad_norm:.3e} from gene_expr "
                "even when DiffFerroGEM outputs were completely detached!"
            )
        else:
            # Even stronger proof: h_ferro_det has no grad_fn at all wrt x2 once DiffFerroGEM is detached!
            assert x2.grad is None
    finally:
        handle.remove()
