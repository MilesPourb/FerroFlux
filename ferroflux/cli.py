"""Turnkey Command-Line Interface for FerroFlux-FM (`ferroflux` / `python -m ferroflux.cli`).

Provides zero-shot and checkpoint-loaded mechanistic ferroptosis inference from:
- Transcriptome matrices (`.csv`, `.tsv`, `.parquet`, `.h5ad`), automatically mapped to the
  canonical 2,048-gene vocabulary (`data/processed/gene_vocab.json`)
- Small-molecule SMILES strings (live RDKit 2048-bit Morgan FP + 12 physicochemical/warhead
  descriptors) or named ferroptosis probes (`RSL3`, `ML210`, `Erastin`, `IKE`, `Liproxstatin-1`,
  `Ferrostatin-1`, `DFO`, `FIN56`, `iFSP1`, `Brequinar`, `Fluvastatin`, `Concanamycin A`,
  `Elesclomol`)
- Parameter-free physical nanoparticle specifications (`--np-radius-nm`, `--np-zeta-mv`,
  `--np-fe-load`, `--np-receptor`, `--np-dose-um`)
- Extracellular cystine / nutrient boundary conditions (`--cys2-um`, `--ifng-ng-ml`)
- Optional `eval()`-only AlphaGenome 2 cis-regulatory predictions (`--alphagenome-json`)
- Optional 2D tissue spatial coordinates (`--coords-csv`) for trigger-wave velocity & Moran's I
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ferroflux.models.ferroflux_fm import IMMUNOGENIC_6_PATHWAYS, FerroFluxFM
from ferroflux.security.license_guard import assert_ag2_inference_only
from ferroflux.security.quarantine import audit_parquet_directory, verify_frozen_checkpoint

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
DEFAULT_VOCAB = ROOT_DIR / "data" / "processed" / "gene_vocab.json"


def _load_vocab(vocab_path: Path = DEFAULT_VOCAB) -> tuple[int, dict[str, int], dict[str, str]]:
    if vocab_path.is_file():
        d = json.loads(vocab_path.read_text(encoding="utf-8"))
        return (
            int(d.get("n_genes", 2048)),
            {str(k).upper(): int(v) for k, v in d.get("gene_to_idx", {}).items()},
            {str(k).upper(): str(v).upper() for k, v in d.get("alias_to_canonical", {}).items()},
        )
    return 2048, {}, {}


def _read_expression_matrix(path: Path, n_genes: int, gene_to_idx: dict[str, int], aliases: dict[str, str]) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t", index_col=0)
    elif suffix == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(path)
        mat = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        df = pd.DataFrame(mat, index=adata.obs_names.astype(str), columns=adata.var_names.astype(str))
    else:
        df = pd.read_csv(path, index_col=0)

    # Orient so rows = samples, columns = genes if genes appear on index
    if gene_to_idx:
        col_hits = sum(1 for c in df.columns if aliases.get(str(c).upper(), str(c).upper()) in gene_to_idx)
        idx_hits = sum(1 for r in df.index if aliases.get(str(r).upper(), str(r).upper()) in gene_to_idx)
        if idx_hits > col_hits and idx_hits >= 5:
            df = df.T

    sample_ids = [str(s) for s in df.index]
    bsz = len(sample_ids)
    expr = np.zeros((bsz, n_genes), dtype=np.float32)
    mask = np.zeros((bsz, n_genes), dtype=np.float32)

    if gene_to_idx:
        matched_any = False
        for col in df.columns:
            sym = aliases.get(str(col).strip().upper(), str(col).strip().upper())
            if sym in gene_to_idx:
                g_idx = gene_to_idx[sym]
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                expr[:, g_idx] = vals
                mask[:, g_idx] = 1.0
                matched_any = True
        if matched_any:
            return sample_ids, torch.from_numpy(expr), torch.from_numpy(mask)

    # Positional fallback when columns are unnamed / already aligned
    num_df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    raw = num_df.to_numpy(dtype=np.float32)
    copy_g = min(raw.shape[1], n_genes)
    expr[:, :copy_g] = raw[:, :copy_g]
    mask[:, :copy_g] = 1.0
    return sample_ids, torch.from_numpy(expr), torch.from_numpy(mask)


def _featurize_smiles(smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2048-bit Morgan FP + 12 physicochemical/mechanistic descriptors via RDKit."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_np = mfpgen.GetFingerprintAsNumPy(mol).astype(np.float32)

    # Substructure SMARTS motifs matching chem_desc_a[6..11]
    smarts_motifs = [
        "[$(C(=O)CCl),$(C(=O)C=C),$(N(=O)=O)]",  # 6: covalent_warhead (chloroacetamide / Michael)
        "[$(c1ccccc1N),$(c1cc(O)ccc1)]",        # 7: rta_motif (diarylamine / hindered phenol)
        "[$(C(=O)NO),$(c1cc(O)c(O)cc1)]",       # 8: fe_chelator (hydroxamate / catechol)
        "OO",                                    # 9: endoperoxide (FINO2-like)
        "CC(O)CC(O)CC(=O)O",                    # 10: statin_hmgcr
        "C1CCC2C(C1)CCC3C2CCC4C3CCC4",          # 11: dhcr7_sterol
    ]
    motif_hits = []
    for sma in smarts_motifs:
        patt = Chem.MolFromSmarts(sma)
        motif_hits.append(1.0 if (patt is not None and mol.HasSubstructMatch(patt)) else 0.0)

    desc = np.array(
        [
            Descriptors.MolWt(mol) / 500.0,
            Crippen.MolLogP(mol) / 5.0,
            Descriptors.TPSA(mol) / 140.0,
            Lipinski.NumHDonors(mol) / 5.0,
            Lipinski.NumHAcceptors(mol) / 10.0,
            Lipinski.NumRotatableBonds(mol) / 10.0,
            *motif_hits,
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(fp_np), torch.from_numpy(desc)


def _load_or_init_model(checkpoint: Path | None, n_genes: int, device: torch.device, sha256_manifest: Path | None = None) -> FerroFluxFM:
    if checkpoint is not None and checkpoint.is_file():
        if sha256_manifest is not None and sha256_manifest.is_file():
            verify_frozen_checkpoint(checkpoint, sha256_manifest)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_init_kwargs" in payload:
            model = FerroFluxFM(**payload["model_init_kwargs"])
            model.load_state_dict(payload["model_state_dict"])
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            model = FerroFluxFM.from_config(variant=payload.get("model_size", "base"), n_genes=int(payload.get("n_genes", n_genes)))
            model.load_state_dict(payload["model_state_dict"])
        else:
            model = FerroFluxFM(n_genes=n_genes)
            model.load_state_dict(payload)
    else:
        # Auto-discover latest trained checkpoint if available, else instantiate clean Base model
        auto_ckpts = sorted((ROOT_DIR / "checkpoints").rglob("ferroflux_*.pt")) if (ROOT_DIR / "checkpoints").is_dir() else []
        if auto_ckpts:
            return _load_or_init_model(auto_ckpts[0], n_genes=n_genes, device=device, sha256_manifest=sha256_manifest)
        model = FerroFluxFM.from_config(variant="base", n_genes=n_genes)

    model.to(device)
    model.eval()
    return model


def _cmd_predict(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    n_genes_vocab, gene_to_idx, aliases = _load_vocab(args.gene_vocab)
    model = _load_or_init_model(args.checkpoint, n_genes=n_genes_vocab, device=device, sha256_manifest=args.verify_sha256)
    model.eval()

    sample_ids, gene_expr, gene_mask = _read_expression_matrix(
        args.expression_csv,
        n_genes=model.n_genes,
        gene_to_idx=gene_to_idx,
        aliases=aliases,
    )
    bsz = gene_expr.shape[0]

    batch: dict[str, Any] = {
        "gene_expr": gene_expr.to(device),
        "gene_mask": gene_mask.to(device),
        "cys2_media": torch.full((bsz,), max(0.0, float(args.cys2_um) / 200.0), device=device),
        "ifng_media": torch.full((bsz,), max(0.0, float(args.ifng_ng_ml) / 10.0), device=device),
        "delta_t": torch.full((bsz,), float(args.exposure_hours), device=device),
        "rta_conc": torch.full((bsz,), float(args.rta_um), device=device),
        "chelator_conc": torch.full((bsz,), float(args.chelator_um), device=device),
        "concanamycin_a": torch.full((bsz,), float(args.concanamycin_um), device=device),
    }

    if args.smiles:
        fp, desc = _featurize_smiles(args.smiles)
        batch["chem_fp_a"] = fp.unsqueeze(0).expand(bsz, -1).to(device)
        batch["chem_desc_a"] = desc.unsqueeze(0).expand(bsz, -1).to(device)
        batch["dose_a"] = torch.full((bsz,), math.log10(max(0.0, float(args.dose_um)) + 1e-3), device=device)

    if args.compound:
        cname = args.compound.strip().lower().replace("-", "").replace("_", "")
        dose_val = torch.full((bsz,), float(args.dose_um), device=device)
        probe_map = {
            "rsl3": "gpx4_inhib",
            "ml210": "gpx4_inhib",
            "ml162": "gpx4_inhib",
            "fin56": "fin56_conc",
            "erastin": "xc_inhib",
            "ike": "xc_inhib",
            "ifsp1": "fsp1_inhib",
            "brequinar": "dhodh_inhib",
            "fluvastatin": "statin_conc",
            "lovastatin": "statin_conc",
            "liproxstatin1": "rta_conc",
            "lip1": "rta_conc",
            "ferrostatin1": "rta_conc",
            "fer1": "rta_conc",
            "dfo": "chelator_conc",
            "deferoxamine": "chelator_conc",
            "concanamycina": "concanamycin_a",
            "bafilomycina1": "concanamycin_a",
            "elesclomol": "elesclomol_conc",
        }
        if cname in probe_map:
            batch[probe_map[cname]] = dose_val

    if args.np_radius_nm > 0.0 and args.np_dose_um > 0.0:
        np_row = torch.tensor(
            [[float(args.np_radius_nm), float(args.np_zeta_mv), float(args.np_fe_load), float(args.np_receptor), float(args.np_dose_um)]],
            device=device,
        ).expand(bsz, -1)
        batch["np_params"] = np_row

    if args.alphagenome_json is not None:
        assert_ag2_inference_only(is_training=model.training, context="ferroflux.cli.predict")
        ag2_data = json.loads(args.alphagenome_json.read_text(encoding="utf-8"))
        tracks = torch.tensor(ag2_data["ag2_tracks"], dtype=torch.float32, device=device)
        if tracks.ndim == 2:
            tracks = tracks.unsqueeze(0).expand(bsz, -1, -1)
        batch["ag2_tracks"] = tracks
        batch["use_ag2"] = True

    with torch.no_grad():
        out = model(batch)
        c_wave_val = float("nan")
        morans_val = float("nan")
        if args.coords_csv is not None and args.coords_csv.is_file():
            cdf = pd.read_csv(args.coords_csv)
            coords = torch.tensor(cdf.iloc[:, -2:].to_numpy(dtype=np.float32), device=device)
            w_out = model.predict_spatial_wave(gene_expr.to(device), coords=coords)
            c_wave_val = float(w_out["c_wave"].item())
            morans_val = float(w_out["morans_i"].item())

    imm = out["immunogenic_pathways"].cpu().numpy()
    df_out = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "viability_pred": out["viability_pred"].cpu().numpy(),
            "ferroptosis_prob": out["ferroptosis_prob"].cpu().numpy(),
            "H_ferro": out["h_ferro"].cpu().numpy(),
            "H_cupro": out["h_cupro"].cpu().numpy(),
            "H_disulf": out["h_disulf"].cpu().numpy(),
            "H_apop": out["h_apop"].cpu().numpy(),
            "H_necro": out["h_necro"].cpu().numpy(),
            "PL_OOH": out["pl_ooh"].cpu().numpy(),
            "seven_DHC_ss": out["seven_dhc_ss"].cpu().numpy(),
            "Fe2_lyso": out["fe2_lyso"].cpu().numpy(),
            "g_rupture": out["g_rupture"].cpu().numpy(),
            "e_GPX4_active": out["e_gpx4_active"].cpu().numpy(),
            "SFTI": out["sfti_score"].cpu().numpy(),
            "cGAS_STING_score": imm[:, 0],
            "TLR_ferroptosis_score": imm[:, 1],
            "IFN_response_score": imm[:, 2],
            "CD274_PDL1_score": imm[:, 3],
            "sterol_suppression_score": imm[:, 4],
            "M1_vs_M2_TAM_score": imm[:, 5],
            "icb_response_prob": torch.sigmoid(out["icb_response_logit"]).cpu().numpy(),
            "c_wave": c_wave_val,
            "morans_i": morans_val,
        }
    )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output_csv, index=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(df_out.to_json(orient="records", indent=2), encoding="utf-8")
    if args.output_csv is None and args.output_json is None:
        print(df_out.to_json(orient="records", indent=2))
    else:
        print(
            json.dumps(
                {
                    "status": "OK",
                    "n_samples": bsz,
                    "output_csv": str(args.output_csv) if args.output_csv else None,
                    "output_json": str(args.output_json) if args.output_json else None,
                },
                indent=2,
            )
        )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    audit_parquet_directory(args.data_dir)
    print(json.dumps({"status": "CLEAN", "audited_dir": str(args.data_dir)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ferroflux",
        description="FerroFlux-FM: Mechanistic Neural-Biophysical Foundation Model CLI",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_pred = sub.add_parser("predict", help="Run mechanistic ferroptosis, SFTI & spatial-wave predictions")
    p_pred.add_argument("--expression-csv", type=Path, required=True, help="Input expression (.csv/.tsv/.parquet/.h5ad)")
    p_pred.add_argument("--checkpoint", type=Path, default=None, help="Path to frozen FerroFlux-FM .pt checkpoint")
    p_pred.add_argument("--verify-sha256", type=Path, default=None, help="Optional frozen SHA-256 manifest JSON")
    p_pred.add_argument("--gene-vocab", type=Path, default=DEFAULT_VOCAB, help="Path to gene_vocab.json")
    p_pred.add_argument("--smiles", type=str, default=None, help="Small-molecule SMILES string")
    p_pred.add_argument("--compound", type=str, default=None, help="Canonical probe name (e.g., RSL3, Erastin, Lip-1, DFO)")
    p_pred.add_argument("--dose-um", type=float, default=1.0, help="Small-molecule concentration in uM")
    p_pred.add_argument("--rta-um", type=float, default=0.0, help="Co-administered Liproxstatin-1 / Ferrostatin-1 (uM)")
    p_pred.add_argument("--chelator-um", type=float, default=0.0, help="Co-administered DFO / CPX iron chelator (uM)")
    p_pred.add_argument("--concanamycin-um", type=float, default=0.0, help="Co-administered Concanamycin A V-ATPase block (uM)")
    p_pred.add_argument("--np-radius-nm", type=float, default=0.0, help="Nanoparticle hydrodynamic diameter/size (nm, e.g. 6.0 for C' dot)")
    p_pred.add_argument("--np-zeta-mv", type=float, default=-5.0, help="Nanoparticle zeta potential (mV)")
    p_pred.add_argument("--np-fe-load", type=float, default=1.0, help="Nanoparticle surface Fe3+ loading factor")
    p_pred.add_argument("--np-receptor", type=float, default=1.0, help="Targeting ligand-receptor match factor")
    p_pred.add_argument("--np-dose-um", type=float, default=0.0, help="Nanoparticle dose (uM)")
    p_pred.add_argument("--cys2-um", type=float, default=200.0, help="Extracellular cystine in uM (200=replete, 2=starved)")
    p_pred.add_argument("--ifng-ng-ml", type=float, default=0.0, help="Extracellular IFN-gamma (ng/mL)")
    p_pred.add_argument("--exposure-hours", type=float, default=24.0, help="Perturbation exposure duration (hours)")
    p_pred.add_argument("--alphagenome-json", type=Path, default=None, help="Zero-shot eval()-only AlphaGenome 2 JSON tracks")
    p_pred.add_argument("--coords-csv", type=Path, default=None, help="Optional 2D spatial (x, y) CSV for trigger-wave rollout")
    p_pred.add_argument("--output-csv", "--output", dest="output_csv", type=Path, default=None, help="Output predictions CSV")
    p_pred.add_argument("--output-json", type=Path, default=None, help="Output predictions JSON")
    p_pred.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p_pred.set_defaults(func=_cmd_predict)

    p_aud = sub.add_parser("audit", help="Run zero-leakage quarantine & inode cap audit on Parquet directory")
    p_aud.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data" / "processed")
    p_aud.set_defaults(func=_cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
