#!/usr/bin/env bash
set -euo pipefail

ROOT="/anvil/projects/x-cis261485/Ferroptosis_ML"
PY="/anvil/projects/x-cis261485/.conda/envs/ferroflux/bin/python"
ART_DIR="/home/x-mpourbaghima/.gemini/antigravity/brain/fbdbf148-c361-462e-87f3-14ddf9a1b5db"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONPATH="${ROOT}"

cd "${ROOT}"

echo "[$(date -Is)] === STEP 0: Verify BLAS-Factorized PerceiverTrunk CPU Step Speed ==="
nice -n 10 "${PY}" -c '
import time, torch
from ferroflux.models.ferroflux_fm import FerroFluxFM
from ferroflux.models.losses import FerroFluxCurriculumLoss
from ferroflux.training.train_curriculum import SyntheticCurriculumDataset
torch.set_num_threads(8)
ds = SyntheticCurriculumDataset(n_samples=256, n_genes=2048)
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=128)))
m = FerroFluxFM(n_genes=2048, d_model=192, n_layers=3, n_heads=4)
lf = FerroFluxCurriculumLoss()
t0 = time.perf_counter()
for _ in range(5):
    m.zero_grad(set_to_none=True)
    out = m(b)
    lf(out, b, stage=3)["loss"].backward()
dt = time.perf_counter() - t0
print(f"[SPEED-CHECK] 5_steps_sec={dt:.3f}s  per_step_sec={dt/5.0:.3f}s", flush=True)
'

echo "[$(date -Is)] === STEP 1: Strictly Out-Of-Sample OOD (6 Splits x 2 Tracks x 3 Seeds) + 7 Retrained Ablations + 72 Baselines ==="
nice -n 10 "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
  --model-size v2_login --tracks open academic --seeds 101 202 303 --epochs 2 \
  --max-threads 8 --max-test-per-split 1500 \
  --output-dir "${ROOT}/results/tables"

nice -n 10 "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
  --verify --output-dir "${ROOT}/results/tables"

echo "[$(date -Is)] === STEP 2: Train Clean FerroFlux-FM v2 Deployment Ensembles (v2_open & v2_academic) ==="
rm -rf "${ROOT}/checkpoints/v2_clean"
nice -n 10 "${PY}" -m ferroflux.training.train_curriculum \
  --data-dir "${ROOT}/data/processed" \
  --output-dir "${ROOT}/checkpoints/v2_clean/v2_open" \
  --license-track open --model-size compact --precision fp32 --device cpu \
  --seeds 101 202 303 --epochs-per-stage 2 2 2 1 --max-steps-per-stage 80 --batch-size 128

nice -n 10 "${PY}" -m ferroflux.training.train_curriculum \
  --data-dir "${ROOT}/data/processed" \
  --output-dir "${ROOT}/checkpoints/v2_clean/v2_academic" \
  --license-track academic --model-size compact --precision fp32 --device cpu \
  --seeds 101 202 303 --epochs-per-stage 2 2 2 1 --max-steps-per-stage 80 --batch-size 128

echo "[$(date -Is)] === STEP 3: Cryptographic SHA-256 Lock of All 6 V2 Checkpoints ==="
nice -n 10 "${PY}" -c '
from pathlib import Path
from ferroflux.security.quarantine import freeze_checkpoint_sha256
root = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
manifest = root / "configs" / "frozen_checkpoints_v2_sha256.json"
if manifest.exists():
    manifest.unlink()
pts = sorted((root / "checkpoints" / "v2_clean").rglob("*.pt"))
assert len(pts) == 6, f"Expected 6 V2 checkpoints, found {len(pts)}"
for p in pts:
    freeze_checkpoint_sha256(p, manifest)
print(f"[SHA256-LOCK] Frozen {len(pts)} V2 checkpoints into {manifest}", flush=True)
'

nice -n 10 "${PY}" -m ferroflux.validation.stage0_bradbury_blind_eval \
  --check-locks-only \
  --checkpoint-dir "${ROOT}/checkpoints/v2_clean" \
  --manifest "${ROOT}/configs/frozen_checkpoints_v2_sha256.json"

echo "[$(date -Is)] === STEP 4: De-Confounded Stage-0 Blind Validation (Zero Circularity / Zero Tautology) ==="
nice -n 10 "${PY}" -m ferroflux.validation.stage0_bradbury_blind_eval \
  --checkpoint-dir "${ROOT}/checkpoints/v2_clean" \
  --manifest "${ROOT}/configs/frozen_checkpoints_v2_sha256.json" \
  --tracks open academic --device cpu \
  --output-tables-dir "${ROOT}/results/tables"

echo "[$(date -Is)] === STEP 5: Re-Render Honest Publication Figures 1-6 (300 DPI PNG + Vector PDF) ==="
nice -n 10 "${PY}" -m ferroflux.validation.generate_nature_figures \
  --output-dir "${ROOT}/results/figures" \
  --tables-dir "${ROOT}/results/tables" \
  --checkpoint-dir "${ROOT}/checkpoints/v2_clean"

cp "${ROOT}/results/figures/"Fig*.png "${ART_DIR}/"

echo "[$(date -Is)] === STEP 6: Post-Remediation Zero-Trust Verification & Full Pytest Suite ==="
nice -n 10 "${PY}" -m ferroflux.audit.verify_v2_remediation
nice -n 10 "${PY}" -m pytest "${ROOT}/tests" -q

echo "[$(date -Is)] === ALL V2 REMEDIATION & VALIDATION STEPS COMPLETE ==="

