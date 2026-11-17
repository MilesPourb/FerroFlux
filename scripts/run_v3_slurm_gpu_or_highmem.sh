#!/usr/bin/env bash
# FerroFlux-FM v3 Full-Convergence SLURM Execution Orchestrator (GPU + 128-Core HighMem Parallel Shards)
# Zero login-node compute; zero data leakage; 5-seed ensembles across 6 OOD splits, 6 baselines, 8 ablations,
# and 10 deployment checkpoints (v3_open + v3_academic).
set -euo pipefail

ROOT="/anvil/projects/x-cis261485/Ferroptosis_ML"
PY="/anvil/projects/x-cis261485/.conda/envs/ferroflux/bin/python"
SCRATCH_STAGE="/anvil/scratch/x-mpourbaghima/ferroflux_staging"
SHARDS_DIR="${SCRATCH_STAGE}/v3_shards"
LOGS_DIR="${SCRATCH_STAGE}/v3_slurm_logs"
ART_DIR="/home/x-mpourbaghima/.gemini/antigravity/brain/fbdbf148-c361-462e-87f3-14ddf9a1b5db"

export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE=1
cd "${ROOT}"

STAGE_CMD="${1:-submit}"

SPLITS=(
  "1_leave_out_scaffold"
  "2_leave_out_lineage_ccrcc_melanoma"
  "3_2d_to_3d_organoid_pdx"
  "4_dual_crispr_fin_epistasis"
  "5_cuproptosis_elesclomol_specificity"
  "6_sfti_tumor_vs_normal_til"
)

case "${STAGE_CMD}" in
  prewarm)
    export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    echo "[$(date -Is)] Pre-warming Stage-0.5 RDKit multi-scale molecular feature cache..."
    "${PY}" -c '
from pathlib import Path
from ferroflux.data.dataset import FerroChemSARPretrainDataset
ds = FerroChemSARPretrainDataset(data_dir=Path("/anvil/projects/x-cis261485/Ferroptosis_ML/data/processed"), license_track="academic")
print(f"[PREWARM-OK] Cached {len(ds):,} unique Stage-0.5 SAR molecules.", flush=True)
'
    ;;

  ood_track)
    TRACK="${2:?Usage: ood_track <open|academic>}"
    export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 NUMEXPR_NUM_THREADS=20
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    PIDS=()
    for idx in "${!SPLITS[@]}"; do
      sp="${SPLITS[$idx]}"
      snum=$((idx + 1))
      echo "[$(date -Is)] Launching OOD worker track=${TRACK} split=${sp}..."
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track "${TRACK}" \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 202 303 404 505 \
        --epochs 3 \
        --sar-epochs 8 \
        --max-samples-per-shard 60000 \
        --max-threads 20 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${SHARDS_DIR}/${TRACK}_s${snum}" \
        > "${LOGS_DIR}/ood_${TRACK}_s${snum}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then
        FAIL=1
      fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
      echo "[ERROR] One or more OOD shard workers failed for track=${TRACK}!" >&2
      exit 1
    fi
    echo "[$(date -Is)] Completed all 6 OOD shards for track=${TRACK}."
    ;;

  ablations)
    A_START="${2:-0}"
    A_END="${3:-8}"
    ABL_THREADS="${OMP_NUM_THREADS:-8}"
    export OMP_NUM_THREADS="${ABL_THREADS}" MKL_NUM_THREADS="${ABL_THREADS}" OPENBLAS_NUM_THREADS="${ABL_THREADS}" NUMEXPR_NUM_THREADS="${ABL_THREADS}"
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    PIDS=()
    for aidx in $(seq "${A_START}" "${A_END}"); do
      echo "[$(date -Is)] Launching retrained ablation worker idx=${aidx}..."
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-ablation-idx "${aidx}" \
        --seeds 101 202 303 404 505 \
        --epochs 3 \
        --sar-epochs 8 \
        --max-samples-per-shard 60000 \
        --max-threads "${ABL_THREADS}" \
        --max-test-per-split 1500 \
        --shard-out-prefix "${SHARDS_DIR}/acad_abl${aidx}" \
        > "${LOGS_DIR}/ablation_${aidx}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then
        FAIL=1
      fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
      echo "[ERROR] One or more ablation workers (${A_START}..${A_END}) failed!" >&2
      exit 1
    fi
    echo "[$(date -Is)] Completed retrained ablation workers (${A_START}..${A_END})."
    ;;

  checkpoints)
    ONLY_TRK="${2:-both}"
    CKPT_THREADS="${OMP_NUM_THREADS:-6}"
    export OMP_NUM_THREADS="${CKPT_THREADS}" MKL_NUM_THREADS="${CKPT_THREADS}" OPENBLAS_NUM_THREADS="${CKPT_THREADS}" NUMEXPR_NUM_THREADS="${CKPT_THREADS}"
    mkdir -p "${ROOT}/checkpoints/v3_clean/v3_open" "${ROOT}/checkpoints/v3_clean/v3_academic" "${LOGS_DIR}"
    PIDS=()
    if [[ "${ONLY_TRK}" == "both" ]]; then
      TRKS=(open academic)
    else
      TRKS=("${ONLY_TRK}")
    fi
    for trk in "${TRKS[@]}"; do
      for sd in 101 202 303 404 505; do
        echo "[$(date -Is)] Launching deployment checkpoint trainer track=${trk} seed=${sd}..."
        "${PY}" -m ferroflux.training.train_curriculum \
          --data-dir "${ROOT}/data/processed" \
          --output-dir "${ROOT}/checkpoints/v3_clean/v3_${trk}" \
          --license-track "${trk}" \
          --model-size v3 \
          --precision fp32 \
          --device auto \
          --seeds "${sd}" \
          --epochs-per-stage 4 4 5 3 \
          --max-steps-per-stage 160 \
          --batch-size 128 \
          > "${LOGS_DIR}/ckpt_v3_${trk}_seed${sd}.log" 2>&1 &
        PIDS+=("$!")
      done
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then
        FAIL=1
      fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
      echo "[ERROR] One or more v3 deployment checkpoint workers failed!" >&2
      exit 1
    fi
    echo "[$(date -Is)] Completed FerroFlux-FM v3 deployment checkpoints (${ONLY_TRK})."
    ;;

  finalize)
    export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
    echo "[$(date -Is)] 1. Merging OOD & Ablation Shards + Strict-OOS Verification..."
    "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
      --merge-shards-dir "${SHARDS_DIR}" \
      --output-dir "${ROOT}/results/tables" \
      --n-perm 2000
    "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
      --verify --output-dir "${ROOT}/results/tables"

    echo "[$(date -Is)] 2. Freezing SHA-256 Manifest for All 10 FerroFlux-FM v3 Checkpoints..."
    "${PY}" -c '
from pathlib import Path
from ferroflux.security.quarantine import freeze_checkpoint_sha256
root = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
manifest = root / "configs" / "frozen_checkpoints_v3_sha256.json"
if manifest.exists():
    manifest.unlink()
pts = sorted((root / "checkpoints" / "v3_clean").rglob("*.pt"))
assert len(pts) == 10, f"Expected 10 v3 checkpoints, found {len(pts)}"
for p in pts:
    freeze_checkpoint_sha256(p, manifest)
print(f"[SHA256-V3-LOCK] Frozen {len(pts)} v3 checkpoints into {manifest}", flush=True)
'
    "${PY}" -m ferroflux.validation.stage0_bradbury_blind_eval \
      --check-locks-only \
      --checkpoint-dir "${ROOT}/checkpoints/v3_clean" \
      --manifest "${ROOT}/configs/frozen_checkpoints_v3_sha256.json"

    echo "[$(date -Is)] 3. Running De-Confounded Stage-0 Blind Validation..."
    "${PY}" -m ferroflux.validation.stage0_bradbury_blind_eval \
      --checkpoint-dir "${ROOT}/checkpoints/v3_clean" \
      --manifest "${ROOT}/configs/frozen_checkpoints_v3_sha256.json" \
      --tracks open academic \
      --device auto \
      --output-tables-dir "${ROOT}/results/tables"

    echo "[$(date -Is)] 4. Rendering Publication Figures 1-6 (300 DPI PNG + Vector PDF)..."
    "${PY}" -m ferroflux.validation.generate_nature_figures \
      --output-dir "${ROOT}/results/figures" \
      --tables-dir "${ROOT}/results/tables" \
      --checkpoint-dir "${ROOT}/checkpoints/v3_clean"
    if [[ -d "${ART_DIR}" ]]; then
      cp "${ROOT}/results/figures/"Fig*.png "${ART_DIR}/"
    fi

    echo "[$(date -Is)] 5. Running Zero-Trust Forensic Verification & Full Pytest Suite..."
    "${PY}" -m ferroflux.audit.verify_v2_remediation
    "${PY}" -m pytest "${ROOT}/tests" -q
    echo "[$(date -Is)] === FERROFLUX-FM V3 PIPELINE 100% COMPLETE ==="
    ;;

  abl_and_ckpt_1node)
    bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" prewarm
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" ablations 0 8 &
    P_ABL=$!
    OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" checkpoints both &
    P_CKPT=$!
    wait "${P_ABL}"
    wait "${P_CKPT}"
    ;;

  submit)
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    rm -f "${SHARDS_DIR}"/*.json "${SHARDS_DIR}"/*.npz
    J_OOD_OPEN=$(sbatch --parsable -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 01:30:00 \
      -J ffv3_ood_open -o "${LOGS_DIR}/slurm_ood_open_%j.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh prewarm && bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ood_track open")
    J_OOD_ACAD=$(sbatch --parsable -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 01:30:00 \
      -J ffv3_ood_acad -o "${LOGS_DIR}/slurm_ood_acad_%j.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh prewarm && bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ood_track academic")
    J_ABL_CKPT=$(sbatch --parsable -A cis261485 -p highmem -N 1 -n 128 --mem=800G -t 01:30:00 \
      -J ffv3_abl_ckpt -o "${LOGS_DIR}/slurm_abl_ckpt_%j.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh abl_and_ckpt_1node")
    J_FIN=$(sbatch --parsable --dependency=afterok:${J_OOD_OPEN}:${J_OOD_ACAD}:${J_ABL_CKPT} \
      -A cis261485 -p highmem -N 1 -n 32 --mem=160G -t 00:45:00 \
      -J ffv3_finalize -o "${LOGS_DIR}/slurm_finalize_%j.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh finalize")
    echo "Submitted 4-job highmem DAG: ood_open=${J_OOD_OPEN}, ood_acad=${J_OOD_ACAD}, abl_ckpt=${J_ABL_CKPT} -> finalize=${J_FIN}"
    ;;

  *)
    echo "Unknown subcommand: ${STAGE_CMD}" >&2
    exit 2
    ;;
esac
