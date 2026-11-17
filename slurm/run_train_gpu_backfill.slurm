#!/bin/bash
#SBATCH -A cis261485-ai
#SBATCH -p ai
#SBATCH --job-name=ferroflux_gpu_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/anvil/scratch/x-mpourbaghima/ferroflux_staging/slurm_train_gpu_%j.out
#SBATCH --error=/anvil/scratch/x-mpourbaghima/ferroflux_staging/slurm_train_gpu_%j.err
#
# Rapid GPU-Debug Backfill Usage (30-min fast turnaround on Purdue Anvil):
#   sbatch -p gpu-debug --time=00:30:00 slurm/run_train_gpu_backfill.slurm

set -euo pipefail

export ROOT_DIR="/anvil/projects/x-cis261485/Ferroptosis_ML"
export STAGING_DIR="/anvil/scratch/x-mpourbaghima/ferroflux_staging"
export TMPDIR="${STAGING_DIR}"
export HF_HOME="${STAGING_DIR}/hf_cache"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NVIDIA_TF32_OVERRIDE=1

ENV_DIR="/anvil/projects/x-cis261485/.conda/envs/ferroflux"
export PATH="${ENV_DIR}/bin:${PATH}"
PYTHON_BIN="${ENV_DIR}/bin/python"

mkdir -p "${STAGING_DIR}" "${HF_HOME}" "${ROOT_DIR}/checkpoints" "${ROOT_DIR}/results/tables" "${ROOT_DIR}/results/figures"
cd "${ROOT_DIR}"

echo "[$(date -Is)] Starting FerroFlux-FM CUDA bf16 GPU Training & OOD Evaluation on $(hostname)"
nvidia-smi || true

# 1. Pre-flight Zero-Leakage Holdout Quarantine & <500-File Inode Cap Audit
"${PYTHON_BIN}" -m ferroflux.security.quarantine --audit-data-dir "${ROOT_DIR}/data/processed"

# 2. CUDA bf16 5-Seed 4-Stage Curriculum Training ('open' + 'academic')
for TRACK in open academic; do
    "${PYTHON_BIN}" -m ferroflux.training.train_curriculum \
        --config "${ROOT_DIR}/configs/train_config.yaml" \
        --data-dir "${ROOT_DIR}/data/processed" \
        --output-dir "${ROOT_DIR}/checkpoints/${TRACK}" \
        --license-track "${TRACK}" \
        --model-size base \
        --precision bf16 \
        --seeds 101 202 303 404 505 \
        --device cuda
done

# Cryptographically lock frozen checkpoints with SHA-256 prior to Stage-0 evaluation
for CKPT in "${ROOT_DIR}"/checkpoints/*/*.pt; do
    if [[ -f "${CKPT}" ]]; then
        "${PYTHON_BIN}" -m ferroflux.security.quarantine --freeze-ckpt "${CKPT}"
    fi
done

# 3. Reviewer #2 Baselines + Full Compositional OOD & Ablation Suite
"${PYTHON_BIN}" -m ferroflux.baselines.tabular_and_chem_baselines \
    --data-dir "${ROOT_DIR}/data/processed" \
    --output-csv "${ROOT_DIR}/results/tables/baselines_ood_summary.csv" \
    --license-track both \
    --n-jobs 16

"${PYTHON_BIN}" -m ferroflux.training.evaluate_ood_and_ablations \
    --config "${ROOT_DIR}/configs/train_config.yaml" \
    --data-dir "${ROOT_DIR}/data/processed" \
    --checkpoint-dir "${ROOT_DIR}/checkpoints" \
    --output-dir "${ROOT_DIR}/results/tables" \
    --device cuda

echo "[$(date -Is)] FerroFlux-FM CUDA bf16 Training & Evaluation COMPLETE."
