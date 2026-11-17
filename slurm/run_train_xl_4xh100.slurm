#!/bin/bash
#SBATCH -A cis261485-ai
#SBATCH -p ai
#SBATCH --job-name=ferroflux_xl_4xh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=/anvil/scratch/x-mpourbaghima/ferroflux_staging/slurm_train_xl_%j.out
#SBATCH --error=/anvil/scratch/x-mpourbaghima/ferroflux_staging/slurm_train_xl_%j.err

set -euo pipefail

export ROOT_DIR="/anvil/projects/x-cis261485/Ferroptosis_ML"
export STAGING_DIR="/anvil/scratch/x-mpourbaghima/ferroflux_staging"
export TMPDIR="${STAGING_DIR}"
export HF_HOME="${STAGING_DIR}/hf_cache"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ENV_DIR="/anvil/projects/x-cis261485/.conda/envs/ferroflux"
export PATH="${ENV_DIR}/bin:${PATH}"
PYTHON_BIN="${ENV_DIR}/bin/python"

mkdir -p "${STAGING_DIR}" "${HF_HOME}" "${ROOT_DIR}/checkpoints/xl_open" "${ROOT_DIR}/checkpoints/xl_academic" "${ROOT_DIR}/results/tables"
cd "${ROOT_DIR}"

TRAIN_PID=""
handle_usr1_requeue() {
    echo "[$(date -Is)] Caught SLURM SIGUSR1 (300s pre-walltime warning); flushing checkpoint & requeueing job ${SLURM_JOB_ID}..."
    if [[ -n "${TRAIN_PID}" ]]; then
        kill -USR1 "${TRAIN_PID}" 2>/dev/null || kill -TERM "${TRAIN_PID}" 2>/dev/null || true
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap 'handle_usr1_requeue' USR1

echo "[$(date -Is)] Starting FerroFlux-FM-XL Multi-GPU (4x H100 DDP bf16) Pretraining on $(hostname)"
nvidia-smi || true

# 1. Pre-flight Zero-Leakage Holdout Quarantine & <500-File Inode Cap Audit
"${PYTHON_BIN}" -m ferroflux.security.quarantine --audit-data-dir "${ROOT_DIR}/data/processed"

# 2. Multi-GPU Distributed Data Parallel Pretraining for FerroFlux-FM-XL ('open' + 'academic')
for TRACK in open academic; do
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
        -m ferroflux.training.train_curriculum \
        --config "${ROOT_DIR}/configs/train_config.yaml" \
        --data-dir "${ROOT_DIR}/data/processed" \
        --output-dir "${ROOT_DIR}/checkpoints/xl_${TRACK}" \
        --license-track "${TRACK}" \
        --model-size xl \
        --precision bf16 \
        --resume-auto \
        --seeds 101 202 303 404 505 &
    TRAIN_PID=$!
    wait "${TRAIN_PID}"
    TRAIN_PID=""
done

# 3. Cryptographically lock all XL checkpoints with SHA-256 prior to Stage-0 blind evaluation
for CKPT in "${ROOT_DIR}"/checkpoints/xl_*/*.pt; do
    if [[ -f "${CKPT}" ]]; then
        "${PYTHON_BIN}" -m ferroflux.security.quarantine --freeze-ckpt "${CKPT}"
    fi
done

# 4. Full Compositional OOD & Ablation Suite on FerroFlux-FM-XL
"${PYTHON_BIN}" -m ferroflux.training.evaluate_ood_and_ablations \
    --config "${ROOT_DIR}/configs/train_config.yaml" \
    --data-dir "${ROOT_DIR}/data/processed" \
    --checkpoint-dir "${ROOT_DIR}/checkpoints" \
    --model-size xl \
    --output-dir "${ROOT_DIR}/results/tables" \
    --device cuda

echo "[$(date -Is)] FerroFlux-FM-XL 4xH100 Pretraining, SHA-256 Lock & OOD Evaluation COMPLETE."
