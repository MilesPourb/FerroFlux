#!/bin/bash
#SBATCH -A cis261485
#SBATCH -p highmem
#SBATCH --job-name=ferroflux_train_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/anvil/scratch/x-mpourbaghima/ferroflux_staging/slurm_train_cpu_%j.out
#SBATCH --error=/anvil/scratch/x-mpourbaghima/ferroflux_staging/slurm_train_cpu_%j.err

set -euo pipefail

export ROOT_DIR="/anvil/projects/x-cis261485/Ferroptosis_ML"
export STAGING_DIR="/anvil/scratch/x-mpourbaghima/ferroflux_staging"
export TMPDIR="${STAGING_DIR}"
export HF_HOME="${STAGING_DIR}/hf_cache"
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

ENV_DIR="/anvil/projects/x-cis261485/.conda/envs/ferroflux"
export PATH="${ENV_DIR}/bin:${PATH}"
PYTHON_BIN="${ENV_DIR}/bin/python"

mkdir -p "${STAGING_DIR}" "${HF_HOME}" "${ROOT_DIR}/checkpoints" "${ROOT_DIR}/results/tables" "${ROOT_DIR}/results/figures"
cd "${ROOT_DIR}"

echo "[$(date -Is)] Starting FerroFlux-FM 128-Core High-Memory CPU Training & OOD Suite on $(hostname)"

# 1. Pre-flight Zero-Leakage Holdout Quarantine & <500-File Inode Cap Audit
echo "[$(date -Is)] Step 1/4: Auditing processed Parquet shards for zero holdout leakage..."
"${PYTHON_BIN}" -m ferroflux.security.quarantine --audit-data-dir "${ROOT_DIR}/data/processed"

# 2. 5-Seed 4-Stage Curriculum Pretraining ('open' Apache-2.0 + 'academic' tracks)
#    Runs up to 4 concurrent ensemble workers across 128 EPYC cores (4 x 32 OMP threads).
echo "[$(date -Is)] Step 2/4: Running 5-seed FerroFlux-FM curriculum pretraining (open & academic)..."
for TRACK in open academic; do
    "${PYTHON_BIN}" -m ferroflux.training.train_curriculum \
        --config "${ROOT_DIR}/configs/train_config.yaml" \
        --data-dir "${ROOT_DIR}/data/processed" \
        --output-dir "${ROOT_DIR}/checkpoints/${TRACK}" \
        --license-track "${TRACK}" \
        --model-size base \
        --seeds 101 202 303 404 505 \
        --parallel-seeds 4 \
        --device cpu
done

# Freeze SHA-256 hashes of all trained checkpoints prior to any Stage-0 blind evaluation
for CKPT in "${ROOT_DIR}"/checkpoints/*/*.pt; do
    if [[ -f "${CKPT}" ]]; then
        "${PYTHON_BIN}" -m ferroflux.security.quarantine --freeze-ckpt "${CKPT}"
    fi
done

# 3. Reviewer #2 Tabular, Morgan-FP MLP & LinearPerturb Baselines across all 6 OOD splits
echo "[$(date -Is)] Step 3/4: Running all 6 Reviewer #2 baselines across 6 OOD splits..."
"${PYTHON_BIN}" -m ferroflux.baselines.tabular_and_chem_baselines \
    --data-dir "${ROOT_DIR}/data/processed" \
    --output-csv "${ROOT_DIR}/results/tables/baselines_ood_summary.csv" \
    --license-track both \
    --n-jobs 64

# 4. Compositional OOD Evaluation & 6 Architectural Ablations
echo "[$(date -Is)] Step 4/4: Running FerroFlux-FM compositional OOD & ablation evaluation..."
"${PYTHON_BIN}" -m ferroflux.training.evaluate_ood_and_ablations \
    --config "${ROOT_DIR}/configs/train_config.yaml" \
    --data-dir "${ROOT_DIR}/data/processed" \
    --checkpoint-dir "${ROOT_DIR}/checkpoints" \
    --output-dir "${ROOT_DIR}/results/tables"

echo "[$(date -Is)] FerroFlux-FM CPU High-Memory Training, Baselines & OOD Evaluation COMPLETE."
