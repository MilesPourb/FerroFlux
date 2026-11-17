#!/bin/bash
#SBATCH --job-name=ferroflux_etl
#SBATCH --account=cis261485
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=/anvil/scratch/x-mpourbaghima/ferroflux_staging/slurm_etl_%j.out
#SBATCH --error=/anvil/scratch/x-mpourbaghima/ferroflux_staging/slurm_etl_%j.err

set -euo pipefail

export ROOT_DIR="/anvil/projects/x-cis261485/Ferroptosis_ML"
export STAGING_DIR="/anvil/scratch/x-mpourbaghima/ferroflux_staging"
export TMPDIR="${STAGING_DIR}/tmp"
export HF_HOME="${STAGING_DIR}/hf_cache"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

mkdir -p "${TMPDIR}" "${HF_HOME}" "${ROOT_DIR}/data/processed"

PYTHON_BIN="/anvil/projects/x-cis261485/.conda/envs/ferroflux/bin/python"
cd "${ROOT_DIR}"

echo "[$(date -Is)] Starting FerroFlux-FM Phase 2A + Phase 2B Data ETL on $(hostname)"

# Run Phase 2A (Multi-Omics, Pharmacogenomics, Perturb-seq, LINCS, GEO, Bio-Nano, Clinical ICB)
"${PYTHON_BIN}" -m ferroflux.data.ingest_depmap_pharmaco \
    --staging-dir "${STAGING_DIR}/phase2a/depmap_pharmaco" \
    --output-dir "${ROOT_DIR}/data/processed" &
PID_2A_1=$!

"${PYTHON_BIN}" -m ferroflux.data.ingest_perturb_geo_bionano \
    --staging-dir "${STAGING_DIR}/phase2a/perturb_geo" \
    --output-dir "${ROOT_DIR}/data/processed" &
PID_2A_2=$!

# Run Phase 2B if scripts are present
if [[ -f "${ROOT_DIR}/ferroflux/data/ingest_proteo_metab_lipid.py" ]]; then
    "${PYTHON_BIN}" -m ferroflux.data.ingest_proteo_metab_lipid &
    PID_2B_1=$!
else
    PID_2B_1=""
fi

if [[ -f "${ROOT_DIR}/ferroflux/data/ingest_chem_synergy_epigen.py" ]]; then
    "${PYTHON_BIN}" -m ferroflux.data.ingest_chem_synergy_epigen &
    PID_2B_2=$!
else
    PID_2B_2=""
fi

wait "${PID_2A_1}"
wait "${PID_2A_2}"
[[ -n "${PID_2B_1}" ]] && wait "${PID_2B_1}"
[[ -n "${PID_2B_2}" ]] && wait "${PID_2B_2}"

echo "[$(date -Is)] Running Zero-Leakage Quarantine & <500-File Inode Audit..."
"${PYTHON_BIN}" -m ferroflux.security.quarantine --audit-data-dir "${ROOT_DIR}/data/processed"

echo "[$(date -Is)] FerroFlux-FM ETL & Zero-Leakage Audit COMPLETE."
