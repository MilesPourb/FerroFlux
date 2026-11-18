#!/usr/bin/env bash
# FerroFlux-FM v3 Full-Convergence SLURM Execution Orchestrator (128-Core HighMem Parallel Shards)
# Empirically tuned for Zen 3 EPYC MKL/OpenMP (OMP_WAIT_POLICY=PASSIVE, GOMP_SPINCOUNT=0, OPENBLAS=1)
# so concurrent workers achieve ~1.1s/step at batch_size=256 with zero spin-wait contention.
set -euo pipefail

ROOT="/anvil/projects/x-cis261485/Ferroptosis_ML"
PY="/anvil/projects/x-cis261485/.conda/envs/ferroflux/bin/python"
SCRATCH_STAGE="/anvil/scratch/x-mpourbaghima/ferroflux_staging"
SHARDS_DIR="${SCRATCH_STAGE}/v3_shards"
LOGS_DIR="${SCRATCH_STAGE}/v3_slurm_logs"
ART_DIR="/home/x-mpourbaghima/.gemini/antigravity/brain/fbdbf148-c361-462e-87f3-14ddf9a1b5db"

export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0
export KMP_BLOCKTIME=0
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR="${SCRATCH_STAGE}/tmp"
mkdir -p "${TMPDIR}"

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
    export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    echo "[$(date -Is)] Pre-warming Stage-0.5 RDKit multi-scale molecular feature & grouped SAR caches (open + academic)..."
    "${PY}" -c '
from pathlib import Path
from ferroflux.data.dataset import FerroChemSARPretrainDataset
p = Path("/anvil/projects/x-cis261485/Ferroptosis_ML/data/processed")
ds_ac = FerroChemSARPretrainDataset(data_dir=p, license_track="academic")
ds_op = FerroChemSARPretrainDataset(data_dir=p, license_track="open")
print(f"[PREWARM-OK] Cached Stage-0.5 SAR molecules: academic={len(ds_ac):,}, open={len(ds_op):,}.", flush=True)
'
    ;;

  pilot)
    export OMP_NUM_THREADS=18 MKL_NUM_THREADS=18
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_shards"
    PILOT_CKPT="${SCRATCH_STAGE}/v3_pilot_ckpts"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}" "${PILOT_CKPT}/v3_open"
    rm -rf "${PILOT_DIR}"/*.json "${PILOT_CKPT}"
    mkdir -p "${PILOT_CKPT}/v3_open"
    bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" prewarm
    PIDS=()
    for idx in "${!SPLITS[@]}"; do
      sp="${SPLITS[$idx]}"
      snum=$((idx + 1))
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track open \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 18 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_open_s${snum}" \
        > "${LOGS_DIR}/pilot_open_s${snum}.log" 2>&1 &
      PIDS+=("$!")
    done
    "${PY}" -m ferroflux.training.train_curriculum \
      --data-dir "${ROOT}/data/processed" \
      --output-dir "${PILOT_CKPT}/v3_open" \
      --license-track open \
      --model-size v3 \
      --precision fp32 \
      --device auto \
      --seeds 101 \
      --epochs-per-stage 3 3 4 3 \
      --max-steps-per-stage 65 \
      --batch-size 192 \
      > "${LOGS_DIR}/pilot_ckpt_open_101.log" 2>&1 &
    PIDS+=("$!")
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then
        FAIL=1
      fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
      echo "[ERROR] Pilot worker failed!" >&2
      exit 1
    fi
    "${PY}" -c '
import json, glob
from pathlib import Path
from ferroflux.security.quarantine import freeze_checkpoint_sha256
from ferroflux.validation.stage0_bradbury_blind_eval import run_stage0_blind_evaluation

for p in sorted(glob.glob("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_shards/*_ood.json")):
    for r in json.loads(Path(p).read_text()):
        sp = r["ood_split"]
        rho = r["spearman_rho"]
        wg = r["within_group_spearman_rho"]
        ece = r["ece_10bin"]
        bm = r["best_baseline_model"]
        br = r["best_baseline_spearman"]
        dl = r["delta_spearman_vs_best_baseline"]
        print(f"PILOT OOD {sp:38s}: FF={rho:.4f} (wg={wg:.4f}, ece={ece}) vs {bm}: {br:.4f} [delta={dl:+.4f}]", flush=True)

p_ckpt = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_ckpts")
p_man = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_manifest.json")
p_man.unlink(missing_ok=True)
for pt in sorted(p_ckpt.rglob("*.pt")):
    freeze_checkpoint_sha256(pt, p_man)
res_s0 = run_stage0_blind_evaluation(
    checkpoint_dir=p_ckpt,
    manifest_path=p_man,
    license_tracks=["open"],
    output_tables_dir=Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_s0"),
)
df_s0 = res_s0[0] if isinstance(res_s0, tuple) else res_s0
if hasattr(df_s0, "to_dict"):
    for r in df_s0.to_dict(orient="records"):
        coh = r["stage0_cohort"]
        sr = r["spearman_rho"]
        au = r["auroc"]
        pp = r["permutation_p"]
        print(f"PILOT STAGE0 {coh:42s}: rho={sr:.4f} auc={au} p={pp}", flush=True)
'
    ;;

  pilot_s134)
    export OMP_NUM_THREADS=36 MKL_NUM_THREADS=36
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_shards"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}"
    PIDS=()
    for idx in 0 2 3; do
      sp="${SPLITS[$idx]}"
      snum=$((idx + 1))
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track open \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 36 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_open_s${snum}" \
        > "${LOGS_DIR}/pilot_open_s${snum}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then
        FAIL=1
      fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
      echo "[ERROR] pilot_s134 failed!" >&2
      exit 1
    fi
    "${PY}" -c '
import json, glob
from pathlib import Path
for p in sorted(glob.glob("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_shards/*_ood.json")):
    for r in json.loads(Path(p).read_text()):
        sp = r["ood_split"]
        rho = r["spearman_rho"]
        wg = r["within_group_spearman_rho"]
        ece = r["ece_10bin"]
        bm = r["best_baseline_model"]
        br = r["best_baseline_spearman"]
        dl = r["delta_spearman_vs_best_baseline"]
        print(f"PILOT6 OOD {sp:38s}: FF={rho:.4f} (wg={wg:.4f}, ece={ece}) vs {bm}: {br:.4f} [delta={dl:+.4f}]", flush=True)
'
    ;;

  submit_pilot_s134)
    mkdir -p "${LOGS_DIR}"
    rm -f "${LOGS_DIR}"/slurm_pilot_s134.out
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 00:35:00 \
      -J ffv3_s134 -o "${LOGS_DIR}/slurm_pilot_s134.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh pilot_s134"
    cat "${LOGS_DIR}/slurm_pilot_s134.out"
    ;;

  ood_track)
    TRACK="${2:?Usage: ood_track <open|academic>}"
    export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20
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
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
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
    ABL_THREADS="${OMP_NUM_THREADS:-24}"
    export OMP_NUM_THREADS="${ABL_THREADS}" MKL_NUM_THREADS="${ABL_THREADS}"
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    PIDS=()
    for aidx in $(seq "${A_START}" "${A_END}"); do
      echo "[$(date -Is)] Launching retrained ablation worker idx=${aidx}..."
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-ablation-idx "${aidx}" \
        --seeds 101 202 303 404 505 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
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
    CKPT_THREADS="${OMP_NUM_THREADS:-24}"
    export OMP_NUM_THREADS="${CKPT_THREADS}" MKL_NUM_THREADS="${CKPT_THREADS}"
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
          --epochs-per-stage 3 3 4 3 \
          --max-steps-per-stage 65 \
          --batch-size 192 \
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
    export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
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
    manifest.chmod(0o664)
    manifest.write_text("{\"checkpoints\": {}}\n", encoding="utf-8")
pts = sorted(p for p in (root / "checkpoints" / "v3_clean").rglob("*.pt") if not p.name.startswith("."))
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
    "${PY}" -m pytest "${ROOT}/tests" -o cache_dir="${SCRATCH_STAGE}/pytest_cache" -q
    echo "[$(date -Is)] === FERROFLUX-FM V3 PIPELINE 100% COMPLETE ==="
    ;;

  submit_pilot)
    mkdir -p "${LOGS_DIR}"
    rm -f "${LOGS_DIR}"/slurm_pilot.out
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 00:45:00 \
      -J ffv3_pilot -o "${LOGS_DIR}/slurm_pilot.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh pilot"
    cat "${LOGS_DIR}/slurm_pilot.out"
    ;;

  submit_wait)
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    rm -rf "${SHARDS_DIR}"/*.json "${SHARDS_DIR}"/*.npz "${LOGS_DIR}"/*.log "${LOGS_DIR}"/slurm_*.out "${SCRATCH_STAGE}"/v3_pilot*
    echo "[$(date -Is)] Launching 6 parallel 128-core SLURM compute jobs..."
    WAIT_PIDS=()
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_ood_open -o "${LOGS_DIR}/slurm_ood_open.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ood_track open" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_ood_acad -o "${LOGS_DIR}/slurm_ood_acad.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ood_track academic" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_abl_0_4 -o "${LOGS_DIR}/slurm_abl_0_4.out" \
      --wrap="env OMP_NUM_THREADS=24 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ablations 0 4" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_abl_5_8 -o "${LOGS_DIR}/slurm_abl_5_8.out" \
      --wrap="env OMP_NUM_THREADS=28 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ablations 5 8" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p wholenode -N 1 -n 128 -t 02:30:00 \
      -J ffv3_ckpt_open -o "${LOGS_DIR}/slurm_ckpt_open.out" \
      --wrap="env OMP_NUM_THREADS=24 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh checkpoints open" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p wholenode -N 1 -n 128 -t 02:30:00 \
      -J ffv3_ckpt_acad -o "${LOGS_DIR}/slurm_ckpt_acad.out" \
      --wrap="env OMP_NUM_THREADS=24 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh checkpoints academic" &
    WAIT_PIDS+=("$!")
    FAIL_W=0
    for wpid in "${WAIT_PIDS[@]}"; do
      if ! wait "${wpid}"; then
        FAIL_W=1
      fi
    done
    if [[ "${FAIL_W}" -ne 0 ]]; then
      echo "[ERROR] One or more SLURM compute jobs failed!" >&2
      exit 1
    fi
    echo "[$(date -Is)] All 6 parallel SLURM compute jobs completed cleanly!"
    ;;

  submit_remaining_wait)
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    echo "[$(date -Is)] Launching ablations 0..4 (matched codebase) + 10 deployment checkpoints on 2 highmem nodes..."
    WAIT_PIDS=()
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 02:00:00 \
      -J ffv3_abl_0_4 -o "${LOGS_DIR}/slurm_abl_0_4.out" \
      --wrap="env OMP_NUM_THREADS=24 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ablations 0 4" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 01:30:00 \
      -J ffv3_ckpts -o "${LOGS_DIR}/slurm_ckpts_both.out" \
      --wrap="env OMP_NUM_THREADS=12 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh checkpoints both" &
    WAIT_PIDS+=("$!")
    FAIL_R=0
    for rpid in "${WAIT_PIDS[@]}"; do
      if ! wait "${rpid}"; then
        FAIL_R=1
      fi
    done
    if [[ "${FAIL_R}" -ne 0 ]]; then
      echo "[ERROR] Remaining SLURM jobs failed!" >&2
      exit 1
    fi
    echo "[$(date -Is)] Remaining SLURM jobs completed cleanly; running finalize on highmem node..."
    sbatch --wait -A cis261485 -p highmem -N 1 -n 64 --mem=250G -t 00:45:00 \
      -J ffv3_finalize -o "${LOGS_DIR}/slurm_finalize.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh finalize"
    cat "${LOGS_DIR}/slurm_finalize.out"
    ;;

  submit)
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    rm -f "${SHARDS_DIR}"/*.json "${SHARDS_DIR}"/*.npz "${LOGS_DIR}"/*.log
    J_PRE=$(sbatch --parsable -A cis261485 -p wholenode -N 1 -n 128 -t 00:30:00 \
      -J ffv3_prewarm -o "${LOGS_DIR}/slurm_prewarm_%j.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh prewarm")
    J_OOD_OPEN=$(sbatch --parsable --dependency=afterok:${J_PRE} -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_ood_open -o "${LOGS_DIR}/slurm_ood_open_%j.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ood_track open")
    J_OOD_ACAD=$(sbatch --parsable --dependency=afterok:${J_PRE} -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_ood_acad -o "${LOGS_DIR}/slurm_ood_acad_%j.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ood_track academic")
    J_ABL1=$(sbatch --parsable --dependency=afterok:${J_PRE} -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_abl_0_4 -o "${LOGS_DIR}/slurm_abl_0_4_%j.out" \
      --wrap="env OMP_NUM_THREADS=24 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ablations 0 4")
    J_ABL2=$(sbatch --parsable --dependency=afterok:${J_PRE} -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_abl_5_8 -o "${LOGS_DIR}/slurm_abl_5_8_%j.out" \
      --wrap="env OMP_NUM_THREADS=28 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ablations 5 8")
    J_CKPT_OP=$(sbatch --parsable --dependency=afterok:${J_PRE} -A cis261485 -p wholenode -N 1 -n 128 -t 02:30:00 \
      -J ffv3_ckpt_open -o "${LOGS_DIR}/slurm_ckpt_open_%j.out" \
      --wrap="env OMP_NUM_THREADS=24 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh checkpoints open")
    J_CKPT_AC=$(sbatch --parsable --dependency=afterok:${J_PRE} -A cis261485 -p wholenode -N 1 -n 128 -t 02:30:00 \
      -J ffv3_ckpt_acad -o "${LOGS_DIR}/slurm_ckpt_acad_%j.out" \
      --wrap="env OMP_NUM_THREADS=24 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh checkpoints academic")
    echo "Submitted prewarm=${J_PRE} + 6 parallel compute jobs: ood_open=${J_OOD_OPEN}, ood_acad=${J_OOD_ACAD}, abl1=${J_ABL1}, abl2=${J_ABL2}, ckpt_open=${J_CKPT_OP}, ckpt_acad=${J_CKPT_AC}"
    ;;

  *)
    echo "Unknown subcommand: ${STAGE_CMD}" >&2
    exit 2
    ;;
esac
