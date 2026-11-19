#!/usr/bin/env bash
# FerroFlux-FM v3 Full-Convergence SLURM Execution Orchestrator (128-Core HighMem Parallel Shards)
# Empirically tuned for Zen 3 EPYC MKL/OpenMP (OMP_WAIT_POLICY=PASSIVE, GOMP_SPINCOUNT=0, OPENBLAS=1)
# so 3 concurrent workers/node x 40 threads achieve ~0.48s/step at batch_size=256 with zero contention.
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

STAGE_CMD="${1:-submit_wait}"

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

  pilot_node_a)
    export OMP_NUM_THREADS=40 MKL_NUM_THREADS=40
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r2"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}"
    PIDS=()
    for idx in 0 1 2; do
      sp="${SPLITS[$idx]}"
      snum=$((idx + 1))
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 40 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_acad_s${snum}" \
        > "${LOGS_DIR}/pilot_acad_s${snum}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    exit "${FAIL}"
    ;;

  pilot_node_b)
    export OMP_NUM_THREADS=40 MKL_NUM_THREADS=40
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r2"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}"
    PIDS=()
    for idx in 3 4 5; do
      sp="${SPLITS[$idx]}"
      snum=$((idx + 1))
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 40 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_acad_s${snum}" \
        > "${LOGS_DIR}/pilot_acad_s${snum}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    exit "${FAIL}"
    ;;

  pilot_node_c)
    export OMP_NUM_THREADS=40 MKL_NUM_THREADS=40
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r2"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}"
    PIDS=()
    for aidx in 0 3 4; do
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-ablation-idx "${aidx}" \
        --seeds 101 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 40 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_abl${aidx}" \
        > "${LOGS_DIR}/pilot_abl${aidx}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    exit "${FAIL}"
    ;;

  pilot_node_d)
    export OMP_NUM_THREADS=30 MKL_NUM_THREADS=30
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r2"
    PILOT_CKPT="${SCRATCH_STAGE}/v3_pilot_ckpts"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}" "${PILOT_CKPT}/v3_open" "${PILOT_CKPT}/v3_academic"
    PIDS=()
    for aidx in 2 7; do
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-ablation-idx "${aidx}" \
        --seeds 101 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 30 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_abl${aidx}" \
        > "${LOGS_DIR}/pilot_abl${aidx}.log" 2>&1 &
      PIDS+=("$!")
    done
    for trk in open academic; do
      "${PY}" -m ferroflux.training.train_curriculum \
        --data-dir "${ROOT}/data/processed" \
        --output-dir "${PILOT_CKPT}/v3_${trk}" \
        --license-track "${trk}" \
        --model-size v3 \
        --precision fp32 \
        --device auto \
        --seeds 101 \
        --epochs-per-stage 3 3 4 3 \
        --max-steps-per-stage 65 \
        --batch-size 192 \
        > "${LOGS_DIR}/pilot_ckpt_${trk}_101.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    exit "${FAIL}"
    ;;

  submit_pilot_r2)
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r2"
    PILOT_CKPT="${SCRATCH_STAGE}/v3_pilot_ckpts"
    rm -rf "${PILOT_DIR}" "${PILOT_CKPT}"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}" "${PILOT_CKPT}/v3_open" "${PILOT_CKPT}/v3_academic"
    echo "[$(date -Is)] Launching 1-seed Round-2 verification across 4 idle highmem nodes..."
    WP=()
    for sub in a b c d; do
      sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 00:45:00 \
        -J "ffr2_pil_${sub}" -o "${LOGS_DIR}/slurm_pil_${sub}.out" \
        --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh pilot_node_${sub}" &
      WP+=("$!")
    done
    FAIL=0
    for p in "${WP[@]}"; do
      if ! wait "${p}"; then FAIL=1; fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
      echo "[ERROR] Pilot job failed!" >&2
      exit 1
    fi
    "${PY}" - <<'PYEOF'
import glob, json
from pathlib import Path
from ferroflux.security.quarantine import freeze_checkpoint_sha256
from ferroflux.validation.stage0_bradbury_blind_eval import run_stage0_blind_evaluation

for p in sorted(glob.glob("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_r2/*_ood.json")):
    for r in json.loads(Path(p).read_text()):
        print(f"PILOT OOD {r['ood_split']:38s}: FF={r['spearman_rho']:.4f} (wg={r['within_group_spearman_rho']:.4f}) vs {r['best_baseline_model']}: {r['best_baseline_spearman']:.4f} [delta={r['delta_spearman_vs_best_baseline']:+.4f}]", flush=True)

abl_rows = []
for p in sorted(glob.glob("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_r2/*_abl.json")):
    abl_rows.extend(json.loads(Path(p).read_text()))
abl_rows.sort(key=lambda x: x.get("_ablation_idx", 99))
full_r = abl_rows[0]["spearman_rho"] if abl_rows else 0.0
for r in abl_rows:
    print(f"PILOT ABL {r['ablation_variant']:28s}: rho={r['spearman_rho']:.4f} [delta_vs_full={r['spearman_rho'] - full_r:+.4f}] sec46_drop={r['sec46_depletion_drop']}", flush=True)

p_ckpt = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_ckpts")
p_man = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_manifest.json")
p_man.unlink(missing_ok=True)
for pt in sorted(p_ckpt.rglob("*.pt")):
    freeze_checkpoint_sha256(pt, p_man)
res_s0 = run_stage0_blind_evaluation(
    checkpoint_dir=p_ckpt,
    manifest_path=p_man,
    license_tracks=["open", "academic"],
    output_tables_dir=Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_s0"),
)
df_s0 = res_s0[0] if isinstance(res_s0, tuple) else res_s0
for r in df_s0.to_dict(orient="records"):
    print(f"PILOT S0 [{r['license_track']:8s}] {r['stage0_cohort']:52s}: rho={r['spearman_rho']:.4f} auc={r['auroc']} p={r['permutation_p']}", flush=True)
PYEOF
    ;;

  ood_slice)
    TRACK="${2:?Usage: ood_slice <open|academic> <s_start_1based> <s_end_1based>}"
    S_START="${3:-1}"
    S_END="${4:-6}"
    OOD_THREADS="${OMP_NUM_THREADS:-40}"
    export OMP_NUM_THREADS="${OOD_THREADS}" MKL_NUM_THREADS="${OOD_THREADS}"
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    PIDS=()
    for snum in $(seq "${S_START}" "${S_END}"); do
      idx=$((snum - 1))
      sp="${SPLITS[$idx]}"
      echo "[$(date -Is)] Launching OOD worker track=${TRACK} split=${sp} (threads=${OOD_THREADS})..."
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track "${TRACK}" \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 202 303 404 505 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads "${OOD_THREADS}" \
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
      echo "[ERROR] One or more OOD shard workers failed for track=${TRACK} (${S_START}..${S_END})!" >&2
      exit 1
    fi
    echo "[$(date -Is)] Completed OOD shards (${S_START}..${S_END}) for track=${TRACK}."
    ;;

  ood_track)
    TRACK="${2:?Usage: ood_track <open|academic>}"
    OMP_NUM_THREADS=20 bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" ood_slice "${TRACK}" 1 6
    ;;

  ablations)
    A_START="${2:-0}"
    A_END="${3:-8}"
    ABL_THREADS="${OMP_NUM_THREADS:-40}"
    export OMP_NUM_THREADS="${ABL_THREADS}" MKL_NUM_THREADS="${ABL_THREADS}"
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    PIDS=()
    for aidx in $(seq "${A_START}" "${A_END}"); do
      echo "[$(date -Is)] Launching retrained ablation worker idx=${aidx} (threads=${ABL_THREADS})..."
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
    if [[ "${ONLY_TRK}" == "both" ]]; then
      TRKS=(open academic)
    else
      TRKS=("${ONLY_TRK}")
    fi
    FAIL=0
    for trk in "${TRKS[@]}"; do
      PIDS=()
      for sd in 101 202 303 404 505; do
        echo "[$(date -Is)] Launching deployment checkpoint trainer track=${trk} seed=${sd} (threads=${CKPT_THREADS})..."
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
      for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
          FAIL=1
        fi
      done
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

  submit_wait)
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    rm -rf "${SHARDS_DIR}"/*.json "${SHARDS_DIR}"/*.npz "${SHARDS_DIR}"/.lock_* "${LOGS_DIR}"/*.log "${LOGS_DIR}"/slurm_*.out
    echo "[$(date -Is)] Launching 4 QoS-compliant SLURM highmem compute jobs (MaxSubmitPU=4, 128-core nodes)..."
    WAIT_PIDS=()
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=800G -t 03:30:00 \
      -J ffv3_ood_open -o "${LOGS_DIR}/slurm_ood_open.out" \
      --wrap="env OMP_NUM_THREADS=20 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ood_slice open 1 6" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=800G -t 03:30:00 \
      -J ffv3_ood_acad -o "${LOGS_DIR}/slurm_ood_acad.out" \
      --wrap="env OMP_NUM_THREADS=20 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ood_slice academic 1 6" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=800G -t 03:30:00 \
      -J ffv3_abl_all -o "${LOGS_DIR}/slurm_abl_all.out" \
      --wrap="env OMP_NUM_THREADS=14 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh ablations 0 8" &
    WAIT_PIDS+=("$!")
    sbatch --wait -A cis261485 -p highmem -N 1 -n 128 --mem=600G -t 03:00:00 \
      -J ffv3_ckpts -o "${LOGS_DIR}/slurm_ckpts_both.out" \
      --wrap="env OMP_NUM_THREADS=24 bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh checkpoints both" &
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
    echo "[$(date -Is)] All 4 SLURM compute jobs completed cleanly! Running finalize on highmem..."
    sbatch --wait -A cis261485 -p highmem -N 1 -n 64 --mem=250G -t 00:45:00 \
      -J ffv3_finalize -o "${LOGS_DIR}/slurm_finalize.out" \
      --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh finalize"
    cat "${LOGS_DIR}/slurm_finalize.out"
    ;;

  ood_list)
    TRACK="${2:?Usage: ood_list <open|academic> <snum1> [snum2 ...]}"
    shift 2
    OOD_THREADS="${OMP_NUM_THREADS:-30}"
    export OMP_NUM_THREADS="${OOD_THREADS}" MKL_NUM_THREADS="${OOD_THREADS}"
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    PIDS=()
    for snum in "$@"; do
      idx=$((snum - 1))
      sp="${SPLITS[$idx]}"
      echo "[$(date -Is)] Launching OOD worker track=${TRACK} split=${sp} (threads=${OOD_THREADS})..."
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track "${TRACK}" \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 202 303 404 505 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads "${OOD_THREADS}" \
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
      echo "[ERROR] One or more OOD list workers failed for track=${TRACK} ($*)!" >&2
      exit 1
    fi
    echo "[$(date -Is)] Completed OOD list shards ($*) for track=${TRACK}."
    ;;

  checkpoints_seq)
    ONLY_TRK="${2:?Usage: checkpoints_seq <open|academic>}"
    CKPT_THREADS="${OMP_NUM_THREADS:-8}"
    export OMP_NUM_THREADS="${CKPT_THREADS}" MKL_NUM_THREADS="${CKPT_THREADS}"
    mkdir -p "${ROOT}/checkpoints/v3_clean/v3_${ONLY_TRK}" "${LOGS_DIR}"
    for sd in 101 202 303 404 505; do
      echo "[$(date -Is)] Running sequential checkpoint trainer track=${ONLY_TRK} seed=${sd} (threads=${CKPT_THREADS})..."
      "${PY}" -m ferroflux.training.train_curriculum \
        --data-dir "${ROOT}/data/processed" \
        --output-dir "${ROOT}/checkpoints/v3_clean/v3_${ONLY_TRK}" \
        --license-track "${ONLY_TRK}" \
        --model-size v3 \
        --precision fp32 \
        --device auto \
        --seeds "${sd}" \
        --epochs-per-stage 3 3 4 3 \
        --max-steps-per-stage 65 \
        --batch-size 192 \
        > "${LOGS_DIR}/ckpt_v3_${ONLY_TRK}_seed${sd}.log" 2>&1
    done
    echo "[$(date -Is)] Completed sequential FerroFlux-FM v3 checkpoints (${ONLY_TRK})."
    ;;

  ckpt_single)
    ONLY_TRK="${2:?Usage: ckpt_single <open|academic> <seed>}"
    SD="${3:?Usage: ckpt_single <open|academic> <seed>}"
    CKPT_THREADS="${OMP_NUM_THREADS:-8}"
    export OMP_NUM_THREADS="${CKPT_THREADS}" MKL_NUM_THREADS="${CKPT_THREADS}"
    mkdir -p "${ROOT}/checkpoints/v3_clean/v3_${ONLY_TRK}" "${LOGS_DIR}" "${SHARDS_DIR}"
    PT_FILE="${ROOT}/checkpoints/v3_clean/v3_${ONLY_TRK}/ferroflux_v3_${ONLY_TRK}_seed${SD}.pt"
    if [[ -f "${PT_FILE}" ]] && [[ $(stat -c %Y "${PT_FILE}") -gt 1795046400 ]]; then
      echo "[$(date -Is)] Checkpoint ${PT_FILE} already updated in Round 2; skipping."
      exit 0
    fi
    if ! mkdir "${SHARDS_DIR}/.lock_ckpt_${ONLY_TRK}_${SD}" 2>/dev/null; then
      echo "[$(date -Is)] Checkpoint ${ONLY_TRK} seed=${SD} already claimed; skipping."
      exit 0
    fi
    echo "[$(date -Is)] Training single v3 checkpoint track=${ONLY_TRK} seed=${SD} (threads=${CKPT_THREADS})..."
    "${PY}" -m ferroflux.training.train_curriculum \
      --data-dir "${ROOT}/data/processed" \
      --output-dir "${ROOT}/checkpoints/v3_clean/v3_${ONLY_TRK}" \
      --license-track "${ONLY_TRK}" \
      --model-size v3 \
      --precision fp32 \
      --device auto \
      --seeds "${SD}" \
      --epochs-per-stage 3 3 4 3 \
      --max-steps-per-stage 65 \
      --batch-size 192 \
      > "${LOGS_DIR}/ckpt_v3_${ONLY_TRK}_seed${SD}.log" 2>&1
    echo "[$(date -Is)] Completed single v3 checkpoint track=${ONLY_TRK} seed=${SD}."
    ;;

  locked_ood_single)
    TRACK="${2:?Usage: locked_ood_single <open|academic> <snum>}"
    SNUM="${3:?Usage: locked_ood_single <open|academic> <snum>}"
    OOD_THREADS="${OMP_NUM_THREADS:-12}"
    export OMP_NUM_THREADS="${OOD_THREADS}" MKL_NUM_THREADS="${OOD_THREADS}"
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    TAG="${TRACK}_s${SNUM}"
    if [[ -f "${SHARDS_DIR}/${TAG}_ood.json" ]]; then
      echo "[$(date -Is)] ${TAG}_ood.json already exists; skipping."
      exit 0
    fi
    if ! mkdir "${SHARDS_DIR}/.lock_${TAG}" 2>/dev/null; then
      echo "[$(date -Is)] ${TAG} already claimed by another worker; skipping."
      exit 0
    fi
    idx=$((SNUM - 1))
    sp="${SPLITS[$idx]}"
    echo "[$(date -Is)] Claimed ${TAG} (${sp}, threads=${OOD_THREADS})..."
    "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
      --model-size v3 \
      --only-track "${TRACK}" \
      --only-split "${sp}" \
      --skip-ablations \
      --seeds 101 202 303 404 505 \
      --epochs 2 \
      --sar-epochs 8 \
      --max-samples-per-shard 200000 \
      --max-threads "${OOD_THREADS}" \
      --max-test-per-split 1500 \
      --shard-out-prefix "${SHARDS_DIR}/${TAG}" \
      > "${LOGS_DIR}/ood_${TAG}.log" 2>&1
    ;;

  locked_abl_single)
    AIDX="${2:?Usage: locked_abl_single <aidx>}"
    ABL_THREADS="${OMP_NUM_THREADS:-12}"
    export OMP_NUM_THREADS="${ABL_THREADS}" MKL_NUM_THREADS="${ABL_THREADS}"
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    TAG="acad_abl${AIDX}"
    if [[ -f "${SHARDS_DIR}/${TAG}_abl.json" ]]; then
      echo "[$(date -Is)] ${TAG}_abl.json already exists; skipping."
      exit 0
    fi
    if ! mkdir "${SHARDS_DIR}/.lock_${TAG}" 2>/dev/null; then
      echo "[$(date -Is)] ${TAG} already claimed by another worker; skipping."
      exit 0
    fi
    echo "[$(date -Is)] Claimed ${TAG} (threads=${ABL_THREADS})..."
    "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
      --model-size v3 \
      --only-track academic \
      --only-ablation-idx "${AIDX}" \
      --seeds 101 202 303 404 505 \
      --epochs 2 \
      --sar-epochs 8 \
      --max-samples-per-shard 200000 \
      --max-threads "${ABL_THREADS}" \
      --max-test-per-split 1500 \
      --shard-out-prefix "${SHARDS_DIR}/${TAG}" \
      > "${LOGS_DIR}/ablation_${AIDX}.log" 2>&1
    ;;

  wave2_hm_abl)
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    CLAIMED=()
    for aidx in 0 1 2 4 5 6 7 8; do
      TAG="acad_abl${aidx}"
      if [[ ! -f "${SHARDS_DIR}/${TAG}_abl.json" ]] && mkdir "${SHARDS_DIR}/.lock_${TAG}" 2>/dev/null; then
        CLAIMED+=("${aidx}")
      fi
    done
    N_CL="${#CLAIMED[@]}"
    if [[ "${N_CL}" -eq 0 ]]; then
      echo "[$(date -Is)] All Wave-2 ablations already claimed/completed!"
      exit 0
    fi
    THR=$(( 126 / N_CL ))
    if [[ "${THR}" -gt 40 ]]; then THR=40; fi
    export OMP_NUM_THREADS="${THR}" MKL_NUM_THREADS="${THR}"
    echo "[$(date -Is)] Wave-2 HighMem Ablations claimed ${N_CL} shards (${CLAIMED[*]}) at ${THR} threads each..."
    PIDS=()
    for aidx in "${CLAIMED[@]}"; do
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-ablation-idx "${aidx}" \
        --seeds 101 202 303 404 505 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads "${THR}" \
        --max-test-per-split 1500 \
        --shard-out-prefix "${SHARDS_DIR}/acad_abl${aidx}" \
        > "${LOGS_DIR}/ablation_${aidx}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    # Also sweep any still-unclaimed Round-2 deployment checkpoints in parallel on 128 cores:
    CK_PIDS=()
    for trk in open academic; do
      for sd in 101 202 303 404 505; do
        env OMP_NUM_THREADS=12 bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" ckpt_single "${trk}" "${sd}" &
        CK_PIDS+=("$!")
      done
    done
    for pid in "${CK_PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    exit "${FAIL}"
    ;;

  wave2_hm_ood)
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    CANDIDATES=("open:1" "open:5" "open:6" "academic:1" "academic:2" "academic:3" "academic:4" "academic:5" "academic:6")
    CLAIMED=()
    for item in "${CANDIDATES[@]}"; do
      trk="${item%%:*}"
      snum="${item##*:}"
      TAG="${trk}_s${snum}"
      if [[ ! -f "${SHARDS_DIR}/${TAG}_ood.json" ]] && mkdir "${SHARDS_DIR}/.lock_${TAG}" 2>/dev/null; then
        CLAIMED+=("${item}")
      fi
    done
    N_CL="${#CLAIMED[@]}"
    if [[ "${N_CL}" -eq 0 ]]; then
      echo "[$(date -Is)] All Wave-2 OOD shards already claimed/completed!"
      exit 0
    fi
    THR=$(( 128 / N_CL ))
    if [[ "${THR}" -gt 40 ]]; then THR=40; fi
    export OMP_NUM_THREADS="${THR}" MKL_NUM_THREADS="${THR}"
    echo "[$(date -Is)] Wave-2 HighMem OOD claimed ${N_CL} shards (${CLAIMED[*]}) at ${THR} threads each..."
    PIDS=()
    for item in "${CLAIMED[@]}"; do
      trk="${item%%:*}"
      snum="${item##*:}"
      idx=$((snum - 1))
      sp="${SPLITS[$idx]}"
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track "${trk}" \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 202 303 404 505 \
        --epochs 2 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads "${THR}" \
        --max-test-per-split 1500 \
        --shard-out-prefix "${SHARDS_DIR}/${trk}_s${snum}" \
        > "${LOGS_DIR}/ood_${trk}_s${snum}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    exit "${FAIL}"
    ;;

  *)
    echo "Unknown subcommand: ${STAGE_CMD}" >&2
    exit 2
    ;;
esac
