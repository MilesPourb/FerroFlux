#!/usr/bin/env bash
# FerroFlux-FM v3 Full-Convergence SLURM Execution Orchestrator (128-Core HighMem + Shared Parallel Queue)
# Empirically tuned for Zen 3 EPYC MKL/OpenMP (OMP_WAIT_POLICY=PASSIVE, GOMP_SPINCOUNT=0, OPENBLAS=1)
set -euo pipefail

ROOT="/anvil/projects/x-cis261485/Ferroptosis_ML"
PY="/anvil/projects/x-cis261485/.conda/envs/ferroflux/bin/python"
SCRATCH_STAGE="/anvil/scratch/x-mpourbaghima/ferroflux_staging"
SHARDS_DIR="${SCRATCH_STAGE}/v3_shards_r8"
OOS_CKPT_DIR="${SCRATCH_STAGE}/v3_oos_ckpts_r8"
LOCKS_DIR="${SCRATCH_STAGE}/v3_locks_r8"
LOGS_DIR="${SCRATCH_STAGE}/v3_slurm_logs_r8"
ART_DIR="/home/x-mpourbaghima/.gemini/antigravity/brain/fbdbf148-c361-462e-87f3-14ddf9a1b5db"

export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0
export KMP_BLOCKTIME=0
export KMP_AFFINITY=disabled
export OMP_PROC_BIND=false
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR="${SCRATCH_STAGE}/tmp"
mkdir -p "${TMPDIR}" "${LOGS_DIR}"

cd "${ROOT}"

STAGE_CMD="${1:-submit_r3_fast}"

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
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    mkdir -p "${SHARDS_DIR}" "${OOS_CKPT_DIR}" "${LOCKS_DIR}" "${LOGS_DIR}"
    "${PY}" -c '
from pathlib import Path
from ferroflux.data.dataset import FerroChemSARPretrainDataset
p = Path("/anvil/projects/x-cis261485/Ferroptosis_ML/data/processed")
ds_ac = FerroChemSARPretrainDataset(data_dir=p, license_track="academic")
ds_op = FerroChemSARPretrainDataset(data_dir=p, license_track="open")
print(f"[PREWARM-OK] Cached Stage-0.5 SAR molecules: academic={len(ds_ac):,}, open={len(ds_op):,}.", flush=True)
'
    ;;

  # ---------------------------------------------------------------------------
  # 1-Seed End-to-End Empirical Pilot (12 OOD shards + 9 ablations + Stage-0)
  # ---------------------------------------------------------------------------
  pilot_hm_1)
    export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r3"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}"
    PIDS=()
    for snum in 1 2 3 4 5 6; do
      idx=$((snum - 1))
      sp="${SPLITS[$idx]}"
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track open \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 \
        --epochs 3 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 20 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_open_s${snum}" \
        > "${LOGS_DIR}/pilot_open_s${snum}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do if ! wait "${pid}"; then FAIL=1; fi; done
    exit "${FAIL}"
    ;;

  pilot_hm_2)
    export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r3"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}"
    PIDS=()
    for snum in 1 2 3 4 5 6; do
      idx=$((snum - 1))
      sp="${SPLITS[$idx]}"
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-split "${sp}" \
        --skip-ablations \
        --seeds 101 \
        --epochs 3 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 20 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_acad_s${snum}" \
        > "${LOGS_DIR}/pilot_acad_s${snum}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do if ! wait "${pid}"; then FAIL=1; fi; done
    exit "${FAIL}"
    ;;

  pilot_hm_3)
    export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r3"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}"
    PIDS=()
    for aidx in 0 1 2 3 4; do
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-ablation-idx "${aidx}" \
        --seeds 101 \
        --epochs 3 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 24 \
        --max-test-per-split 1500 \
        --shard-out-prefix "${PILOT_DIR}/pilot_abl${aidx}" \
        > "${LOGS_DIR}/pilot_abl${aidx}.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do if ! wait "${pid}"; then FAIL=1; fi; done
    exit "${FAIL}"
    ;;

  pilot_hm_4)
    export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20
    PILOT_DIR="${SCRATCH_STAGE}/v3_pilot_r3"
    PILOT_CKPT="${SCRATCH_STAGE}/v3_pilot_ckpts_r3"
    mkdir -p "${PILOT_DIR}" "${LOGS_DIR}" "${PILOT_CKPT}/v3_open" "${PILOT_CKPT}/v3_academic"
    PIDS=()
    for aidx in 5 6 7 8; do
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-ablation-idx "${aidx}" \
        --seeds 101 \
        --epochs 3 \
        --sar-epochs 8 \
        --max-samples-per-shard 200000 \
        --max-threads 20 \
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
        --max-steps-per-stage 85 \
        --batch-size 256 \
        > "${LOGS_DIR}/pilot_ckpt_${trk}_101.log" 2>&1 &
      PIDS+=("$!")
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do if ! wait "${pid}"; then FAIL=1; fi; done
    exit "${FAIL}"
    ;;

  pilot_summary)
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    "${PY}" - <<'PYEOF'
import glob, json
from pathlib import Path
from ferroflux.security.quarantine import freeze_checkpoint_sha256
from ferroflux.validation.stage0_bradbury_blind_eval import run_stage0_blind_evaluation

ood_all = []
for p in sorted(glob.glob("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_r3/*_ood.json")):
    ood_all.extend(json.loads(Path(p).read_text()))
ood_all.sort(key=lambda r: (0 if r["license_track"] == "open" else 1, r["ood_split"]))
for r in ood_all:
    print(f"PILOT OOD [{r['license_track']:8s}] {r['ood_split']:38s}: FF={r['spearman_rho']:.4f} (wg={r['within_group_spearman_rho']:.4f}) vs {r['best_baseline_model']}: {r['best_baseline_spearman']:.4f} [delta={r['delta_spearman_vs_best_baseline']:+.4f}]", flush=True)

abl_rows = []
for p in sorted(glob.glob("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_r3/*_abl.json")):
    abl_rows.extend(json.loads(Path(p).read_text()))
abl_rows.sort(key=lambda x: x.get("_ablation_idx", 99))
full_r = abl_rows[0]["spearman_rho"] if abl_rows else 0.0
for r in abl_rows:
    print(f"PILOT ABL {r['ablation_variant']:28s}: rho={r['spearman_rho']:.4f} [delta_vs_full={r['spearman_rho'] - full_r:+.4f}] rescue={r['rta_dfo_rescue_rate']} sec46_drop={r['sec46_depletion_drop']}", flush=True)

p_ckpt = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_ckpts_r3")
p_man = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_manifest_r3.json")
p_man.unlink(missing_ok=True)
for pt in sorted(p_ckpt.rglob("*.pt")):
    freeze_checkpoint_sha256(pt, p_man)
res_s0 = run_stage0_blind_evaluation(
    checkpoint_dir=p_ckpt,
    manifest_path=p_man,
    license_tracks=["open", "academic"],
    output_tables_dir=Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/v3_pilot_s0_r3"),
)
df_s0 = res_s0[0] if isinstance(res_s0, tuple) else res_s0
for r in df_s0.to_dict(orient="records"):
    print(f"PILOT S0  [{r['license_track']:8s}] {r['stage0_cohort']:52s}: rho={r['spearman_rho']:.4f} auc={r['auroc']} p={r['permutation_p']}", flush=True)
PYEOF
    ;;

  # ---------------------------------------------------------------------------
  # Full 5-Seed Atomic Work-Queue Worker (OOD seeds + Ablation seeds + Ckpts)
  # ---------------------------------------------------------------------------
  single_task_loop)
    THR="${2:-16}"
    WID="${3:-0}"
    REV_MODE="${4:-forward}"
    export OMP_NUM_THREADS="${THR}" MKL_NUM_THREADS="${THR}"
    mkdir -p "${SHARDS_DIR}" "${OOS_CKPT_DIR}" "${LOCKS_DIR}" "${LOGS_DIR}" \
      "${ROOT}/checkpoints/v3_clean/v3_open" "${ROOT}/checkpoints/v3_clean/v3_academic"

    run_ckpt_queue() {
      local CKPT_TASKS=()
      for sd in 101 202 303 404 505; do
        for trk in open academic; do
          CKPT_TASKS+=("${trk}:${sd}")
        done
      done
      if [[ "${REV_MODE}" == "reverse" ]]; then
        local REV_CK=()
        for (( idx_c=${#CKPT_TASKS[@]}-1 ; idx_c>=0 ; idx_c-- )) ; do
          REV_CK+=("${CKPT_TASKS[$idx_c]}")
        done
        CKPT_TASKS=("${REV_CK[@]}")
      fi
      for item in "${CKPT_TASKS[@]}"; do
        IFS=':' read -r trk sd <<< "${item}"
        local OUT_PT="${ROOT}/checkpoints/v3_clean/v3_${trk}/ferroflux_v3_${trk}_seed${sd}.pt"
        local DONE_MARK="${LOCKS_DIR}/done_ckpt_${trk}_${sd}"
        local LOCK_D="${LOCKS_DIR}/lock_ckpt_${trk}_${sd}"
        if [[ -f "${DONE_MARK}" ]] && [[ -f "${OUT_PT}" ]]; then continue; fi
        if ! mkdir "${LOCK_D}" 2>/dev/null; then continue; fi
        echo "[$(date -Is)] [W${WID}] Training Deployment Checkpoint track=${trk} seed=${sd}..."
        "${PY}" -m ferroflux.training.train_curriculum \
          --data-dir "${ROOT}/data/processed" \
          --output-dir "${ROOT}/checkpoints/v3_clean/v3_${trk}" \
          --license-track "${trk}" \
          --model-size v3 \
          --precision fp32 \
          --device auto \
          --seeds "${sd}" \
          --epochs-per-stage 3 3 4 3 \
          --max-steps-per-stage 85 \
          --batch-size 256 \
          > "${LOGS_DIR}/train_ckpt_${trk}_${sd}.log" 2>&1
        touch "${DONE_MARK}"
      done
    }

    if [[ "${REV_MODE}" == "reverse" ]]; then
      run_ckpt_queue
    fi

    # Persistent in-RAM Python worker for all 105 OOD & Ablation seed models (0s dataset reload across tasks!)
    "${PY}" - "${THR}" "${WID}" "${REV_MODE}" "${OOS_CKPT_DIR}" "${LOCKS_DIR}" <<'PYEOF'
import gc, os, sys, time, torch
from pathlib import Path
from ferroflux.data.dataset import FerroFluxMultimodalDataset, get_fast_indexer
from ferroflux.training.evaluate_ood_and_ablations import (
    ABLATION_9_SPECS,
    CANONICAL_6_OOD_SPLITS,
    _build_six_ood_splits,
    _extract_forbidden_test_chemistry,
    _train_strict_oos_model,
)

thr = int(sys.argv[1])
wid = sys.argv[2]
rev_mode = sys.argv[3]
oos_dir = Path(sys.argv[4])
locks_dir = Path(sys.argv[5])
root_dir = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
data_dir = root_dir / "data" / "processed"
cfg_path = root_dir / "configs" / "train_config.yaml"

torch.set_num_threads(thr)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

tasks = []
# Tier 1: Split 1 (open & academic) + Ablations 0 & 6 (20 tasks -> Wave 1 ~12 min, grouped by track!)
for sd in (101, 202, 303, 404, 505):
    tasks.append(("ood", "open", 1, sd))
for sd in (101, 202, 303, 404, 505):
    tasks.append(("ood", "academic", 1, sd))
    tasks.append(("abl", "academic", 0, sd))
    tasks.append(("abl", "academic", 6, sd))
# Tier 2: Remaining 7 retrained ablations on academic (35 tasks -> Wave 2, 0 dataset reloads!)
for sd in (101, 202, 303, 404, 505):
    for aidx in (1, 8, 2, 5, 7, 4, 3):
        tasks.append(("abl", "academic", aidx, sd))
# Tier 3: Remaining OOD Splits 2-6 grouped by track (25 academic then 25 open -> at most 1 reload!)
for trk in ("academic", "open"):
    for sd in (101, 202, 303, 404, 505):
        for snum in (2, 3, 4, 5, 6):
            tasks.append(("ood", trk, snum, sd))

if rev_mode == "reverse":
    tasks = list(reversed(tasks))

ds_cache = {}

def get_ds_and_splits(trk: str):
    if trk not in ds_cache:
        if len(ds_cache) >= 1:
            ds_cache.clear()
            gc.collect()
        t0 = time.perf_counter()
        ds = FerroFluxMultimodalDataset(data_dir=data_dir, license_track=trk, max_samples_per_shard=200000)
        _ = get_fast_indexer(ds)
        sp = _build_six_ood_splits(ds)
        ds_cache[trk] = (ds, sp)
        print(f"[{time.strftime('%H:%M:%S')}] [W{wid}] Cached dataset+indexer track={trk} ({len(ds):,} rows) in {time.perf_counter()-t0:.1f}s", flush=True)
    return ds_cache[trk]

for kind, trk, idx_num, sd in tasks:
    if kind == "ood":
        s_idx = idx_num - 1
        canon_split = CANONICAL_6_OOD_SPLITS[s_idx][0]
        eff_sd = sd + s_idx * 10
        out_pt = oos_dir / f"oos_{trk}_{canon_split}_{eff_sd}.pt"
        lock_d = locks_dir / f"lock_ood_{trk}_s{idx_num}_{sd}"
        if out_pt.is_file():
            continue
        if lock_d.exists() and (time.time() - lock_d.stat().st_mtime) > 7200.0:
            try:
                lock_d.rmdir()
            except OSError:
                pass
        try:
            lock_d.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        ds, sp = get_ds_and_splits(trk)
        tr_idx, te_idx_full = sp[canon_split]
        forbid_te = set(te_idx_full)
        f_smi, f_ik14, f_scaf = _extract_forbidden_test_chemistry(ds, te_idx_full, canon_split)
        t_kind = "chronos_fitness" if canon_split in ("3_2d_to_3d_organoid_pdx", "4_dual_crispr_fin_epistasis") else "viability"
        print(f"[{time.strftime('%H:%M:%S')}] [W{wid}] Training OOD track={trk} split={canon_split} seed={sd}...", flush=True)
        t_s = time.perf_counter()
        m_tr = _train_strict_oos_model(
            dataset=ds,
            train_indices=tr_idx,
            forbid_test_set=forbid_te,
            collate_fn=ds.collate_fn,
            n_genes=ds.n_genes,
            model_size="v3",
            config_path=cfg_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            data_dir=data_dir,
            license_track=trk,
            split_name=canon_split,
            target_kind=t_kind,
            forbid_smiles=f_smi,
            forbid_ik14=f_ik14,
            forbid_scaffolds=f_scaf,
            epochs=3,
            sar_epochs=8,
            seed=eff_sd,
            smoke_test=False,
        )
        tmp_p = out_pt.with_name(f".tmp_{os.getpid()}_{out_pt.name}")
        torch.save({"state_dict": m_tr.cpu().state_dict(), "train_sec": round(time.perf_counter() - t_s, 2)}, tmp_p)
        tmp_p.replace(out_pt)
        del m_tr
        gc.collect()
        print(f"[{time.strftime('%H:%M:%S')}] [W{wid}] Saved {out_pt.name} ({time.perf_counter()-t_s:.1f}s)", flush=True)
    elif kind == "abl":
        aidx = idx_num
        out_pt = oos_dir / f"oos_abl{aidx}_{trk}_{sd}.pt"
        lock_d = locks_dir / f"lock_abl_{aidx}_{sd}"
        if out_pt.is_file():
            continue
        if lock_d.exists() and (time.time() - lock_d.stat().st_mtime) > 7200.0:
            try:
                lock_d.rmdir()
            except OSError:
                pass
        try:
            lock_d.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        ds, sp = get_ds_and_splits(trk)
        abl_split = "2_leave_out_lineage_ccrcc_melanoma"
        ref_tr, ref_te = sp[abl_split]
        f_smi, f_ik14, f_scaf = _extract_forbidden_test_chemistry(ds, ref_te, abl_split)
        aname, aflags = ABLATION_9_SPECS[aidx]
        print(f"[{time.strftime('%H:%M:%S')}] [W{wid}] Training Ablation idx={aidx} ({aname}) seed={sd}...", flush=True)
        t_s = time.perf_counter()
        m_abl = _train_strict_oos_model(
            dataset=ds,
            train_indices=ref_tr,
            forbid_test_set=set(ref_te),
            collate_fn=ds.collate_fn,
            n_genes=ds.n_genes,
            model_size="v3",
            config_path=cfg_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            data_dir=data_dir,
            license_track=trk,
            split_name=abl_split,
            target_kind="viability",
            forbid_smiles=f_smi,
            forbid_ik14=f_ik14,
            forbid_scaffolds=f_scaf,
            epochs=3,
            sar_epochs=8,
            ablation_kw=aflags,
            seed=sd,
            smoke_test=False,
        )
        tmp_p = out_pt.with_name(f".tmp_{os.getpid()}_{out_pt.name}")
        torch.save({"state_dict": m_abl.cpu().state_dict(), "train_sec": round(time.perf_counter() - t_s, 2)}, tmp_p)
        tmp_p.replace(out_pt)
        del m_abl
        gc.collect()
        print(f"[{time.strftime('%H:%M:%S')}] [W{wid}] Saved {out_pt.name} ({time.perf_counter()-t_s:.1f}s)", flush=True)
PYEOF

    if [[ "${REV_MODE}" != "reverse" ]]; then
      run_ckpt_queue
    fi
    echo "[$(date -Is)] [W${WID}] Queue drained!"
    ;;

  queue_node)
    N_WORKERS="${2:-8}"
    THR_PER_W="${3:-16}"
    TAG_NODE="${4:-node}"
    DIR_MODE="${5:-forward}"
    PIDS=()
    for w_i in $(seq 1 "${N_WORKERS}"); do
      C_START=$(( (w_i - 1) * THR_PER_W ))
      C_END=$(( w_i * THR_PER_W - 1 ))
      if command -v taskset >/dev/null 2>&1 && [[ "${C_END}" -lt 128 ]]; then
        taskset -c "${C_START}-${C_END}" bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" single_task_loop "${THR_PER_W}" "${TAG_NODE}_${w_i}" "${DIR_MODE}" &
      else
        bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" single_task_loop "${THR_PER_W}" "${TAG_NODE}_${w_i}" "${DIR_MODE}" &
      fi
      PIDS+=("$!")
      sleep 3
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    exit "${FAIL}"
    ;;

  submit_r4_prod)
    mkdir -p "${SHARDS_DIR}" "${OOS_CKPT_DIR}" "${LOCKS_DIR}" "${LOGS_DIR}"
    bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" prewarm
    # 2 immediate 128-core highmem jobs (10 workers x 12 threads each = 20 fast workers on 750G RAM, 0 OOM!)
    for h_i in 1 2; do
      sbatch -A cis261485 -p highmem -N 1 -n 128 --mem=750G -t 01:50:00 \
        -J "ff7_hm${h_i}" -o "${LOGS_DIR}/slurm_hm${h_i}.out" \
        --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh queue_node 10 12 hm${h_i} forward"
    done
    # 2 wholenode 128-core jobs (7 workers x 18 threads each = 14 workers on 200G RAM)
    for w_i in 1 2; do
      sbatch -A cis261485 -p wholenode -N 1 -n 128 --mem=200G -t 01:50:00 \
        -J "ff7_wn${w_i}" -o "${LOGS_DIR}/slurm_wn${w_i}.out" \
        --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh queue_node 7 18 wn${w_i} forward"
    done
    # 12 shared 16-core backfill workers pulling deployment checkpoints + reverse end of the atomic queue
    for s_i in $(seq 1 12); do
      sbatch -A cis261485 -p shared -N 1 -n 16 --mem=28G -t 01:50:00 \
        -J "ff7_sh${s_i}" -o "${LOGS_DIR}/slurm_sh${s_i}.out" \
        --wrap="bash ${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh single_task_loop 16 sh${s_i} reverse"
    done
    ;;

  eval_all_shards)
    export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
    mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"
    PIDS=()
    # Evaluate all 12 5-seed OOD shards + 6 baselines concurrently on 128-core node
    for trk in open academic; do
      for snum in 1 2 3 4 5 6; do
        idx=$((snum - 1))
        sp="${SPLITS[$idx]}"
        "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
          --model-size v3 \
          --only-track "${trk}" \
          --only-split "${sp}" \
          --skip-ablations \
          --seeds 101 202 303 404 505 \
          --max-samples-per-shard 200000 \
          --max-threads 10 \
          --max-test-per-split 1500 \
          --load-oos-ckpt-dir "${OOS_CKPT_DIR}" \
          --shard-out-prefix "${SHARDS_DIR}/${trk}_s${snum}" \
          > "${LOGS_DIR}/eval_ood_${trk}_s${snum}.log" 2>&1 &
        PIDS+=("$!")
      done
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
      echo "[ERROR] OOD shard evaluation failed!" >&2
      exit 1
    fi

    # Evaluate all 9 5-seed retrained ablations concurrently
    PIDS=()
    for aidx in 0 1 2 3 4 5 6 7 8; do
      "${PY}" -m ferroflux.training.evaluate_ood_and_ablations \
        --model-size v3 \
        --only-track academic \
        --only-ablation-idx "${aidx}" \
        --seeds 101 202 303 404 505 \
        --max-samples-per-shard 200000 \
        --max-threads 14 \
        --max-test-per-split 1500 \
        --load-oos-ckpt-dir "${OOS_CKPT_DIR}" \
        --shard-out-prefix "${SHARDS_DIR}/acad_abl${aidx}" \
        > "${LOGS_DIR}/eval_abl_${aidx}.log" 2>&1 &
      PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then FAIL=1; fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
      echo "[ERROR] Ablation shard evaluation failed!" >&2
      exit 1
    fi
    ;;

  finalize)
    export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
    echo "[$(date -Is)] 0. Evaluating all 5-seed OOD & Ablation ensembles from strict-OOS checkpoints..."
    bash "${ROOT}/scripts/run_v3_slurm_gpu_or_highmem.sh" eval_all_shards

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

  *)
    echo "Unknown subcommand: ${STAGE_CMD}" >&2
    exit 2
    ;;
esac
