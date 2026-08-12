#!/usr/bin/env bash
# =============================================================================
# TOY systematic sweep for Gene-Query JEPA — 144 runs, all 8 GPUs
# =============================================================================
# Index: docs/examples/GENE_QUERY_JEPA.md
#
# Data: TOY roster (~480 cells: 2 batches x 16 per cell type), train=val=same
# cells. This is the overfit-on-purpose honesty experiment, NOT full LPS.
#
# Grid (2 x 2 x 2 x 3 x 6 = 144 runs):
#   freeze       ∈ {true, false}
#   VICReg       ∈ {off, on = var 1.0 / cov 0.04}
#   contrastive  ∈ {off, on = λ 0.3, τ 0.1}
#   Q (queries)  ∈ {64, 128, 256}
#   L (enc depth)∈ {1, 2, 3, 4, 5, 6}
#
# Fixed toy recipe: epochs 30, batch 16, lr 1e-4, predictor_layers 1,
# lambda_gene 1.0, lambda_cell 0.1, no early stop.
# Checkpoints: best epoch by val/gene_gap_vs_copy_src per run (weights only).
#
# Parallelism: one toy job per GPU, up to 8 at once, queue of 144.
#
# Output: /mnt/sod2-project/csb4/stuke1/perturbgen/gene_query_jepa/toy_runs/
#           systematic_144/<run_id>/{hparams.env, train.log, DONE,
#                                    toy_logs/version_0/metrics.csv,
#                                    checkpoints/*.ckpt}
#         + grid_manifest.tsv, status.tsv, suite_*.log at the suite root.
#
# Usage:
#   cd /home/stuke1/perturbgen/Perturbgen
#   source /home/stuke1/perturbgen/.venv/bin/activate
#   export PYTHONPATH=/home/stuke1/perturbgen/Perturbgen
#   bash docs/examples/run_gene_query_toy_sweep.sh
#
# Handy overrides:
#   DRY_RUN=1                        # list jobs, run nothing
#   SKIP_DONE=1                      # resume after interruption
#   Q_FILTER=64  L_FILTER=1,2        # subset the grid
#   FREEZE_FILTER=true  VIC_FILTER=0  CONTR_FILTER=1
#   ONLY=fzT_vic0_contr0_q64_L1      # explicit run ids (comma list)
#   MAX_RUNS=8                       # stop after N jobs
#   GPUS=0,1,2,3  MAX_PARALLEL=4     # use fewer GPUs
#   SAVE_CKPT=false                  # metrics only, no checkpoints
# =============================================================================
set -euo pipefail

WORKSPACE=/home/stuke1/perturbgen
REPO="${WORKSPACE}/Perturbgen"
SUITE_ROOT="${SUITE_ROOT:-/mnt/sod2-project/csb4/stuke1/perturbgen/gene_query_jepa/toy_runs/systematic_144}"

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO}"
# shellcheck disable=SC1091
source "${WORKSPACE}/.venv/bin/activate"

# ---- fixed toy recipe (the parameters we agreed on) ----
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCHES_PER_TYPE="${BATCHES_PER_TYPE:-2}"
LR="${LR:-1e-4}"
PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-1}"
EARLY_STOP="${EARLY_STOP:-false}"
CONTR_ON="${CONTR_ON:-0.3}"          # λ when the contrastive factor is on
CONTR_TAU="${CONTR_TAU:-0.1}"
VIC_VAR_ON="${VIC_VAR_ON:-1.0}"      # VICReg coeffs when on
VIC_COV_ON="${VIC_COV_ON:-0.04}"
SAVE_CKPT="${SAVE_CKPT:-true}"
CKPT_TOP_K="${CKPT_TOP_K:-1}"        # keep best epoch only (disk!)
CKPT_SAVE_LAST="${CKPT_SAVE_LAST:-false}"
CKPT_WEIGHTS_ONLY="${CKPT_WEIGHTS_ONLY:-true}"

# ---- parallelism ----
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_CSV}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_LIST[@]}}"

# ---- controls ----
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-0}"
ONLY="${ONLY:-}"
Q_FILTER="${Q_FILTER:-}"
L_FILTER="${L_FILTER:-}"
FREEZE_FILTER="${FREEZE_FILTER:-}"
VIC_FILTER="${VIC_FILTER:-}"
CONTR_FILTER="${CONTR_FILTER:-}"
MAX_RUNS="${MAX_RUNS:-0}"

mkdir -p "${SUITE_ROOT}"
MASTER_LOG="${SUITE_ROOT}/suite_$(date +%Y%m%d_%H%M%S).log"
STATUS_TSV="${SUITE_ROOT}/status.tsv"
MANIFEST="${SUITE_ROOT}/grid_manifest.tsv"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=== Gene-Query JEPA TOY sweep (144 alternatives) ==="
echo "TOY data (~480 cells) — not full LPS."
echo "suite_root=${SUITE_ROOT}"
echo "gpus=${GPUS_CSV}  max_parallel=${MAX_PARALLEL}  epochs=${EPOCHS}  batch=${BATCH_SIZE}"
echo

# ---- build the job list ----
echo -e "run_id\tfreeze\tvic_var/cov\tlambda_contr\tQ\tL\tout_dir" > "${MANIFEST}"
[[ -f "${STATUS_TSV}" ]] || echo -e "timestamp\trun_id\tstatus\tseconds\tgpu\tout_dir" > "${STATUS_TSV}"

JOBS=()
for FREEZE in true false; do
  [[ "${FREEZE}" == "true" ]] && FZ_TAG=fzT || FZ_TAG=fzF
  for VIC in 0 1; do
    if [[ "${VIC}" == "1" ]]; then
      VIC_TAG=vic1; VIC_VAR="${VIC_VAR_ON}"; VIC_COV="${VIC_COV_ON}"
    else
      VIC_TAG=vic0; VIC_VAR=0; VIC_COV=0
    fi
    for CONTR in 0 1; do
      if [[ "${CONTR}" == "1" ]]; then
        CONTR_TAG="contr${CONTR_ON}"; LAMBDA_CONTR="${CONTR_ON}"
      else
        CONTR_TAG=contr0; LAMBDA_CONTR=0
      fi
      for Q in 64 128 256; do
        for L in 1 2 3 4 5 6; do
          RUN_ID="${FZ_TAG}_${VIC_TAG}_${CONTR_TAG}_q${Q}_L${L}"
          OUT_DIR="${SUITE_ROOT}/${RUN_ID}"
          echo -e "${RUN_ID}\t${FREEZE}\t${VIC_VAR}/${VIC_COV}\t${LAMBDA_CONTR}\t${Q}\t${L}\t${OUT_DIR}" >> "${MANIFEST}"
          JOBS+=("${RUN_ID}|${FREEZE}|${VIC_VAR}|${VIC_COV}|${LAMBDA_CONTR}|${Q}|${L}|${OUT_DIR}|${VIC}|${CONTR}")
        done
      done
    done
  done
done
echo "Grid size: ${#JOBS[@]} (manifest: ${MANIFEST})"

# ---- filters ----
want() {
  local id="$1" out="$2" freeze="$3" vic="$4" contr="$5" q="$6" l="$7"
  [[ -n "${ONLY}"          && ",${ONLY},"     != *",${id},"*    ]] && return 1
  [[ -n "${Q_FILTER}"      && ",${Q_FILTER}," != *",${q},"*     ]] && return 1
  [[ -n "${L_FILTER}"      && ",${L_FILTER}," != *",${l},"*     ]] && return 1
  [[ -n "${FREEZE_FILTER}" && "${freeze}"     != "${FREEZE_FILTER}" ]] && return 1
  [[ -n "${VIC_FILTER}"    && "${vic}"        != "${VIC_FILTER}"    ]] && return 1
  [[ -n "${CONTR_FILTER}"  && "${contr}"      != "${CONTR_FILTER}"  ]] && return 1
  [[ "${SKIP_DONE}" == "1" && -f "${out}/DONE" ]] && return 1
  return 0
}

QUEUE=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r RUN_ID FREEZE VIC_VAR VIC_COV LAMBDA_CONTR Q L OUT_DIR VIC_F CONTR_F <<< "${spec}"
  want "${RUN_ID}" "${OUT_DIR}" "${FREEZE}" "${VIC_F}" "${CONTR_F}" "${Q}" "${L}" && QUEUE+=("${spec}")
done
if [[ "${MAX_RUNS}" != "0" && "${#QUEUE[@]}" -gt "${MAX_RUNS}" ]]; then
  QUEUE=("${QUEUE[@]:0:${MAX_RUNS}}")
fi
echo "Queued after filters: ${#QUEUE[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  for spec in "${QUEUE[@]}"; do
    IFS='|' read -r RUN_ID _rest <<< "${spec}"
    echo "[DRY_RUN] ${RUN_ID}"
  done
  echo "Dry run only — nothing launched."
  exit 0
fi

# ---- one job = one toy training on one GPU ----
run_one() {
  local RUN_ID="$1" FREEZE="$2" VIC_VAR="$3" VIC_COV="$4"
  local LAMBDA_CONTR="$5" Q="$6" L="$7" OUT_DIR="$8" GPU="$9"

  mkdir -p "${OUT_DIR}"
  cat > "${OUT_DIR}/hparams.env" <<EOF
DATASET=TOY_480_cells
RUN_ID=${RUN_ID}
FREEZE_ENCODER=${FREEZE}
VICREG_VAR=${VIC_VAR}
VICREG_COV=${VIC_COV}
LAMBDA_CONTRASTIVE=${LAMBDA_CONTR}
CONTRASTIVE_TAU=${CONTR_TAU}
N_QUERIES=${Q}
ENC_LAYERS=${L}
PREDICTOR_LAYERS=${PREDICTOR_LAYERS}
BATCH_SIZE=${BATCH_SIZE}
BATCHES_PER_TYPE=${BATCHES_PER_TYPE}
EPOCHS=${EPOCHS}
LR=${LR}
GPU=${GPU}
EOF

  local t0 rc
  t0=$(date +%s)
  echo ">> START gpu=${GPU} ${RUN_ID}"

  set +e
  TOY_GPU="${GPU}" \
  TOY_OUT="${OUT_DIR}" \
  FREEZE_ENCODER="${FREEZE}" \
  VICREG_VAR="${VIC_VAR}" \
  VICREG_COV="${VIC_COV}" \
  LAMBDA_CONTRASTIVE="${LAMBDA_CONTR}" \
  CONTRASTIVE_TAU="${CONTR_TAU}" \
  N_QUERIES="${Q}" \
  ENC_LAYERS="${L}" \
  PREDICTOR_LAYERS="${PREDICTOR_LAYERS}" \
  EPOCHS="${EPOCHS}" \
  EARLY_STOP="${EARLY_STOP}" \
  LR="${LR}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  BATCHES_PER_TYPE="${BATCHES_PER_TYPE}" \
  SAVE_CKPT="${SAVE_CKPT}" \
  CKPT_TOP_K="${CKPT_TOP_K}" \
  CKPT_SAVE_LAST="${CKPT_SAVE_LAST}" \
  CKPT_WEIGHTS_ONLY="${CKPT_WEIGHTS_ONLY}" \
  python -u docs/examples/toy_train_gene_query_jepa.py \
    > "${OUT_DIR}/train.log" 2>&1
  rc=$?
  set -e

  local elapsed=$(( $(date +%s) - t0 ))
  if [[ ${rc} -eq 0 ]]; then
    {
      echo "status=ok"
      echo "elapsed_sec=${elapsed}"
      echo "gpu=${GPU}"
    } > "${OUT_DIR}/DONE"
    echo -e "$(date -Iseconds)\t${RUN_ID}\tok\t${elapsed}\t${GPU}\t${OUT_DIR}" >> "${STATUS_TSV}"
    echo "<< OK   gpu=${GPU} ${RUN_ID} (${elapsed}s)"
  else
    echo -e "$(date -Iseconds)\t${RUN_ID}\tfail\t${elapsed}\t${GPU}\t${OUT_DIR}" >> "${STATUS_TSV}"
    echo "<< FAIL gpu=${GPU} ${RUN_ID} rc=${rc} (${elapsed}s) — see ${OUT_DIR}/train.log"
  fi
}

# ---- worker pool: one job per free GPU ----
declare -A PID_OF_GPU=()   # gpu -> pid

reap_finished() {
  local gpu pid
  for gpu in "${!PID_OF_GPU[@]}"; do
    pid="${PID_OF_GPU[${gpu}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      unset "PID_OF_GPU[${gpu}]"
    fi
  done
}

qi=0
total=${#QUEUE[@]}
while [[ ${qi} -lt ${total} || ${#PID_OF_GPU[@]} -gt 0 ]]; do
  # fill idle GPUs from the queue
  for gpu in "${GPU_LIST[@]}"; do
    [[ ${#PID_OF_GPU[@]} -ge ${MAX_PARALLEL} ]] && break
    [[ -n "${PID_OF_GPU[${gpu}]:-}" ]] && continue
    [[ ${qi} -ge ${total} ]] && break
    IFS='|' read -r RUN_ID FREEZE VIC_VAR VIC_COV LAMBDA_CONTR Q L OUT_DIR _ _ <<< "${QUEUE[${qi}]}"
    run_one "${RUN_ID}" "${FREEZE}" "${VIC_VAR}" "${VIC_COV}" \
      "${LAMBDA_CONTR}" "${Q}" "${L}" "${OUT_DIR}" "${gpu}" &
    PID_OF_GPU["${gpu}"]=$!
    qi=$((qi + 1))
  done
  # wait for at least one job to finish, then reap
  if [[ ${#PID_OF_GPU[@]} -gt 0 ]]; then
    wait -n 2>/dev/null || true
    reap_finished
  fi
done

echo
echo "=== TOY sweep finished: $(awk -F'\t' 'NR>1&&$3=="ok"' "${STATUS_TSV}" | wc -l) ok, $(awk -F'\t' 'NR>1&&$3=="fail"' "${STATUS_TSV}" | wc -l) fail ==="
echo "Summary:"
python docs/examples/summarize_gene_query_toy_sweep.py --suite-root "${SUITE_ROOT}" || true
