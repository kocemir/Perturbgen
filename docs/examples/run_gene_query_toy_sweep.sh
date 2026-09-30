#!/usr/bin/env bash
# =============================================================================
# Gene-Query JEPA — one toy sweep (Q × enc L × pred L)
# =============================================================================
# Trainer always logs present-query gene_mse (not in the objective).
# Toy split is batches × batch-size per class (default 4 train + 1 val, bs=16):
#   BATCH_SIZE=16 BATCHES_PER_TYPE=4 VAL_BATCHES_PER_TYPE=1
#   VAL_BATCHES_PER_TYPE=0 → val reuses train cells
#
# Default: 20 epochs, encoder freeze:5,5e-7:5,1e-6:5,freeze:5 (no 1e-5).
#   EPOCHS=20 CELL_POOL=cls LAMBDA_CELL=0.3 VIC_VAR=0.3
#
# Usage:
#   bash docs/examples/run_gene_query_toy_sweep.sh
#   DRY_RUN=1 SKIP_DONE=1 ONLY=q128_encL1_predL1 GPUS=6,7
# =============================================================================
set -euo pipefail

WORKSPACE=/home/stuke1/perturbgen
REPO="${WORKSPACE}/Perturbgen"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO}"
# shellcheck disable=SC1091
source "${WORKSPACE}/.venv/bin/activate"

csv_split() { IFS=',' read -r -a "$1" <<< "$2"; }
span_tag() {
  local -n _a=$1
  if [[ ${#_a[@]} -eq 1 ]]; then echo "${_a[0]}"; else echo "${_a[0]}-${_a[-1]}"; fi
}

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCHES_PER_TYPE="${BATCHES_PER_TYPE:-4}"
VAL_BATCHES_PER_TYPE="${VAL_BATCHES_PER_TYPE:-1}"
LR="${LR:-1e-4}"
FRAC_SHARED="${FRAC_SHARED:-0.4}"
FRAC_TGT_ONLY="${FRAC_TGT_ONLY:-0.5}"
ENCODER_LR_SCHEDULE="${ENCODER_LR_SCHEDULE:-freeze:5,5e-7:5,1e-6:5,freeze:5}"
EARLY_STOP="${EARLY_STOP:-false}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
LAMBDA_GENE="${LAMBDA_GENE:-1.0}"
LAMBDA_CELL="${LAMBDA_CELL:-0.3}"
CELL_POOL="${CELL_POOL:-cls}"
LAMBDA_CONTR="${LAMBDA_CONTR:-0}"
CONTR_TAU="${CONTR_TAU:-0.1}"
VIC_VAR="${VIC_VAR:-0.3}"
VIC_COV="${VIC_COV:-0.04}"
SAVE_CKPT="${SAVE_CKPT:-true}"
CKPT_TOP_K="${CKPT_TOP_K:-1}"
CKPT_SAVE_LAST="${CKPT_SAVE_LAST:-true}"
CKPT_WEIGHTS_ONLY="${CKPT_WEIGHTS_ONLY:-true}"

csv_split Q_ARR "${Q_LIST:-128}"
csv_split ENC_ARR "${ENC_LAYERS_LIST:-1,2,3}"
csv_split PRED_ARR "${PRED_LAYERS_LIST:-1,2,3}"
SPLIT_TAG="${BATCHES_PER_TYPE}p${VAL_BATCHES_PER_TYPE}"
SUITE_ROOT="${SUITE_ROOT:-/mnt/sod2-project/csb4/stuke1/perturbgen/gene_query_jepa/toy_runs/toy_q$(span_tag Q_ARR)_encL$(span_tag ENC_ARR)_predL$(span_tag PRED_ARR)_ep${EPOCHS}_${SPLIT_TAG}}"

last_two_gpus() {
  mapfile -t _idx < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
  [[ ${#_idx[@]} -eq 0 ]] && { echo "0,1"; return; }
  [[ ${#_idx[@]} -eq 1 ]] && { echo "${_idx[0]}"; return; }
  echo "${_idx[-2]},${_idx[-1]}"
}

GPUS_CSV="${GPUS:-$(last_two_gpus)}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_CSV}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_LIST[@]}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-0}"
ONLY="${ONLY:-}"
Q_FILTER="${Q_FILTER:-}"
ENC_FILTER="${ENC_FILTER:-}"
PRED_FILTER="${PRED_FILTER:-}"
MAX_RUNS="${MAX_RUNS:-0}"

mkdir -p "${SUITE_ROOT}"
MASTER_LOG="${SUITE_ROOT}/suite_$(date +%Y%m%d_%H%M%S).log"
STATUS_TSV="${SUITE_ROOT}/status.tsv"
MANIFEST="${SUITE_ROOT}/grid_manifest.tsv"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=== Gene-Query JEPA toy sweep ==="
echo "suite_root=${SUITE_ROOT}"
echo "gpus=${GPUS_CSV}  epochs=${EPOCHS}  Q=${Q_LIST:-128} encL=${ENC_LAYERS_LIST:-1,2,3} predL=${PRED_LAYERS_LIST:-1,2,3}"
echo "schedule=${ENCODER_LR_SCHEDULE}  shared=${FRAC_SHARED} tgt-only=${FRAC_TGT_ONLY}"
echo "pool=${CELL_POOL}  lg=${LAMBDA_GENE} lc=${LAMBDA_CELL}  vic=${VIC_VAR}/${VIC_COV}"
echo "toy: ${BATCHES_PER_TYPE} train + ${VAL_BATCHES_PER_TYPE} val batches/class  bs=${BATCH_SIZE}"
echo

echo -e "run_id\tQ\tencL\tpredL\tout_dir" > "${MANIFEST}"
[[ -f "${STATUS_TSV}" ]] || echo -e "timestamp\trun_id\tstatus\tseconds\tgpu\tout_dir" > "${STATUS_TSV}"

JOBS=()
for Q in "${Q_ARR[@]}"; do
  for ENC_L in "${ENC_ARR[@]}"; do
    for PRED_L in "${PRED_ARR[@]}"; do
      RUN_ID="q${Q}_encL${ENC_L}_predL${PRED_L}"
      OUT_DIR="${SUITE_ROOT}/${RUN_ID}"
      echo -e "${RUN_ID}\t${Q}\t${ENC_L}\t${PRED_L}\t${OUT_DIR}" >> "${MANIFEST}"
      JOBS+=("${RUN_ID}|${Q}|${ENC_L}|${PRED_L}|${OUT_DIR}")
    done
  done
done
echo "Grid size: ${#JOBS[@]}"

want() {
  local id="$1" out="$2" q="$3" enc="$4" pred="$5"
  [[ -n "${ONLY}"        && ",${ONLY},"        != *",${id},"*   ]] && return 1
  [[ -n "${Q_FILTER}"    && ",${Q_FILTER},"    != *",${q},"*    ]] && return 1
  [[ -n "${ENC_FILTER}"  && ",${ENC_FILTER},"  != *",${enc},"*  ]] && return 1
  [[ -n "${PRED_FILTER}" && ",${PRED_FILTER}," != *",${pred},"* ]] && return 1
  [[ "${SKIP_DONE}" == "1" && -f "${out}/DONE" ]] && return 1
  return 0
}

QUEUE=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r RUN_ID Q ENC_L PRED_L OUT_DIR <<< "${spec}"
  want "${RUN_ID}" "${OUT_DIR}" "${Q}" "${ENC_L}" "${PRED_L}" && QUEUE+=("${spec}")
done
if [[ "${MAX_RUNS}" != "0" && "${#QUEUE[@]}" -gt "${MAX_RUNS}" ]]; then
  QUEUE=("${QUEUE[@]:0:${MAX_RUNS}}")
fi
echo "Queued: ${#QUEUE[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY_RUN] %s\n' "${QUEUE[@]%%|*}"
  echo "Dry run only."
  exit 0
fi

run_one() {
  local RUN_ID="$1" Q="$2" ENC_L="$3" PRED_L="$4" OUT_DIR="$5" GPU="$6"
  mkdir -p "${OUT_DIR}"
  cat > "${OUT_DIR}/hparams.env" <<EOF
DATA=toy
RUN_ID=${RUN_ID}
N_QUERIES=${Q}
ENC_LAYERS=${ENC_L}
PREDICTOR_LAYERS=${PRED_L}
FRAC_SHARED=${FRAC_SHARED}
FRAC_TGT_ONLY=${FRAC_TGT_ONLY}
ENCODER_LR_SCHEDULE=${ENCODER_LR_SCHEDULE}
LAMBDA_GENE=${LAMBDA_GENE}
LAMBDA_CELL=${LAMBDA_CELL}
CELL_POOL=${CELL_POOL}
VIC_VAR=${VIC_VAR}
VIC_COV=${VIC_COV}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
BATCHES_PER_TYPE=${BATCHES_PER_TYPE}
VAL_BATCHES_PER_TYPE=${VAL_BATCHES_PER_TYPE}
LR=${LR}
GPU=${GPU}
SUITE=${SUITE_ROOT}
EOF

  local t0 rc elapsed
  t0=$(date +%s)
  echo ">> START gpu=${GPU} ${RUN_ID}"
  set +e
  python -u docs/examples/train_gene_query_jepa.py \
    --data toy --gpu "${GPU}" --output-dir "${OUT_DIR}" \
    --freeze-encoder "${FREEZE_ENCODER}" \
    --encoder-lr-schedule "${ENCODER_LR_SCHEDULE}" \
    --vicreg-var "${VIC_VAR}" --vicreg-cov "${VIC_COV}" \
    --lambda-gene "${LAMBDA_GENE}" --lambda-cell "${LAMBDA_CELL}" \
    --cell-pool "${CELL_POOL}" \
    --lambda-contrastive "${LAMBDA_CONTR}" --contrastive-tau "${CONTR_TAU}" \
    --n-queries "${Q}" --encoder-layers "${ENC_L}" --predictor-layers "${PRED_L}" \
    --max-len "${MAX_LEN:-512}" \
    --frac-shared "${FRAC_SHARED}" --frac-tgt-only "${FRAC_TGT_ONLY}" \
    --epochs "${EPOCHS}" --early-stop "${EARLY_STOP}" --lr "${LR}" \
    --batch-size "${BATCH_SIZE}" \
    --batches-per-type "${BATCHES_PER_TYPE}" \
    --val-batches-per-type "${VAL_BATCHES_PER_TYPE}" \
    --save-ckpt "${SAVE_CKPT}" --ckpt-top-k "${CKPT_TOP_K}" \
    --ckpt-save-last "${CKPT_SAVE_LAST}" --ckpt-weights-only "${CKPT_WEIGHTS_ONLY}" \
    > "${OUT_DIR}/train.log" 2>&1
  rc=$?
  set -e
  elapsed=$(( $(date +%s) - t0 ))
  if [[ ${rc} -eq 0 ]]; then
    { echo "status=ok"; echo "elapsed_sec=${elapsed}"; echo "gpu=${GPU}"; } > "${OUT_DIR}/DONE"
    echo -e "$(date -Iseconds)\t${RUN_ID}\tok\t${elapsed}\t${GPU}\t${OUT_DIR}" >> "${STATUS_TSV}"
    echo "<< OK   gpu=${GPU} ${RUN_ID} (${elapsed}s)"
  else
    echo -e "$(date -Iseconds)\t${RUN_ID}\tfail\t${elapsed}\t${GPU}\t${OUT_DIR}" >> "${STATUS_TSV}"
    echo "<< FAIL gpu=${GPU} ${RUN_ID} rc=${rc} (${elapsed}s) — ${OUT_DIR}/train.log"
  fi
}

declare -A PID_OF_GPU=()
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
  for gpu in "${GPU_LIST[@]}"; do
    [[ ${#PID_OF_GPU[@]} -ge ${MAX_PARALLEL} ]] && break
    [[ -n "${PID_OF_GPU[${gpu}]:-}" ]] && continue
    [[ ${qi} -ge ${total} ]] && break
    IFS='|' read -r RUN_ID Q ENC_L PRED_L OUT_DIR <<< "${QUEUE[${qi}]}"
    run_one "${RUN_ID}" "${Q}" "${ENC_L}" "${PRED_L}" "${OUT_DIR}" "${gpu}" &
    PID_OF_GPU["${gpu}"]=$!
    qi=$((qi + 1))
  done
  if [[ ${#PID_OF_GPU[@]} -gt 0 ]]; then
    wait -n 2>/dev/null || true
    reap_finished
  fi
done

echo
echo "=== sweep finished: $(awk -F'\t' 'NR>1&&$3=="ok"' "${STATUS_TSV}" | wc -l) ok, $(awk -F'\t' 'NR>1&&$3=="fail"' "${STATUS_TSV}" | wc -l) fail ==="
python docs/examples/summarize_gene_query_toy_sweep.py --suite-root "${SUITE_ROOT}" || true
python docs/examples/plot_gene_query_toy_sweep.py --suite-root "${SUITE_ROOT}" --out-dir "${SUITE_ROOT}/plots" || true
