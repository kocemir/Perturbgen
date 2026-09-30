#!/usr/bin/env bash
# Gene-Query JEPA — one full-LPS run (encL=3, predL=3 only; not a layer grid).
# Frozen 90/10 split: train + val. Last 2 GPUs (DDP). Do not launch unless asked.
#
#   bash docs/examples/run_gene_query_full.sh
#   GPUS=6,7 EPOCHS=20 bash docs/examples/run_gene_query_full.sh
#
set -euo pipefail

WORKSPACE=/home/stuke1/perturbgen
REPO="${WORKSPACE}/Perturbgen"
TOKENIZED="${TOKENIZED:-/mnt/sod2-project/csb4/stuke1/perturbgen_reproduction/lps/tokenized_data/LPS_all_tps_2k}"
SPLIT_PATH="${SPLIT_PATH:-${TOKENIZED}/splits/stratified_cell_type_harmonized_seed42_90_10.pkl}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_stuke1}"
mkdir -p "${MPLCONFIGDIR}"
cd "${REPO}"
# shellcheck disable=SC1091
source "${WORKSPACE}/.venv/bin/activate"

last_two_gpus() {
  mapfile -t _idx < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
  [[ ${#_idx[@]} -eq 0 ]] && { echo "0,1"; return; }
  [[ ${#_idx[@]} -eq 1 ]] && { echo "${_idx[0]}"; return; }
  echo "${_idx[-2]},${_idx[-1]}"
}

GPUS="${GPUS:-$(last_two_gpus)}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
ENC_LAYERS="${ENC_LAYERS:-3}"
PRED_LAYERS="${PRED_LAYERS:-3}"
N_QUERIES="${N_QUERIES:-128}"
MAX_LEN="${MAX_LEN:-512}"
CELL_POOL="${CELL_POOL:-cls}"
LAMBDA_GENE="${LAMBDA_GENE:-1.0}"
LAMBDA_CELL="${LAMBDA_CELL:-0}"
LAMBDA_CONTR="${LAMBDA_CONTR:-0}"
VIC_VAR="${VIC_VAR:-0}"
VIC_COV="${VIC_COV:-0}"
ENCODER_LR_SCHEDULE="${ENCODER_LR_SCHEDULE:-freeze:5,5e-7:5,1e-6:5,freeze:5}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
EARLY_STOP="${EARLY_STOP:-false}"

echo "=== Gene-Query JEPA full LPS ==="
echo "gpus=${GPUS}  epochs=${EPOCHS}  encL=${ENC_LAYERS} predL=${PRED_LAYERS} Q=${N_QUERIES} max_len=${MAX_LEN}"
echo "pool=${CELL_POOL}  lg=${LAMBDA_GENE} lc=${LAMBDA_CELL}  vic=${VIC_VAR}/${VIC_COV}"
echo "schedule=${ENCODER_LR_SCHEDULE}"
echo "split=true  split_path=${SPLIT_PATH}"

exec python -u docs/examples/train_gene_query_jepa.py \
  --data full --split true --split-path "${SPLIT_PATH}" --tokenized "${TOKENIZED}" \
  --gpu "${GPUS}" \
  --freeze-encoder "${FREEZE_ENCODER}" \
  --encoder-lr-schedule "${ENCODER_LR_SCHEDULE}" \
  --cell-pool "${CELL_POOL}" \
  --lambda-gene "${LAMBDA_GENE}" --lambda-cell "${LAMBDA_CELL}" \
  --lambda-contrastive "${LAMBDA_CONTR}" \
  --vicreg-var "${VIC_VAR}" --vicreg-cov "${VIC_COV}" \
  --n-queries "${N_QUERIES}" \
  --max-len "${MAX_LEN}" \
  --encoder-layers "${ENC_LAYERS}" --predictor-layers "${PRED_LAYERS}" \
  --epochs "${EPOCHS}" --early-stop "${EARLY_STOP}" --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --strip-tgt-special-tokens false
