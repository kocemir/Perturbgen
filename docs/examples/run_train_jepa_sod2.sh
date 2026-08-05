#!/usr/bin/env bash
# Phase A full-data JEPA on LPS (sod2). Main launch script (2026-08+).
# Index: docs/examples/JEPA_README.md
#
# Encoder modes (env):
#   JEPA_ENCODER=scmaskgit  (default)  pretrained MaskGIT, early-exit N layers
#   JEPA_ENCODER=cell                  CellEncoder (random TF; optional token warm-start)
#
# Freeze (env): FREEZE_ENCODER=true|false
#   - scmaskgit: either OK (fz vs unfz ablation)
#   - cell: MUST be false (random weights need training)
#
# Token warm-start (cell only): WARMSTART_TOKEN_EMB=true|false
#   true  = copy MaskGIT gene embedding table; transformer still random
#   false = fully random (default when JEPA_ENCODER=cell)
# GPUs: 0,1,2,7
set -euo pipefail

WORKSPACE=/home/stuke1/perturbgen
REPO="${WORKSPACE}/Perturbgen"
TOKENIZED="${WORKSPACE}/T_perturb/tokenized_data/LPS_all_tps_2k"
SOD2_ROOT=/mnt/sod2-project/csb4/stuke1/perturbgen
OUTPUT_DIR="${SOD2_ROOT}/T_perturb/res/jepa"
PRETRAIN_CKPT="${REPO}/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,7}"
# Headless: avoid X11/wandb/matplotlib hangs ("No protocol specified").
unset DISPLAY || true
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SOD2_ROOT}/tmp/matplotlib"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_DIR="${SOD2_ROOT}/wandb"
export TMPDIR="${SOD2_ROOT}/tmp"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
# mpi4py is present; clear leftover MPI singleton env so PL won't attach orted.
unset OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK \
  OMPI_COMM_WORLD_LOCAL_SIZE OMPI_UNIVERSE_SIZE PMI_RANK PMI_SIZE PMIX_RANK \
  PMIX_NAMESPACE PMIX_SERVER_URI2 PMIX_SERVER_URI3 2>/dev/null || true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}" "${SOD2_ROOT}/logs"

cd "${WORKSPACE}"
# shellcheck disable=SC1091
source "${WORKSPACE}/.venv/bin/activate"

JEPA_ENCODER="${JEPA_ENCODER:-scmaskgit}"
# Early-exit depth into pretrained 12-layer body (scmaskgit only).
JEPA_ENCODER_LAYERS="${JEPA_ENCODER_LAYERS:-3}"
# CellEncoder capacity (matched roughly to 3L ablation; CLI defaults are weak).
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-8}"
D_FF="${D_FF:-3072}"
D_MODEL="${D_MODEL:-768}"

JEPA_LOSS="${JEPA_LOSS:-cosine}"
VICREG_VAR="${VICREG_VAR:-1.0}"
VICREG_COV="${VICREG_COV:-0.04}"
EPOCHS="${EPOCHS:-30}"
# Global batch; code divides by #GPUs → 256/4 = 64 per GPU.
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"

# Default freeze: scmaskgit often fz; cell must train.
if [[ -z "${FREEZE_ENCODER:-}" ]]; then
  if [[ "${JEPA_ENCODER}" == "cell" ]]; then
    FREEZE_ENCODER=false
  else
    FREEZE_ENCODER=false
  fi
fi

# Token-emb warm-start from MaskGIT (cell only). Off by default for pure random.
if [[ -z "${WARMSTART_TOKEN_EMB:-}" ]]; then
  if [[ "${JEPA_ENCODER}" == "cell" ]]; then
    WARMSTART_TOKEN_EMB=false
  else
    WARMSTART_TOKEN_EMB=true
  fi
fi

if [[ "${JEPA_ENCODER}" == "cell" && ( "${FREEZE_ENCODER}" == "true" || "${FREEZE_ENCODER}" == "1" ) ]]; then
  echo "ERROR: JEPA_ENCODER=cell requires FREEZE_ENCODER=false (random encoder must train)." >&2
  exit 1
fi

FZ_TAG="fz"; [[ "${FREEZE_ENCODER}" == "true" || "${FREEZE_ENCODER}" == "1" ]] || FZ_TAG="unfz"
if [[ "${JEPA_ENCODER}" == "cell" ]]; then
  ENC_TAG="cell_L${NUM_LAYERS}"
else
  ENC_TAG="scmaskgit${JEPA_ENCODER_LAYERS}L"
fi
LOG_FILE="${SOD2_ROOT}/logs/jepa_phaseA_${ENC_TAG}_${FZ_TAG}_bs${BATCH_SIZE}_${JEPA_LOSS}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG_FILE}"
echo "GPUs=${CUDA_VISIBLE_DEVICES} encoder=${JEPA_ENCODER} freeze=${FREEZE_ENCODER} warmstart_tok=${WARMSTART_TOKEN_EMB} loss=${JEPA_LOSS} batch=${BATCH_SIZE} d_model=${D_MODEL} layers=${NUM_LAYERS} d_ff=${D_FF}"

CMD=(
  python -m perturbgen train-jepa
  --train_mode jepa
  --split True
  --splitting_mode stratified
  --split_obs cell_type_harmonized
  --src_dataset "${TOKENIZED}/dataset_2000_hvg_src/normal.dataset"
  --tgt_dataset_folder "${TOKENIZED}/dataset_2000_hvg_tgt"
  --src_adata "${TOKENIZED}/h5ad_pairing_2000_hvg_src/normal.h5ad"
  --tgt_adata_folder "${TOKENIZED}/h5ad_pairing_2000_hvg_tgt"
  --mapping_dict_path "${TOKENIZED}/token_id_to_genename_2000_hvg.pkl"
  --tokenid_to_rowid_path "${TOKENIZED}/tokenid_to_rowid_2000_hvg.pkl"
  --jepa_encoder "${JEPA_ENCODER}"
  --jepa_encoder_layers "${JEPA_ENCODER_LAYERS}"
  --freeze_jepa_encoder "${FREEZE_ENCODER}"
  --num_layers "${NUM_LAYERS}"
  --num_heads "${NUM_HEADS}"
  --d_ff "${D_FF}"
  --d_model "${D_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${SOD2_ROOT}/logs"
  --pred_tps 1 2 3
  --var_list cell_type_harmonized time_after_LPS
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --cellgen_lr "${LR}"
  --cellgen_wd 1e-4
  --n_workers 4
  --ema_decay 0.996
  --normalize_latents true
  --jepa_loss "${JEPA_LOSS}"
  --vicreg_var_coeff "${VICREG_VAR}"
  --vicreg_cov_coeff "${VICREG_COV}"
  --pos_encoding_mode time_pos_sin
  --wandb_mode disabled
  --seed 0
  --num_node 1
)

if [[ "${JEPA_ENCODER}" == "scmaskgit" ]]; then
  CMD+=(--encoder_path "${PRETRAIN_CKPT}")
fi

# Warm-start token emb (optional for cell; unused for scmaskgit load path beyond CLI).
if [[ "${WARMSTART_TOKEN_EMB}" == "true" || "${WARMSTART_TOKEN_EMB}" == "1" ]]; then
  CMD+=(--ckpt_masking_path "${PRETRAIN_CKPT}")
  # scmaskgit also historically passed encoder_path as ckpt; keep both when warmstart on.
  if [[ "${JEPA_ENCODER}" == "scmaskgit" ]]; then
    :
  fi
elif [[ "${JEPA_ENCODER}" == "scmaskgit" ]]; then
  # scmaskgit still needs the ckpt for SCMaskGITCellEncoder via --encoder_path only.
  :
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
