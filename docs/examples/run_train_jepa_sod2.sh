#!/usr/bin/env bash
# Phase A: train cell-trajectory JEPA on LPS (sod2 outputs, GPUs 0,1).
set -euo pipefail

WORKSPACE=/home/stuke1/perturbgen
REPO="${WORKSPACE}/Perturbgen"
TOKENIZED="${WORKSPACE}/T_perturb/tokenized_data/LPS_all_tps_2k"
SOD2_ROOT=/mnt/sod2-project/csb4/stuke1/perturbgen
OUTPUT_DIR="${SOD2_ROOT}/T_perturb/res/jepa"
# Pretrained cohort masking ckpt (not LPS-fine-tuned). Warm-starts token_embedding only.
PRETRAIN_CKPT="${REPO}/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${SOD2_ROOT}/wandb"
export TMPDIR="${SOD2_ROOT}/tmp"
mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${TMPDIR}" "${SOD2_ROOT}/logs"

cd "${WORKSPACE}"
# shellcheck disable=SC1091
source "${WORKSPACE}/.venv/bin/activate"

# Default: pretrained MaskGIT source encoder (scmaskgit). For the lightweight
# CellEncoder alternative, set JEPA_ENCODER=cell (and keep ckpt warm-start).
JEPA_ENCODER="${JEPA_ENCODER:-scmaskgit}"

python -m perturbgen train-jepa \
  --train_mode jepa \
  --split True \
  --splitting_mode stratified \
  --split_obs cell_type_harmonized \
  --src_dataset "${TOKENIZED}/dataset_2000_hvg_src/normal.dataset" \
  --tgt_dataset_folder "${TOKENIZED}/dataset_2000_hvg_tgt" \
  --src_adata "${TOKENIZED}/h5ad_pairing_2000_hvg_src/normal.h5ad" \
  --tgt_adata_folder "${TOKENIZED}/h5ad_pairing_2000_hvg_tgt" \
  --mapping_dict_path "${TOKENIZED}/token_id_to_genename_2000_hvg.pkl" \
  --tokenid_to_rowid_path "${TOKENIZED}/tokenid_to_rowid_2000_hvg.pkl" \
  --encoder_path "${PRETRAIN_CKPT}" \
  --ckpt_masking_path "${PRETRAIN_CKPT}" \
  --jepa_encoder "${JEPA_ENCODER}" \
  --freeze_jepa_encoder false \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${SOD2_ROOT}/logs" \
  --pred_tps 1 2 3 \
  --var_list cell_type_harmonized time_after_LPS \
  --batch_size 64 \
  --epochs 20 \
  --cellgen_lr 1e-4 \
  --cellgen_wd 1e-4 \
  --n_workers 4 \
  --num_layers 2 \
  --d_ff 1024 \
  --d_model 768 \
  --ema_decay 0.996 \
  --normalize_latents true \
  --jepa_loss mse \
  --pos_encoding_mode time_pos_sin \
  --wandb_mode offline \
  --seed 0 \
  --num_node 1
