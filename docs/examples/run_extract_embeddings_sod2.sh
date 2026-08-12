#!/usr/bin/env bash
# Extract embeddings: NEW masking_split epoch=49 ckpt, held-out TEST only.
#   ckpt  (read):  .../T_perturb/res/masking_split/checkpoints/...epoch=49.ckpt
#   output (write): /mnt/sod2-project/.../T_perturb/res/masking_split/embeddings/
# Usage:
#   screen -S gene_emb_test
#   bash /home/stuke1/perturbgen/Perturbgen/docs/examples/run_extract_embeddings_sod2.sh

set -euo pipefail

source /home/stuke1/perturbgen/.venv/bin/activate
cd /home/stuke1/perturbgen

unset DISPLAY WAYLAND_DISPLAY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=offline
export WANDB_DIR=/mnt/sod2-project/csb4/stuke1/perturbgen/wandb
export TMPDIR=/mnt/sod2-project/csb4/stuke1/perturbgen/tmp
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache
export HWLOC_COMPONENTS=-gl HWLOC_GL_LINUX_NVIDIA_DISABLE=1
export PYTHONUNBUFFERED=1

CKPT=/home/stuke1/perturbgen/T_perturb/res/masking_split/checkpoints/20260808_2325_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_0-epoch=49.ckpt
SPLIT=/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/splits/stratified_cell_type_harmonized_seed42_80_10_10.pkl
OUTPUT_DIR=/mnt/sod2-project/csb4/stuke1/perturbgen/T_perturb/res/masking_split/embeddings
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$TMPDIR"
cat > "$OUTPUT_DIR/ckpt_info.txt" <<EOF
ckpt=/home/stuke1/perturbgen/T_perturb/res/masking_split/checkpoints/20260808_2325_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_0-epoch=49.ckpt
split_path=/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/splits/stratified_cell_type_harmonized_seed42_80_10_10.pkl
split=True (held-out test only)
context_mode=True
return_embeddings=True
return_gene_embs=True
pred_tps=1 2 3
EOF

echo "CKPT=$CKPT"
echo "SPLIT=$SPLIT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m perturbgen extract-embedding \
  --test_mode masking \
  --split True \
  --split_path "$SPLIT" \
  --context_mode True \
  --return_embeddings True \
  --return_gene_embs True \
  --gene_embs_condition time_after_LPS \
  --return_attn False \
  --generate False \
  --ckpt_masking_path "$CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --src_dataset /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/dataset_2000_hvg_src/normal.dataset \
  --tgt_dataset_folder /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/dataset_2000_hvg_tgt \
  --src_adata /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/h5ad_pairing_2000_hvg_src/normal.h5ad \
  --tgt_adata_folder /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/h5ad_pairing_2000_hvg_tgt \
  --mapping_dict_path /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/token_id_to_genename_2000_hvg.pkl \
  --tokenid_to_rowid /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/tokenid_to_rowid_2000_hvg.pkl \
  --batch_size 32 \
  --n_workers 0 \
  --pred_tps 1 2 3 \
  --var_list cell_pairing_index time_after_LPS cell_type_harmonized \
  --encoder scmaskgit \
  --encoder_path /home/stuke1/perturbgen/Perturbgen/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt \
  --mask_scheduler pow \
  --pos_encoding_mode time_pos_sin \
  --d_model 768 \
  --d_ff 32 \
  --num_layers 6 \
  --wandb_mode offline

echo "Done. Outputs under: $OUTPUT_DIR"
