#!/usr/bin/env bash
# Extract gene embeddings; write large outputs to sod2 (home disk is full).
# Usage:
#   screen -S gene_emb
#   bash /home/stuke1/perturbgen/Perturbgen/docs/examples/run_extract_embeddings_sod2.sh
#   # Ctrl+A D to detach

set -euo pipefail

source /home/stuke1/perturbgen/.venv/bin/activate
cd /home/stuke1/perturbgen

export CUDA_VISIBLE_DEVICES=5,6,7
export WANDB_MODE=offline
export WANDB_DIR=/mnt/sod2-project/csb4/stuke1/perturbgen/wandb
export TMPDIR=/mnt/sod2-project/csb4/stuke1/perturbgen/tmp

OUTPUT_DIR=/mnt/sod2-project/csb4/stuke1/perturbgen/T_perturb/res/masking
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$TMPDIR"

echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m perturbgen extract-embedding \
  --test_mode masking \
  --split False \
  --splitting_mode stratified \
  --return_embed True \
  --return_attn False \
  --generate False \
  --ckpt_masking_path /home/stuke1/perturbgen/T_perturb/res/masking/checkpoints/20260729_1751_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_0-epoch=19.ckpt \
  --output_dir "$OUTPUT_DIR" \
  --src_dataset /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/dataset_2000_hvg_src/normal.dataset \
  --tgt_dataset_folder /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/dataset_2000_hvg_tgt \
  --src_adata /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/h5ad_pairing_2000_hvg_src/normal.h5ad \
  --tgt_adata_folder /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/h5ad_pairing_2000_hvg_tgt \
  --mapping_dict_path /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/token_id_to_genename_2000_hvg.pkl \
  --batch_size 64 \
  --cellgen_lr 0.0001 \
  --cellgen_wd 0.0001 \
  --count_lr 0.001 \
  --count_wd 0.001 \
  --d_ff 32 \
  --num_layers 6 \
  --n_workers 4 \
  --pred_tps 1 2 3 \
  --var_list cell_pairing_index time_after_LPS cell_type_harmonized \
  --tokenid_to_rowid /home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/tokenid_to_rowid_2000_hvg.pkl \
  --encoder scmaskgit \
  --encoder_path /home/stuke1/perturbgen/Perturbgen/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt \
  --context_mode True \
  --mask_scheduler pow \
  --return_gene_embs True \
  --gene_embs_condition time_after_LPS \
  --pos_encoding_mode time_pos_sin \
  --d_model 768 \
  --wandb_mode offline

echo "Done. Outputs under: $OUTPUT_DIR"
