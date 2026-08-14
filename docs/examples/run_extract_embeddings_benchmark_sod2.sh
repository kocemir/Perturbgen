#!/usr/bin/env bash
# Benchmark embedding extraction (TEST split only).
#
# What this script does:
#   1) Uses one frozen split pickle from notebook 02.
#   2) Runs `perturbgen extract-embedding` once on test_indices.
#   3) Renames output files so they explicitly include benchmark_test naming.
#
# Usage:
#   screen -S emb_benchmark
#   bash /home/stuke1/perturbgen/Perturbgen/docs/examples/run_extract_embeddings_benchmark_sod2.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0
#   BATCH_SIZE=32
#   BENCHMARK_TAG=paper_v1
#   CKPT=/path/to/checkpoint.ckpt
#   SPLIT=/path/to/split.pkl
#   RENAME_OUTPUTS=true

set -euo pipefail

source /home/stuke1/perturbgen/.venv/bin/activate
cd /home/stuke1/perturbgen

unset DISPLAY WAYLAND_DISPLAY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=offline
export WANDB_DIR=/mnt/sod2-project/csb4/stuke1/perturbgen/wandb
export TMPDIR=/mnt/sod2-project/csb4/stuke1/perturbgen/tmp
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache
export HWLOC_COMPONENTS=-gl
export HWLOC_GL_LINUX_NVIDIA_DISABLE=1
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-/home/stuke1/perturbgen/T_perturb/res/masking_split/checkpoints/20260808_2325_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_0-epoch=49.ckpt}"
SPLIT="${SPLIT:-/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/splits/stratified_cell_type_harmonized_seed42_80_10_10.pkl}"
SRC_DATASET=/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/dataset_2000_hvg_src/normal.dataset
TGT_DATASET_FOLDER=/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/dataset_2000_hvg_tgt
SRC_ADATA=/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/h5ad_pairing_2000_hvg_src/normal.h5ad
TGT_ADATA_FOLDER=/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/h5ad_pairing_2000_hvg_tgt
MAPPING_DICT=/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/token_id_to_genename_2000_hvg.pkl
TOKENID_TO_ROWID=/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k/tokenid_to_rowid_2000_hvg.pkl
ENCODER_PATH=/home/stuke1/perturbgen/Perturbgen/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt

BATCH_SIZE="${BATCH_SIZE:-32}"
BENCHMARK_TAG="${BENCHMARK_TAG:-paper_v1}"
BENCHMARK_ROOT=/mnt/sod2-project/csb4/stuke1/perturbgen/T_perturb/res/masking_split
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_ROOT}/embeddings_benchmark_test_${BENCHMARK_TAG}}"
RENAME_OUTPUTS="${RENAME_OUTPUTS:-true}"

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$TMPDIR"

cat > "${OUTPUT_DIR}/benchmark_run_info.txt" <<EOF
tag=${BENCHMARK_TAG}
ckpt=${CKPT}
split_path=${SPLIT}
split=True
context_mode=True
return_embeddings=True
return_gene_embs=True
pred_tps=1 2 3
batch_size=${BATCH_SIZE}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
output_dir=${OUTPUT_DIR}
mode=test_only
rename_outputs=${RENAME_OUTPUTS}
EOF

echo "=== Benchmark embedding extraction (TEST only) ==="
echo "CKPT=$CKPT"
echo "SPLIT=$SPLIT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "BATCH_SIZE=$BATCH_SIZE"

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
  --src_dataset "$SRC_DATASET" \
  --tgt_dataset_folder "$TGT_DATASET_FOLDER" \
  --src_adata "$SRC_ADATA" \
  --tgt_adata_folder "$TGT_ADATA_FOLDER" \
  --mapping_dict_path "$MAPPING_DICT" \
  --tokenid_to_rowid "$TOKENID_TO_ROWID" \
  --batch_size "$BATCH_SIZE" \
  --n_workers 0 \
  --pred_tps 1 2 3 \
  --var_list cell_pairing_index time_after_LPS cell_type_harmonized \
  --encoder scmaskgit \
  --encoder_path "$ENCODER_PATH" \
  --mask_scheduler pow \
  --pos_encoding_mode time_pos_sin \
  --d_model 768 \
  --d_ff 32 \
  --num_layers 6 \
  --wandb_mode offline

if [[ "${RENAME_OUTPUTS,,}" == "true" || "${RENAME_OUTPUTS}" == "1" ]]; then
  python3 - "$OUTPUT_DIR" "$BENCHMARK_TAG" <<'PY'
import sys
from pathlib import Path
import shutil

out_dir = Path(sys.argv[1])
tag = sys.argv[2]
prefix = f"benchmark_test_{tag}"

patterns = [
    ("*_inference_embs_*.h5ad", f"{prefix}_inference_embs.h5ad"),
    ("*_gene_embs_*.csv", f"{prefix}_gene_embs.csv"),
    ("*_cls_embs_*.csv", f"{prefix}_cls_embs.csv"),
]

renamed = []
for glob_pat, new_name in patterns:
    candidates = sorted(out_dir.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    if not candidates:
        continue
    src = candidates[-1]
    dst = out_dir / new_name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    renamed.append((src.name, dst.name))

report_path = out_dir / f"{prefix}_files.txt"
with report_path.open("w", encoding="utf-8") as f:
    f.write("benchmark_test file mapping\n")
    for src_name, dst_name in renamed:
        f.write(f"{src_name} -> {dst_name}\n")

print(f"Wrote rename report: {report_path}")
for src_name, dst_name in renamed:
    print(f"{src_name} -> {dst_name}")
PY
fi

echo "Done. Benchmark test outputs under: $OUTPUT_DIR"
