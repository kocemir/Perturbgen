#!/usr/bin/env bash
# Phase A overfit diagnostic (2026-08-03) — debug only, not full-data.
# See docs/examples/JEPA_README.md
set -euo pipefail

WORKSPACE=/home/stuke1/perturbgen
REPO="${WORKSPACE}/Perturbgen"
SOD2=/mnt/sod2-project/csb4/stuke1/perturbgen
OUT="${SOD2}/T_perturb/res/jepa/phaseA_overfit_$(date +%Y%m%d_%H%M)"

# Default: 6 GPUs. Override with CUDA_VISIBLE_DEVICES=...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,5,6,7}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd "${WORKSPACE}"
if [[ -x "${WORKSPACE}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE}/.venv/bin/activate"
else
  # shellcheck disable=SC1091
  source /home/stuke1/.cache/pypoetry/virtualenvs/perturbgen-SeCBAeJg-py3.11/bin/activate
fi

mkdir -p "${OUT}"

# Count visible GPUs for torchrun
IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${#_gpus[@]}"

# Extra args after -- go to the python script, e.g. --jepa_loss cosine
torchrun --standalone --nproc_per_node="${NPROC}" \
  "${REPO}/docs/examples/jepa_phase_a_overfit.py" \
  --output_dir "${OUT}" \
  --batch_size 64 \
  --sample_mode per_class \
  --min_batches_per_class 2 \
  --val_batches_per_class 1 \
  --epochs 200 \
  --lr 1e-4 \
  --ema_decay 0.996 \
  --jepa_loss cosine \
  --pred_tps 1 2 3 \
  "$@"

echo "Results in ${OUT}"
