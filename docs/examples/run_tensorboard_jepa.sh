#!/usr/bin/env bash
# TensorBoard for JEPA Phase A kept runs (sod2). See JEPA_README.md.
# Default: A unfz 1120 | B fz 1128 | C cell warmstart 1622  → port 6007.
set -euo pipefail

SOD2_LOGS=/mnt/sod2-project/csb4/stuke1/perturbgen/logs
WORKSPACE=/home/stuke1/perturbgen
PORT="${PORT:-6007}"

# shellcheck disable=SC1091
source "${WORKSPACE}/.venv/bin/activate"

# Override with LOGDIR_SPEC='name:path,name2:path2' or single LOGDIR=...
if [[ -n "${LOGDIR:-}" ]]; then
  exec tensorboard --logdir "${LOGDIR}" --port "${PORT}" --bind_all --load_fast=false
fi

LOGDIR_SPEC="${LOGDIR_SPEC:-unfz_1120:${SOD2_LOGS}/20260804_1120_cellgen,fz_1128:${SOD2_LOGS}/20260805_1128_cellgen,cell_1622:${SOD2_LOGS}/20260805_1622_cellgen}"
echo "TensorBoard port=${PORT}"
echo "logdir_spec=${LOGDIR_SPEC}"
echo "Open http://localhost:${PORT} (or SSH tunnel -L ${PORT}:localhost:${PORT})"
exec tensorboard --logdir_spec "${LOGDIR_SPEC}" --port "${PORT}" --bind_all --load_fast=false
