#!/bin/bash
# One v2 run on the 8 GPUs of this node (data parallel, one process per GPU; effective batch 64 = 8 GPUs x micro-batch 1
# x accumulation 8). Micro-batch 1: a sample's forward pass must not depend on what it is batched with (docs/v2/training_audit.md).
#   launch/launch_run_v2.sh <configs/v2/X.yaml> <SEED> [MASTER_PORT] [extra train_v2.py args...]
# Output: $LOCUS_ROOT/work/runs/v2/<family>-s<seed>/, or $OUT if set (audit runs).
set -euo pipefail
source "$(dirname "$0")/env.sh"
CONFIG=$1; SEED=$2; PORT=${3:-29600}; shift $(( $# < 3 ? $# : 3 ))
FAM=$(basename "$CONFIG" .yaml)
OUT=${OUT:-$LOCUS_ROOT/work/runs/v2/$FAM-s$SEED}
unset CUDA_VISIBLE_DEVICES          # torchrun gives each rank its own GPU
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN
mkdir -p "$OUT" "$LOCUS_ROOT/work/logs"
cd "$LCAM_CODE"
exec "$PY" -m torch.distributed.run --nproc-per-node "${NPROC:-8}" --master-port "$PORT" \
  scripts/train_v2.py --config "$CONFIG" --seed "$SEED" --bundle "$BUNDLE" --out "$OUT" \
  --micro-batch "${MICRO_BATCH:-1}" --workers 4 --allow-optimizer-step "$@"
