#!/bin/bash
# One v5 GRPO run on the 8 GPUs of this node (one process per GPU), continuing an SFT adapter.
#   launch/launch_grpo.sh <configs/v5/X.yaml> <SEED> <INIT_ADAPTER> [MASTER_PORT] [extra train_grpo.py args...]
# Output: $LOCUS_ROOT/work/runs/$LCAM_ROUND/<X>-s<seed>/, or $OUT if set.
set -euo pipefail
source "$(dirname "$0")/env.sh"
CONFIG=$1; SEED=$2; INIT=$3; PORT=${4:-29700}; shift $(( $# < 4 ? $# : 4 ))
FAM=$(basename "$CONFIG" .yaml)
OUT=${OUT:-$LOCUS_ROOT/work/runs/$LCAM_ROUND/$FAM-s$SEED}
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN
mkdir -p "$OUT" "$LOCUS_ROOT/work/logs"
cd "$LCAM_CODE"
exec "$PY" -m torch.distributed.run --nproc-per-node "${NPROC:-8}" --master-port "$PORT" \
  scripts/train_grpo.py --config "$CONFIG" --seed "$SEED" --bundle "$BUNDLE" --out "$OUT" --init-adapter "$INIT" "$@"
