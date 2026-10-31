#!/bin/bash
# Work-pool inference over the GPUs of several nodes (scripts/infer.py, one job per GPU at a time).
#   launch/run_infer_pool.sh <taskfile> <nodefile>
# Task line: <out>\t<prompts>\t<adapter|->\t<thinking 0|1>\t[batch]\t[max_new_tokens]; out and adapter are relative to
# $LOCUS_ROOT. Jobs are claimed through mkdir on the shared filesystem; infer.py skips finished prompts, so re-runs resume.
set -uo pipefail
source "$(dirname "$0")/env.sh"
SELFSH=$LCAM_CODE/launch/run_infer_pool.sh
CONFIG=${LCAM_INFER_CONFIG:-configs/$LCAM_ROUND/common.yaml}
TASKS=${1:?task file}; NODES=${2:?node file}
case "$TASKS" in /*) ;; *) TASKS=$LOCUS_ROOT/$TASKS ;; esac
case "$NODES" in /*) ;; *) NODES=$LOCUS_ROOT/$NODES ;; esac
CLAIM=$LOCUS_ROOT/work/claims/$(basename "$TASKS" .txt)
mkdir -p "$CLAIM" "$LOCUS_ROOT/work/logs/infer"
if [ "${LCAM_WORKER:-0}" = 1 ]; then
  n=0
  while IFS=$'\t' read -r out prompts adapter thinking batch maxnew; do
    [ -z "${out:-}" ] && continue
    case "$out" in \#*) continue ;; esac
    key=$(echo "$out" | md5sum | cut -c1-32)
    mkdir "$CLAIM/$key" 2>/dev/null || continue
    [ -e "$LOCUS_ROOT/$out.done" ] && continue
    args=(--config "$CONFIG" --bundle "$BUNDLE" --prompts "$prompts" --out "$LOCUS_ROOT/$out" --batch "${batch:-8}")
    [ -n "${maxnew:-}" ] && args+=(--max-new-tokens "$maxnew")
    [ "$adapter" != "-" ] && args+=(--adapter "$LOCUS_ROOT/$adapter")
    [ "$thinking" = "1" ] && args+=(--thinking)
    log=$LOCUS_ROOT/work/logs/infer/$(echo "$out" | tr '/' '_')-$SELF_IP-gpu$CUDA_VISIBLE_DEVICES.log
    cd "$LCAM_CODE"
    if "$PY" scripts/infer.py "${args[@]}" > "$log" 2>&1; then touch "$LOCUS_ROOT/$out.done"; n=$((n+1))
    else echo "FAILED $out (see $log)" >> "$LOCUS_ROOT/work/logs/infer/failures.log"; rmdir "$CLAIM/$key" 2>/dev/null; fi
  done < "$TASKS"
  echo "worker $SELF_IP gpu$CUDA_VISIBLE_DEVICES done $n jobs"
  exit 0
fi
echo "jobs: $(grep -cv '^#' "$TASKS")   nodes: $(grep -cv '^#' "$NODES")"
while read -r ip; do
  [ -z "${ip:-}" ] && continue
  case "$ip" in \#*) continue ;; esac
  for g in 0 1 2 3 4 5 6 7; do
    printf -v cmd 'LOCUS_ROOT=%q LCAM_ROUND=%q LCAM_MODEL=%q LCAM_CODE=%q BUNDLE=%q PY=%q SELF_IP=%q PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=%q LCAM_WORKER=1 LCAM_INFER_CONFIG=%q setsid nohup %q %q %q > %q 2>&1 < /dev/null &' \
      "$LOCUS_ROOT" "$LCAM_ROUND" "$LCAM_MODEL" "$LCAM_CODE" "$BUNDLE" "$PY" "$ip" "$g" "$CONFIG" "$SELFSH" "$TASKS" "$NODES" "$LOCUS_ROOT/work/logs/infer/worker-$ip-$g.log"
    if [ "$ip" = "$SELF_IP" ]; then eval "$cmd"
    else ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes "$ip" "$cmd" </dev/null 2>/dev/null; fi
  done
  echo "  8 workers on $ip"
done < "$NODES"
