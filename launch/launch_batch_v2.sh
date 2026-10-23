#!/bin/bash
# One 8-GPU v2 run per node from a plan file of "<node_ip> <config> <seed>" lines (scripts/make_v2_tasks.py train).
#   launch/launch_batch_v2.sh work/v2/plan-train.txt
# Finished runs are skipped, running ones reported busy. Logs: $LOCUS_ROOT/work/logs/v2-<family>-s<seed>.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
PLAN=${1:?plan file}
case "$PLAN" in /*) ;; *) PLAN=$LOCUS_ROOT/$PLAN ;; esac
PORT=${PORT:-29600}
LAUNCH=$LCAM_CODE/launch/launch_run_v2.sh
while read -r ip config seed; do
  [ -z "${ip:-}" ] && continue
  case "$ip" in \#*) continue ;; esac
  fam=$(basename "$config" .yaml); tag=$fam-s$seed
  log=$LOCUS_ROOT/work/logs/v2-$tag.log
  if grep -q '"finished"' "$LOCUS_ROOT/work/runs/v2/$tag/run.json" 2>/dev/null; then echo "skip finished  $tag"; continue; fi
  busy="pgrep -f '[s]cripts/train_v2.py.*runs/v2/$tag ' >/dev/null"
  start="LOCUS_ROOT=$LOCUS_ROOT setsid nohup $LAUNCH $config $seed $PORT > $log 2>&1 < /dev/null &"
  if [ "$ip" = "$SELF_IP" ]; then
    if eval "$busy"; then echo "already running  $tag"; else eval "$start"; echo "started  $tag (local)"; fi
  else
    r=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes "$ip" \
      "if $busy; then echo busy; else $start echo ok; fi" </dev/null 2>/dev/null | tail -1)
    echo "$tag  on $ip : ${r:-ssh-failed}"
  fi
  sleep 20          # stagger the 72 GB weight reads from the shared filesystem
done < "$PLAN"
