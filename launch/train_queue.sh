#!/bin/bash
# Train every run of the round on the nodes of NODES, one 8-GPU run per node at a time, until all are finished.
#   launch/train_queue.sh <nodefile>
# Runs = 3 families x the seeds of configs/$LCAM_ROUND/common.yaml (scripts/make_v2_tasks.py train). A node is free when
# no scripts/train_v2.py process runs on it. A run that stops without "finished" in its run.json is moved aside to
# <run>.failed-<time>/ and started again once; a second failure stops the queue (exit 1). Finished runs are skipped,
# so the script can be rerun after a stop. Logs: $LOCUS_ROOT/work/logs/$LCAM_ROUND-<run>.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
NODES=${1:?node file}
case "$NODES" in /*) ;; *) NODES=$LOCUS_ROOT/$NODES ;; esac
Q=$LOCUS_ROOT/work/$LCAM_ROUND/queue
mkdir -p "$Q" "$LOCUS_ROOT/work/logs"
cd "$LCAM_CODE"
mapfile -t NODE_IPS < <(grep -v '^#' "$NODES" | awk 'NF {print $1}')
mapfile -t RUNS < <("$PY" scripts/make_v2_tasks.py train --nodes "$NODES" | awk '{print $2" "$3}')
LAUNCH=${LCAM_TRAIN_LAUNCH:-$LCAM_CODE/launch/launch_run_v2.sh}   # override only for tests
PORT=${PORT:-29600}

on_node() {     # on_node <ip> <command>: run a command on a node (locally for this node)
  if [ "$1" = "$SELF_IP" ]; then bash -c "$2"
  else ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes "$1" "$2" </dev/null 2>/dev/null; fi
}
busy() { on_node "$1" "pgrep -f '[s]cripts/train_v2.py' >/dev/null"; }
finished() { grep -q '"finished"' "$LOCUS_ROOT/work/runs/$LCAM_ROUND/$1/run.json" 2>/dev/null; }

while true; do
  left=0
  for r in "${RUNS[@]}"; do
    set -- $r; config=$1; seed=$2
    tag=$(basename "$config" .yaml)-s$seed
    finished "$tag" && continue
    left=$((left+1))
    if [ -f "$Q/$tag.node" ]; then                       # started earlier: still running, or failed?
      node=$(cat "$Q/$tag.node")
      on_node "$node" "pgrep -f '[s]cripts/train_v2.py.*runs/$LCAM_ROUND/$tag ' >/dev/null" && continue
      sleep "${QUEUE_GRACE:-30}"; finished "$tag" && continue              # it may have just written "finished"
      tries=$(cat "$Q/$tag.tries" 2>/dev/null || echo 1)
      if [ "$tries" -ge 2 ]; then echo "$(date '+%F %T') $tag failed twice; see $LOCUS_ROOT/work/logs/$LCAM_ROUND-$tag.log"; exit 1; fi
      mv "$LOCUS_ROOT/work/runs/$LCAM_ROUND/$tag" "$LOCUS_ROOT/work/runs/$LCAM_ROUND/$tag.failed-$(date +%m%d%H%M)" 2>/dev/null
      cp "$LOCUS_ROOT/work/logs/$LCAM_ROUND-$tag.log" "$Q/$tag.failed-$(date +%m%d%H%M).log" 2>/dev/null
      echo "$(date '+%F %T') $tag stopped without finishing on $node; retrying"
      echo $((tries+1)) > "$Q/$tag.tries"; rm -f "${Q:?}/${tag:?}.node"
    fi
    for ip in "${NODE_IPS[@]}"; do                       # first free node that is not reserved for another run
      reserved=0
      for f in "$Q"/*.node; do [ -f "$f" ] || continue; t=$(basename "$f" .node); finished "$t" && continue; [ "$(cat "$f")" = "$ip" ] && reserved=1; done
      [ "$reserved" = 1 ] && continue
      busy "$ip" && continue
      log=$LOCUS_ROOT/work/logs/$LCAM_ROUND-$tag.log
      on_node "$ip" "LOCUS_ROOT=$LOCUS_ROOT LCAM_ROUND=$LCAM_ROUND LCAM_MODEL=$LCAM_MODEL PY=$PY BUNDLE=$BUNDLE setsid nohup $LAUNCH $config $seed $PORT > $log 2>&1 < /dev/null &"
      echo "$ip" > "$Q/$tag.node"; [ -f "$Q/$tag.tries" ] || echo 1 > "$Q/$tag.tries"
      echo "$(date '+%F %T') started $tag on $ip"
      sleep "${QUEUE_STAGGER:-20}"                       # stagger the weight reads from the shared filesystem
      break
    done
  done
  [ "$left" = 0 ] && { echo "$(date '+%F %T') all ${#RUNS[@]} runs finished"; exit 0; }
  sleep "${QUEUE_POLL:-120}"
done
