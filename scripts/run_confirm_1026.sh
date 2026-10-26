#!/bin/bash
# The two inference controls of 2026-10-26 (no training): C1 direct choice on the 12 panel candidates, C2 judging one
# action (the run's own V0 action and a fixed reference feasible action). Prompts and task lists come in the bundle add-on
# locus-bundle-v1.7.0-addon-confirm-1026.tar; nothing in it is changed here.
#   LOCUS_ROOT=... scripts/run_confirm_1026.sh <stage> [<stage> ...]
# Stages, in order:
#   addon     check the add-on sha256 and extract it into $BUNDLE (new files only)       (ADDON_TAR, ADDON_SHA256)
#   ckpt      sha256 of the nine v2 ckpt-4 adapters -> work/confirm_1026/checkpoint_sha256.txt
#   dev       36 dev requests (B0 and AG-s0) and a format check: JSON parses, C1 choice in A-L, C2 has the five fields.
#             Stops when more than 4 of the 36 answers fail it; prompts are never edited on the basis of these answers.
#   run       the 3,360 requests (28 jobs) on NODES (default work/v2/nodes.txt), waits until every job is done
#   pack      one tar of the answers, task lists, checkpoint hashes and inference logs (no weights, no images)
set -euo pipefail
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
W=$LOCUS_ROOT/work/confirm_1026
NODES=${NODES:-$LOCUS_ROOT/work/v2/nodes.txt}
mkdir -p "$W" "$LOCUS_ROOT/work/logs"

wait_pool() {   # until every job of a task file has its .done marker; stops when one of them failed
  local tasks=$1 fails=$LOCUS_ROOT/work/logs/infer/failures.log
  while true; do
    local left=0
    while IFS=$'\t' read -r out _; do
      [ -z "$out" ] && continue
      [ -e "$LOCUS_ROOT/$out.done" ] && continue
      left=$((left+1))
      if [ -f "$fails" ] && grep -qF "FAILED $out " "$fails"; then echo "job failed: $out (see $fails)"; exit 1; fi
    done < "$tasks"
    [ "$left" = 0 ] && break
    echo "$(date +%T) $left jobs left in $(basename "$tasks")"; sleep 60
  done
}

for stage in "$@"; do
  echo "== $stage $(date '+%F %T')"
  case $stage in
    addon)
      : "${ADDON_TAR:?set ADDON_TAR to locus-bundle-v1.7.0-addon-confirm-1026.tar}" "${ADDON_SHA256:?set ADDON_SHA256}"
      echo "$ADDON_SHA256  $ADDON_TAR" | sha256sum -c -
      tar -xf "$ADDON_TAR" -C "$BUNDLE" --keep-old-files
      "$PY" -c "import json,hashlib,sys; b='$BUNDLE'; m=json.load(open(b+'/ADDON_CONFIRM_1026_MANIFEST.json')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(b+'/'+p,'rb').read()).hexdigest()!=h]; print('add-on files', len(m['files']), 'bad', bad); sys.exit(1 if bad else 0)"
      cp "$BUNDLE/confirm_1026/tasks-confirm.txt" "$BUNDLE/confirm_1026/tasks-confirm-dev.txt" "$BUNDLE/confirm_1026/request_manifest.csv" "$W/" ;;
    ckpt)
      sha256sum "$LOCUS_ROOT"/work/runs/v2/*/checkpoints/ckpt-4/adapter_model.safetensors | tee "$W/checkpoint_sha256.txt"
      (cd "$LCAM_CODE" && git rev-parse HEAD) > "$W/code_commit.txt" ;;
    dev)
      LCAM_INFER_BATCH=1 launch/run_infer_pool.sh "$W/tasks-confirm-dev.txt" "$NODES"
      wait_pool "$W/tasks-confirm-dev.txt"
      "$PY" - "$LOCUS_ROOT" <<'PYEOF'
import glob, json, os, sys
root = sys.argv[1]
def last_json(t):
    best, depth, start = None, 0, None
    for i, ch in enumerate(t or ''):
        if ch == '{':
            start = i if depth == 0 else start
            depth += 1
        elif ch == '}' and depth:
            depth -= 1
            if depth == 0:
                try:
                    best = json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    pass
    return best
bad, n = [], 0
for f in sorted(glob.glob(os.path.join(root, 'work/outputs/confirm_1026/dev/*/*.jsonl'))):
    for l in open(f):
        o, n = json.loads(l), n + 1
        j = last_json(o['raw'])
        if 'c1' in f:
            ok = isinstance(j, dict) and j.get('choice') in list('ABCDEFGHIJKL')
        else:
            ok = isinstance(j, dict) and all(k in j for k in ('legal', 'in_frame', 'size_ok', 'unoccluded', 'apart'))
        if not ok or o.get('finish') != 'stop':
            bad.append((os.path.relpath(f, root), o['task_id'], o.get('finish'), (o['raw'] or '')[:120]))
print(json.dumps({'dev_answers': n, 'format_failures': len(bad), 'examples': bad[:8]}, ensure_ascii=False, indent=1))
json.dump({'dev_answers': n, 'format_failures': bad}, open(os.path.join(root, 'work/confirm_1026/dev_check.json'), 'w'), ensure_ascii=False, indent=1)
sys.exit(0 if n == 36 and len(bad) <= 4 else 1)
PYEOF
      ;;
    run)
      [ -f "$W/dev_check.json" ] || { echo 'run the dev stage first'; exit 1; }
      LCAM_INFER_BATCH=1 launch/run_infer_pool.sh "$W/tasks-confirm.txt" "$NODES"
      wait_pool "$W/tasks-confirm.txt"
      echo "answers: $(cat "$LOCUS_ROOT"/work/outputs/confirm_1026/{B0,A_*,AG-*}/*.jsonl 2>/dev/null | wc -l) (expected 3360)" ;;
    pack)
      cd "$LOCUS_ROOT"
      out=work/confirm_1026/confirm-1026-results.tar
      tar -cf "$out" work/outputs/confirm_1026 work/confirm_1026/*.txt work/confirm_1026/*.csv work/confirm_1026/*.json work/logs/infer
      sha256sum "$out" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
