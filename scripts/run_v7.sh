#!/bin/bash
# v7 (docs/v7/README.md): inference only, on the finished checkpoints of v2, v3_gemma, v4, v6 and v6_gemma. No training.
#   S          every fine-tuned run and both untrained bases on the 120 diagnostic scenes with the reference geometry
#              written into the prompt (does the same fine-tuned model plan well once the geometry is given?)
#   geometry   the Action+Geometry runs of both bases write geometry for all 600 test scenes (the full geometry route)
#   prefix     the Geometry->Action runs continue an answer that starts with the true geometry, or with that geometry
#              mirrored left-right, on all 600 test scenes (does the action follow the geometry the answer states?)
#
#   LOCUS_ROOT=/shared/project NODES=/shared/project/work/v7/nodes.txt ADDON_TAR=.../locus-bundle-v1.7.0-addon-v7.tar \
#     ADDON_SHA256=<sha> nohup scripts/run_v7.sh > /dev/null 2>&1 &
#
# Stages: setup (add-on, files, unit tests), infer (one task file per round, over the GPUs of every node), pack.
# Runs whose endpoint checkpoint is missing are skipped and listed in work/v7/missing.txt. Inference is batch 1 and
# greedy, as in every earlier round. Finished stages and answers are skipped, so after a stop fix the cause and rerun.
set -euo pipefail
export LCAM_ROUND=v2
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
QWEN=$LCAM_MODEL
GEMMA=${GEMMA_MODEL:-$LOCUS_ROOT/models/gemma-4-31B-it}
W=$LOCUS_ROOT/work/v7
ST=$W/stages
ROUNDS="v2 v4 v6 v3_gemma v6_gemma"
mkdir -p "$W" "$ST" "$LOCUS_ROOT/work/logs"
exec > >(tee -a "$W/run_v7.log") 2>&1
for r in ${REDO:-}; do rm -f "${ST:?}/${r:?}.ok"; done
fail() { echo "STOPPED: $*"; echo "fix the cause and run the same command again (finished steps are skipped)"; exit 1; }
model_of() { case "$1" in v3_gemma|v6_gemma) echo "$GEMMA" ;; *) echo "$QWEN" ;; esac; }
echo "== run_v7 $(date '+%F %T')  code $(git rev-parse --short HEAD 2>/dev/null || echo no-git)  qwen $QWEN  gemma $GEMMA"

once() {
  local s=$1; shift
  if [ -f "$ST/$s.ok" ]; then echo "-- $s done earlier"; return; fi
  echo "== $s $(date '+%F %T')"
  "$@"
  touch "$ST/$s.ok"
}

do_setup() {
  if [ -n "${ADDON_TAR:-}" ]; then
    : "${ADDON_SHA256:?set ADDON_SHA256 with ADDON_TAR}"
    echo "$ADDON_SHA256  $ADDON_TAR" | sha256sum -c - || fail 'add-on sha256 differs'
    tar -xf "$ADDON_TAR" -C "$BUNDLE" || fail 'could not extract the add-on'
  fi
  "$PY" -c "import json,hashlib,sys; b='$BUNDLE'; m=json.load(open(b+'/ADDON_V7_MANIFEST.json')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(b+'/'+p,'rb').read()).hexdigest()!=h]; print('add-on v7', len(m['files']), 'files, bad', bad); sys.exit(1 if bad else 0)" \
    || fail 'add-on v7 missing or changed (ADDON_TAR / ADDON_SHA256)'
  for f in eval_prompts/test_diagnostic_S.jsonl eval_prompts/test_geom_predict.jsonl eval_prompts/test_geomact_true.jsonl eval_prompts/test_geomact_mirror.jsonl; do
    [ -f "$BUNDLE/$f" ] || fail "$BUNDLE/$f missing"
  done
  "$PY" -c "import json,sys; c=json.load(open('$GEMMA/config.json')); sys.exit(0 if c['model_type']=='gemma4' else 1)" || fail "$GEMMA is not Gemma 4 31B (set GEMMA_MODEL)"
  "$PY" -m unittest tests.test_v7 || fail 'v7 unit tests failed'
  (git rev-parse HEAD 2>/dev/null && git status --short) > "$W/code_commit.txt" || echo 'no git metadata' > "$W/code_commit.txt"
}

pool() {        # pool <round> <task file>: as in run_v6.sh
  local r=$1 tasks=$2 nodes=${NODES:-} left
  [ -s "$tasks" ] || return 0
  if [ -z "$nodes" ]; then nodes=$W/nodes-self.txt; echo "$SELF_IP" > "$nodes"; fi
  rm -rf "$LOCUS_ROOT/work/claims/$(basename "$tasks" .txt)"
  local flog=$LOCUS_ROOT/work/logs/infer/failures.log f0=0
  [ -f "$flog" ] && f0=$(wc -l < "$flog" | tr -d " ")
  LCAM_ROUND=$r LCAM_MODEL=$(model_of "$r") launch/run_infer_pool.sh "$tasks" "$nodes"
  while true; do
    left=0
    while IFS=$'\t' read -r out _; do
      [ -z "$out" ] && continue
      case "$out" in \#*) continue ;; esac
      [ -e "$LOCUS_ROOT/$out.done" ] && continue
      left=$((left+1))
      [ -f "$flog" ] && tail -n +$((f0+1)) "$flog" | grep -qF "FAILED $out " && fail "inference failed: $out (work/logs/infer/failures.log)"
    done < "$tasks"
    [ "$left" = 0 ] && break
    echo "$(date +%T) $r: $left jobs left in $(basename "$tasks")"; sleep 60
  done
}

do_infer() {
  local r
  : > "$W/missing.txt"
  for r in $ROUNDS; do
    LCAM_ROUND=$r "$PY" scripts/make_v7_tasks.py --root "$LOCUS_ROOT" > "$W/tasks-v7-$r.txt" 2>> "$W/missing.txt" || fail "$r: task list"
    echo "$r: $(grep -cv '^#' "$W/tasks-v7-$r.txt") jobs"
    pool "$r" "$W/tasks-v7-$r.txt"
  done
  [ -s "$W/missing.txt" ] && { echo "skipped (no endpoint checkpoint):"; cat "$W/missing.txt"; } || echo "every run had its endpoint checkpoint"
}

do_pack() {
  cd "$LOCUS_ROOT"
  local out=work/v7/locus-v7-results-$(date +%m%d).tar
  tar -cf "$out" work/outputs/v7 work/v7/*.txt work/v7/run_v7.log
  sha256sum "$out" | tee "$out.sha256"
  echo "packed: $(tar -tf "$out" | grep -c '\.jsonl$') answer files"
  cd "$LCAM_CODE"
}

once setup do_setup
once infer do_infer
once pack do_pack
echo "== v7 finished $(date '+%F %T'); send work/v7/locus-v7-results-*.tar and its .sha256 back"
