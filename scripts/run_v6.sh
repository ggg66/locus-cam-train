#!/bin/bash
# v6 final test (docs/v6/README.md), everything the conclusion needs in one command and one result tar:
#   v6        Qwen3.5-35B-A3B: A_imagine (imagine, then act: the method chosen on the v5 development split) and A_propose
#             (the same candidates and choice without the imagined outcomes, the ablation), 3 seeds each
#   v6_gemma  Gemma 4 31B: A_imagine, 3 seeds (the second base)
# on every training family, with the protocol of the finished controls (v2 / v3_gemma A_base and A_compute, not retrained),
# scored once on the frozen test set. Also packed: the v5 raw answers and runtime patches, and the hosted-model diagnostics
# (run here when AIHUB_API_KEY is set and they are not finished yet).
#
#   LOCUS_ROOT=/shared/project NODES=/shared/project/work/v6/nodes.txt [AIHUB_API_KEY=...] nohup scripts/run_v6.sh > /dev/null 2>&1 &
#
# Stages: setup (files, model types, data alignment, propose_v6), unit, check (both processors on every training record of
# the three sources), train (9 runs, one per node at a time), val (validation V0 of every checkpoint, then checkpoint
# selection, reported separately), test (the v2 suite on every endpoint and on B0 of each base, an environment check
# against the B0 of v2 / v3_gemma; then the suite on validation-best checkpoints that are not the endpoint), confirm (C1 / C2),
# pack. Inference is batch 1, as in v2 and v3_gemma. The criteria are fixed in docs/v6/README.md before this runs.
# NODES (one IP per line, this node included) is optional; without it everything runs on this node in turn.
# Finished stages and runs are skipped, so after a stop fix the cause and run the same command again.
set -euo pipefail
export LCAM_ROUND=v6
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
QWEN=$LCAM_MODEL
GEMMA=${GEMMA_MODEL:-$LOCUS_ROOT/models/gemma-4-31B-it}
W=$LOCUS_ROOT/work/v6
ST=$W/stages
ROUNDS="v6 v6_gemma"
RUNSPEC=$(for f in A_imagine A_propose; do for s in 0 1 2; do echo "v6 $f $s"; done; done; for s in 0 1 2; do echo "v6_gemma A_imagine $s"; done)
mkdir -p "$W" "$ST" "$LOCUS_ROOT/work/logs" "$LOCUS_ROOT/work/runs/v6" "$LOCUS_ROOT/work/runs/v6_gemma" "$LOCUS_ROOT/work/v6_gemma"
exec > >(tee -a "$W/run_v6.log") 2>&1
for r in ${REDO:-}; do rm -f "${ST:?}/${r:?}.ok"; done
fail() { echo "STOPPED: $*"; echo "fix the cause and run the same command again (finished steps are skipped)"; exit 1; }
model_of() { if [ "$1" = v6_gemma ]; then echo "$GEMMA"; else echo "$QWEN"; fi; }
finished() { grep -q '"finished"' "$LOCUS_ROOT/work/runs/$1/run.json" 2>/dev/null; }
echo "== run_v6 $(date '+%F %T')  code $(git rev-parse --short HEAD 2>/dev/null || echo no-git)  qwen $QWEN  gemma $GEMMA"

once() {   # once <stage> <command...>: run a stage unless its marker exists
  local s=$1; shift
  if [ -f "$ST/$s.ok" ]; then echo "-- $s done earlier"; return; fi
  echo "== $s $(date '+%F %T')"
  "$@"
  touch "$ST/$s.ok"
}

# ---------------------------------------------------------------- hosted-model diagnostics, in the background
API_PID=""                                           # one copy only, also across reruns of this script
if [ -f "$LOCUS_ROOT/work/api_diag.tar.gz" ]; then echo "-- hosted-model diagnostics finished earlier"
elif [ -f "$W/api_diag.pid" ] && kill -0 "$(cat "$W/api_diag.pid")" 2>/dev/null; then API_PID=$(cat "$W/api_diag.pid"); echo "-- hosted-model diagnostics still running (pid $API_PID)"
elif [ -n "${AIHUB_API_KEY:-}" ]; then
  echo "== hosted-model diagnostics started in the background (work/api_diag/, log work/v6/api_diag.log)"
  "$PY" -u scripts/run_api_diagnostics.py --bundle "$BUNDLE" --v0 "$LOCUS_ROOT/work/api" --out "$LOCUS_ROOT/work/api_diag" --gateway aihub >> "$W/api_diag.log" 2>&1 &
  API_PID=$!; echo "$API_PID" > "$W/api_diag.pid"
else echo "-- AIHUB_API_KEY not set: the hosted-model diagnostics are not run by this command"; fi

do_setup() {
  "$PY" -c "import sys, transformers, peft, torch; from transformers.models.auto.configuration_auto import CONFIG_MAPPING as M; print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__); sys.exit(0 if 'qwen3_5_moe' in M and 'gemma4' in M else 1)" \
    || fail 'this venv cannot load both qwen3_5_moe and gemma4 (use the venv of the v2, v3_gemma and v5 runs)'
  "$PY" -c "import json,sys; c=json.load(open('$QWEN/config.json')); sys.exit(0 if c['model_type']=='qwen3_5_moe' else 1)" || fail "$QWEN is not Qwen3.5-35B-A3B"
  "$PY" -c "import json,sys; c=json.load(open('$GEMMA/config.json')); sys.exit(0 if c['model_type']=='gemma4' else 1)" || fail "$GEMMA is not Gemma 4 31B (set GEMMA_MODEL)"
  for m in "$BUNDLE"/ADDON_*_MANIFEST.json; do
    [ -f "$m" ] || continue
    "$PY" -c "import json,hashlib,sys; b='$BUNDLE'; m=json.load(open('$m')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(b+'/'+p,'rb').read()).hexdigest()!=h]; print('$(basename "$m")', len(m['files']), 'files, bad', bad); sys.exit(1 if bad else 0)" \
      || fail "bundle files differ from $(basename "$m")"
  done
  for f in training_exports/train/action_sft.jsonl training_exports/train/imagact_v5.jsonl training_exports/validation/imagact_v5.jsonl \
           eval_prompts/validation_V0.jsonl validation_tables/index.json \
           eval_prompts/test_V0.jsonl eval_prompts/test_variants_V0.jsonl eval_prompts/test_target_swap_V0.jsonl \
           eval_prompts/test_diagnostic_geom_predict.jsonl eval_prompts/test_diagnostic_consequence.jsonl \
           eval_prompts/dev_V0.jsonl eval_prompts/dev_V0_placeholder.jsonl \
           eval_prompts/confirm_c1_direct_choice.jsonl eval_prompts/confirm_c2_reference.jsonl; do
    [ -f "$BUNDLE/$f" ] || fail "$BUNDLE/$f missing: the bundle of the v5 runs (v1 bundle with the v2, confirm-1026 and v5 add-ons) is needed"
  done
  "$PY" - "$BUNDLE" <<'PYEOF' || fail 'imagact_v5 is not aligned with action_sft (the roots would not come in the order of A_base)'
import json, sys
b = sys.argv[1] + '/training_exports/train/'
a, m = ([json.loads(l)['root_id'] for l in open(b + f + '.jsonl')] for f in ('action_sft', 'imagact_v5'))
print('training roots', len(a), 'imagact_v5 aligned', a == m)
sys.exit(0 if a == m else 1)
PYEOF
  "$PY" scripts/make_propose_v6.py --bundle "$BUNDLE" || fail 'could not build propose_v6'
  (git rev-parse HEAD 2>/dev/null && git status --short) > "$W/code_commit.txt" || echo 'no git metadata (source snapshot)' > "$W/code_commit.txt"
}

do_unit() {
  "$PY" -m unittest discover -s tests > "$W/unit.log" 2>&1 || { tail -30 "$W/unit.log"; fail 'unit tests failed'; }
  tail -3 "$W/unit.log"
  LCAM_CONFIRM_PROMPTS=$BUNDLE/eval_prompts "$PY" -m unittest tests.test_confirm || fail 'C2 prompt rebuild differs from the frozen prompts'
}

do_check() {   # every training record: the development runs never encoded F09 / F14, Gemma never encoded these sources
  local r kinds
  for r in $ROUNDS; do
    kinds="imagact_v5"; [ "$r" = v6 ] && kinds="imagact_v5 propose_v6"
    LCAM_ROUND=$r LCAM_MODEL=$(model_of "$r") "$PY" scripts/check_data.py --config "configs/$r/common.yaml" --bundle "$BUNDLE" --split train --kinds $kinds --limit 0 > "$W/check_data-$r.json" \
      || fail "data check ($r) crashed"
    "$PY" -c "import json,sys; d=json.load(open('$W/check_data-$r.json')); print('$r', {k:(v['n'], v['n_errors'], v['tokens']['max']) for k,v in d['kinds'].items()}); sys.exit(0 if d['pass'] else 1)" \
      || fail "encoding errors in the $r training data (see $W/check_data-$r.json)"
  done
}

node_list() {
  if [ -n "${NODES:-}" ]; then grep -v '^#' "$NODES" | awk 'NF {print $1}'; else echo "$SELF_IP"; fi
}

one_run() {     # one_run <round> <family> <seed> <ip>: one training run on the 8 GPUs of a node; returns when it ends
  local r=$1 f=$2 s=$3 ip=$4 m log cmd
  m=$(model_of "$r"); log=$LOCUS_ROOT/work/logs/$r-$f-s$s.log
  if [ "$ip" = "$SELF_IP" ]; then LCAM_ROUND=$r LCAM_MODEL=$m launch/launch_run_v2.sh "configs/$r/$f.yaml" "$s" 29640 > "$log" 2>&1; return; fi
  printf -v cmd 'cd %q && LOCUS_ROOT=%q LCAM_ROUND=%q LCAM_MODEL=%q LCAM_CODE=%q BUNDLE=%q PY=%q SELF_IP=%q launch/launch_run_v2.sh %q %q 29640 > %q 2>&1' \
    "$LCAM_CODE" "$LOCUS_ROOT" "$r" "$m" "$LCAM_CODE" "$BUNDLE" "$PY" "$ip" "configs/$r/$f.yaml" "$s" "$log"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes "$ip" "$cmd" </dev/null
}

do_train() {    # the 9 runs, one per node at a time
  local r f s p n=0 failed=""; local -a nodes pids     # pids: a plain `wait` would also wait for the log's tee
  mapfile -t nodes < <(node_list)
  while read -r r f s; do
    if finished "$r/$f-s$s"; then echo "-- $r $f-s$s finished earlier"; continue; fi
    echo "== train $r $f-s$s on ${nodes[n]} $(date '+%F %T')"
    one_run "$r" "$f" "$s" "${nodes[n]}" &
    pids+=($!)
    n=$((n+1))
    if [ "$n" = "${#nodes[@]}" ]; then
      for p in "${pids[@]}"; do wait "$p" || true; done
      pids=(); n=0
    fi
  done <<< "$RUNSPEC"
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  while read -r r f s; do finished "$r/$f-s$s" || failed="$failed $r/$f-s$s"; done <<< "$RUNSPEC"
  [ -z "$failed" ] || fail "training ended without finishing:$failed (work/logs/<round>-<run>.log)"
}

pool() {        # pool <round> <task file>: run it over the nodes with that round's model and config, until every job is done
  local r=$1 tasks=$2 nodes=${NODES:-} left out
  [ -s "$tasks" ] || return 0
  if [ -z "$nodes" ]; then nodes=$W/nodes-self.txt; echo "$SELF_IP" > "$nodes"; fi
  rm -rf "$LOCUS_ROOT/work/claims/$(basename "$tasks" .txt)"
  local flog=$LOCUS_ROOT/work/logs/infer/failures.log f0=0            # only failures of this attempt count
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

do_val() {
  local r f s
  for r in $ROUNDS; do
    LCAM_ROUND=$r LCAM_INFER_BATCH=1 "$PY" scripts/make_v2_tasks.py val > "$W/tasks-$r-val.txt"
    pool "$r" "$W/tasks-$r-val.txt"
  done
  while read -r r f s; do
    "$PY" scripts/select_checkpoint.py --bundle "$BUNDLE" --run "$LOCUS_ROOT/work/runs/$r/$f-s$s" || fail "checkpoint selection failed for $r/$f-s$s"
  done <<< "$RUNSPEC"
}

do_test() {     # the v2 suite: endpoint checkpoint of every run (A_imagine / A_propose: 1024-token floor) and B0
  local r n want
  for r in $ROUNDS; do
    LCAM_ROUND=$r LCAM_INFER_BATCH=1 "$PY" scripts/make_v2_tasks.py test > "$W/tasks-$r-test.txt"
    pool "$r" "$W/tasks-$r-test.txt"
    want=$(( ($(grep -c "^$r " <<< "$RUNSPEC") + 1) * 600 ))
    n=$(cat "$LOCUS_ROOT"/work/outputs/$r/*/test_V0.jsonl | wc -l | tr -d " ")
    [ "$n" = "$want" ] || fail "$r: expected $want test_V0 answers (every run and B0, 600 each), found $n"
    LCAM_ROUND=$r LCAM_INFER_BATCH=1 "$PY" scripts/make_v2_tasks.py valbest --root "$LOCUS_ROOT" > "$W/tasks-$r-valbest.txt"
    pool "$r" "$W/tasks-$r-valbest.txt"
  done
}

do_confirm() {  # C1 / C2 (the frozen settings: batch 1, 2048 tokens); C2 is built from each run's own test_V0 answers
  local r
  for r in $ROUNDS; do
    LCAM_ROUND=$r "$PY" scripts/make_confirm_tasks.py --root "$LOCUS_ROOT" --bundle "$BUNDLE" > "$LOCUS_ROOT/work/$r/tasks-$r-confirm.txt" || fail "$r: confirm tasks"
    pool "$r" "$LOCUS_ROOT/work/$r/tasks-$r-confirm.txt"
    echo "$r confirm answers: $(cat "$LOCUS_ROOT"/work/outputs/$r-confirm/*/*.jsonl | wc -l)"
  done
}

do_pack() {
  if [ -n "$API_PID" ]; then                          # not always our child (an earlier run of this script may have started it)
    echo "-- waiting for the hosted-model diagnostics"
    while kill -0 "$API_PID" 2>/dev/null; do sleep 60; done
    [ -f "$LOCUS_ROOT/work/api_diag.tar.gz" ] || echo "hosted-model diagnostics ended without their tar (work/v6/api_diag.log); packed without them"
  fi
  cd "$LOCUS_ROOT"
  local out=work/v6/locus-v6-results-$(date +%m%d).tar r list
  : > work/v6/checkpoint_sha256.txt
  for r in $ROUNDS; do sha256sum work/runs/$r/*/checkpoints/ckpt-*/adapter_model.safetensors >> work/v6/checkpoint_sha256.txt; done
  (cd "$LCAM_CODE" && git diff > "$LOCUS_ROOT/work/v6/code_diff.patch" 2>/dev/null && git status --porcelain | awk '$1 == "??" {print $2}' \
     | grep -E '\.(py|sh)$' | tar -czf "$LOCUS_ROOT/work/v6/code_untracked.tar.gz" -T - 2>/dev/null) || true
  list=$(for r in $ROUNDS; do ls -d work/outputs/$r work/outputs/$r-valbest work/outputs/$r-confirm work/$r/confirm work/$r/*.txt \
                                     work/runs/$r/*/run.json work/runs/$r/*/metrics.jsonl work/runs/$r/*/trainable_manifest.json work/runs/$r/*/selection.json \
                                     work/runs/$r/*/val work/runs/$r/*/checkpoints/*/exposures.json work/logs/$r-*.log 2>/dev/null; done
         ls -d work/v6/*.txt work/v6/*.json work/v6/*.log work/v6/code_diff.patch work/v6/code_untracked.tar.gz \
               work/outputs/v5 work/v5/dev_summary.* work/v5/runtime_fixes work/runs/v5/*/metrics.jsonl work/runs/v5/*/run.json \
               work/api_diag.tar.gz work/api_diag.tar.gz.sha256 2>/dev/null || true)
  tar -cf "$out" --exclude='*.safetensors' --exclude='*.pt' $list
  sha256sum "$out" | tee "$out.sha256"
  echo "packed: $(tar -tf "$out" | grep -c 'test_V0.jsonl$') test_V0 files; v5 answers $(tar -tf "$out" | grep -c '^work/outputs/v5/.*jsonl$') files; hosted diagnostics $(tar -tf "$out" | grep -c 'api_diag.tar.gz$')"
  cd "$LCAM_CODE"
}

once setup do_setup
once unit do_unit
once check do_check
once train do_train
once val do_val
once test do_test
once confirm do_confirm
once pack do_pack
echo "== v6 finished $(date '+%F %T'); send work/v6/locus-v6-results-*.tar and its .sha256 back"
