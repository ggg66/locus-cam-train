#!/bin/bash
# v6 final test (docs/v6/README.md): the method chosen on the v5 development split, "imagine, then act", trained on every
# training family and scored once on the frozen test set.
#
#   LOCUS_ROOT=/shared/project NODES=/shared/project/work/v6/nodes.txt nohup scripts/run_v6.sh > /dev/null 2>&1 &
#
# Stages: setup (files, model type), unit, check (real processor on every imagact_v5 training record), train (A_imagine,
# 3 seeds, one node each), test (the v2 evaluation suite on the endpoint checkpoint of every seed, and on B0 as an
# environment check against v2's B0), pack. Same protocol as v2 (configs/v6/common.yaml); the controls A_base / A_compute
# are the finished v2 runs and are not retrained. The criterion is fixed in docs/v6/README.md before this runs.
# NODES (one IP per line, this node included) is optional; without it everything runs on this node in turn.
# Finished stages and runs are skipped, so after a stop fix the cause and run the same command again.
set -euo pipefail
export LCAM_ROUND=v6
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
W=$LOCUS_ROOT/work/v6
ST=$W/stages
RUNS=$LOCUS_ROOT/work/runs/v6
FAM=A_imagine
SEEDS="0 1 2"
mkdir -p "$W" "$ST" "$RUNS" "$LOCUS_ROOT/work/logs"
exec > >(tee -a "$W/run_v6.log") 2>&1
for r in ${REDO:-}; do rm -f "${ST:?}/${r:?}.ok"; done
fail() { echo "STOPPED: $*"; echo "fix the cause and run the same command again (finished steps are skipped)"; exit 1; }
finished() { grep -q '"finished"' "$RUNS/$1/run.json" 2>/dev/null; }
echo "== run_v6 $(date '+%F %T')  code $(git rev-parse --short HEAD 2>/dev/null || echo no-git)  model $LCAM_MODEL"

once() {   # once <stage> <command...>: run a stage unless its marker exists
  local s=$1; shift
  if [ -f "$ST/$s.ok" ]; then echo "-- $s done earlier"; return; fi
  echo "== $s $(date '+%F %T')"
  "$@"
  touch "$ST/$s.ok"
}

do_setup() {
  "$PY" -c "import sys, transformers, peft, torch; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__); sys.exit(0 if 'qwen3_5_moe' in CONFIG_MAPPING else 1)" \
    || fail 'this venv cannot load qwen3_5_moe (use the venv of the v2 and v5 runs)'
  "$PY" -c "import json,sys; c=json.load(open('$LCAM_MODEL/config.json')); sys.exit(0 if c['model_type']=='qwen3_5_moe' else 1)" || fail "$LCAM_MODEL is not Qwen3.5-35B-A3B"
  for m in "$BUNDLE"/ADDON_*_MANIFEST.json; do
    [ -f "$m" ] || continue
    "$PY" -c "import json,hashlib,sys; b='$BUNDLE'; m=json.load(open('$m')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(b+'/'+p,'rb').read()).hexdigest()!=h]; print('$(basename "$m")', len(m['files']), 'files, bad', bad); sys.exit(1 if bad else 0)" \
      || fail "bundle files differ from $(basename "$m")"
  done
  for f in training_exports/train/action_sft.jsonl training_exports/train/imagact_v5.jsonl training_exports/validation/imagact_v5.jsonl \
           eval_prompts/test_V0.jsonl eval_prompts/test_variants_V0.jsonl eval_prompts/test_target_swap_V0.jsonl \
           eval_prompts/test_diagnostic_geom_predict.jsonl eval_prompts/test_diagnostic_consequence.jsonl \
           eval_prompts/dev_V0.jsonl eval_prompts/dev_V0_placeholder.jsonl; do
    [ -f "$BUNDLE/$f" ] || fail "$BUNDLE/$f missing: the bundle of the v5 runs (v1 bundle with the v2, confirm-1026 and v5 add-ons) is needed"
  done
  "$PY" - "$BUNDLE" <<'PYEOF' || fail 'imagact_v5 is not aligned with action_sft (the roots would not come in the order of A_base)'
import json, sys
b = sys.argv[1] + '/training_exports/train/'
a, m = ([json.loads(l)['root_id'] for l in open(b + f + '.jsonl')] for f in ('action_sft', 'imagact_v5'))
print('training roots', len(a), 'imagact_v5 aligned', a == m)
sys.exit(0 if a == m else 1)
PYEOF
  (git rev-parse HEAD 2>/dev/null && git status --short) > "$W/code_commit.txt" || echo 'no git metadata (source snapshot)' > "$W/code_commit.txt"
}

do_unit() {
  "$PY" -m unittest discover -s tests > "$W/unit.log" 2>&1 || { tail -30 "$W/unit.log"; fail 'unit tests failed'; }
  tail -3 "$W/unit.log"
}

do_check() {   # every training record of the method, not a sample: the development runs never encoded F09 / F14
  "$PY" scripts/check_data.py --config configs/v6/common.yaml --bundle "$BUNDLE" --split train --kinds imagact_v5 --limit 0 > "$W/check_data-train.json" \
    || fail 'data check crashed'
  "$PY" -c "import json,sys; d=json.load(open('$W/check_data-train.json')); print({k:(v['n'], v['n_errors'], v['tokens']['max']) for k,v in d['kinds'].items()}); sys.exit(0 if d['pass'] else 1)" \
    || fail "encoding errors in the training data (see $W/check_data-train.json)"
}

node_list() {
  if [ -n "${NODES:-}" ]; then grep -v '^#' "$NODES" | awk 'NF {print $1}'; else echo "$SELF_IP"; fi
}

on_node() {     # on_node <ip> <log> <command...>: run a launch script there and return when it ends
  local ip=$1 log=$2 cmd; shift 2
  if [ "$ip" = "$SELF_IP" ]; then "$@" > "$log" 2>&1; return; fi
  printf -v cmd '%q ' "$@"
  printf -v cmd 'cd %q && LOCUS_ROOT=%q LCAM_ROUND=%q LCAM_MODEL=%q LCAM_CODE=%q BUNDLE=%q PY=%q SELF_IP=%q %s > %q 2>&1' \
    "$LCAM_CODE" "$LOCUS_ROOT" "$LCAM_ROUND" "$LCAM_MODEL" "$LCAM_CODE" "$BUNDLE" "$PY" "$ip" "$cmd" "$log"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes "$ip" "$cmd" </dev/null
}

one_seed() { on_node "$2" "$LOCUS_ROOT/work/logs/v6-$FAM-s$1.log" launch/launch_run_v2.sh "configs/v6/$FAM.yaml" "$1" 29640; }

do_train() {    # one seed per node at a time
  local s p n=0 failed=""; local -a nodes todo pids     # pids: a plain `wait` would also wait for the log's tee
  mapfile -t nodes < <(node_list)
  for s in $SEEDS; do
    if finished "$FAM-s$s"; then echo "-- $FAM-s$s finished earlier"; else todo+=("$s"); fi
  done
  for s in ${todo[@]+"${todo[@]}"}; do
    echo "== train $FAM-s$s on ${nodes[n]} $(date '+%F %T')"
    one_seed "$s" "${nodes[n]}" &
    pids+=($!)
    n=$((n+1))
    if [ "$n" = "${#nodes[@]}" ]; then
      for p in "${pids[@]}"; do wait "$p" || true; done
      pids=(); n=0
    fi
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  for s in $SEEDS; do finished "$FAM-s$s" || failed="$failed $FAM-s$s"; done
  [ -z "$failed" ] || fail "training ended without finishing:$failed (work/logs/v6-<run>.log)"
}

do_test() {     # the v2 suite, batch 1 as in v2, v4 and v5 (make_v2_tasks.py: endpoint checkpoint, A_imagine's 1024-token floor)
  local tasks=$W/tasks-test.txt nodes=${NODES:-} left
  LCAM_INFER_BATCH=1 "$PY" scripts/make_v2_tasks.py test > "$tasks"
  if [ -z "$nodes" ]; then nodes=$W/nodes-self.txt; echo "$SELF_IP" > "$nodes"; fi
  rm -rf "$LOCUS_ROOT/work/claims/$(basename "$tasks" .txt)"
  local flog=$LOCUS_ROOT/work/logs/infer/failures.log f0=0            # only failures of this attempt count
  [ -f "$flog" ] && f0=$(wc -l < "$flog")
  launch/run_infer_pool.sh "$tasks" "$nodes"
  while true; do
    left=0
    while IFS=$'\t' read -r out _; do
      [ -z "$out" ] && continue
      [ -e "$LOCUS_ROOT/$out.done" ] && continue
      left=$((left+1))
      [ -f "$flog" ] && tail -n +$((f0+1)) "$flog" | grep -qF "FAILED $out " && fail "inference failed: $out (work/logs/infer/failures.log)"
    done < "$tasks"
    [ "$left" = 0 ] && break
    echo "$(date +%T) $left test jobs left"; sleep 60
  done
  local n
  n=$(cat "$LOCUS_ROOT"/work/outputs/v6/*/test_V0.jsonl | wc -l)
  [ "$n" = 2400 ] || fail "expected 2400 test_V0 answers (3 seeds and B0, 600 each), found $n"
}

do_pack() {
  cd "$LOCUS_ROOT"
  local out=work/v6/locus-v6-results-$(date +%m%d).tar list
  list=$(ls -d work/outputs/v6 work/v6/*.txt work/v6/*.json work/v6/unit.log work/v6/run_v6.log work/runs/v6/*/run.json work/runs/v6/*/metrics.jsonl \
               work/runs/v6/*/trainable_manifest.json work/runs/v6/*/checkpoints/*/exposures.json work/logs/v6-*.log 2>/dev/null || true)
  sha256sum work/runs/v6/*/checkpoints/ckpt-4/adapter_model.safetensors > work/v6/checkpoint_sha256.txt
  tar -cf "$out" --exclude='*.safetensors' --exclude='*.pt' $list work/v6/checkpoint_sha256.txt
  sha256sum "$out" | tee "$out.sha256"
  cd "$LCAM_CODE"
}

once setup do_setup
once unit do_unit
once check do_check
once train do_train
once test do_test
once pack do_pack
echo "== v6 finished $(date '+%F %T'); send work/v6/locus-v6-results-*.tar and its .sha256 back"
