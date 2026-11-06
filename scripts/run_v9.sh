#!/bin/bash
# v9 (docs/v9/README.md): every experiment of the round in one command, on the main split.
#   Training (Qwen3.5-35B-A3B, the v2 protocol, 3 seeds each):
#     A_cons_dense  A_base's 15000 action exposures plus all 111,634 consequence_v9 records (up to 24 candidates per root)
#     A_set         A_base's 15000 action exposures; each answer lists the whole robust feasible set, then the action
#   Then the v2 test suite on every endpoint and on B0, C1 / C2, the v7 probes on the new checkpoints, and
#   Untrained bases under other prompts (Qwen3.5-35B-A3B and Gemma 4 31B, inference only): a random nonzero output example,
#     an output description without example, and the unchanged prompt with thinking on.
#
#   LOCUS_ROOT=/shared/project NODES=/shared/project/work/v9/nodes.txt \
#     ADDON_TAR=/shared/project/data/locus-bundle-v1.7.0-addon-v9.tar.gz ADDON_SHA256=<sha256> \
#     nohup scripts/run_v9.sh > /dev/null 2>&1 &
#
# Stages: setup, unit, check, train (6 runs, one per node at a time), val, test, confirm, probe, base, pack.
# Finished stages and runs are skipped, so after a stop fix the cause and run the same command again.
set -euo pipefail
export LCAM_ROUND=v9
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
QWEN=$LCAM_MODEL
GEMMA=${GEMMA_MODEL:-$LOCUS_ROOT/models/gemma-4-31B-it}
W=$LOCUS_ROOT/work/v9
ST=$W/stages
ROUNDS="v9"
RUNSPEC=$(for f in A_cons_dense A_set; do for s in 0 1 2; do echo "v9 $f $s"; done; done)   # longest runs first
N_CONS_V9=111634
mkdir -p "$W" "$ST" "$LOCUS_ROOT/work/logs" "$LOCUS_ROOT/work/runs/v9"
exec > >(tee -a "$W/run_v9.log") 2>&1
for r in ${REDO:-}; do rm -f "${ST:?}/${r:?}.ok"; done
fail() { echo "STOPPED: $*"; echo "fix the cause and run the same command again (finished steps are skipped)"; exit 1; }
model_of() { case "$1" in v3_gemma) echo "$GEMMA" ;; *) echo "$QWEN" ;; esac; }
finished() { grep -q '"finished"' "$LOCUS_ROOT/work/runs/$1/run.json" 2>/dev/null; }
echo "== run_v9 $(date '+%F %T')  code $(git rev-parse --short HEAD 2>/dev/null || echo no-git)  qwen $QWEN  gemma $GEMMA"

once() {   # once <stage> <command...>: run a stage unless its marker exists
  local s=$1; shift
  if [ -f "$ST/$s.ok" ]; then echo "-- $s done earlier"; return; fi
  echo "== $s $(date '+%F %T')"
  "$@"
  touch "$ST/$s.ok"
}

do_setup() {
  "$PY" -c "import sys, transformers, peft, torch; from transformers.models.auto.configuration_auto import CONFIG_MAPPING as M; print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__); sys.exit(0 if 'qwen3_5_moe' in M and 'gemma4' in M else 1)" \
    || fail 'this venv cannot load qwen3_5_moe and gemma4 (use the venv of the v7 / v8 runs)'
  "$PY" -c "import json,sys; c=json.load(open('$QWEN/config.json')); sys.exit(0 if c['model_type']=='qwen3_5_moe' else 1)" || fail "$QWEN is not Qwen3.5-35B-A3B"
  "$PY" -c "import json,sys; c=json.load(open('$GEMMA/config.json')); sys.exit(0 if c['model_type']=='gemma4' else 1)" || fail "$GEMMA is not Gemma 4 31B (set GEMMA_MODEL)"
  if [ -n "${ADDON_TAR:-}" ]; then
    : "${ADDON_SHA256:?set ADDON_SHA256 with ADDON_TAR}"
    echo "$ADDON_SHA256  $ADDON_TAR" | sha256sum -c - || fail 'add-on sha256 differs'
    tar -xf "$ADDON_TAR" -C "$BUNDLE" || fail 'could not extract the add-on'
  fi
  [ -f "$BUNDLE/ADDON_V9_MANIFEST.json" ] || fail 'the v9 add-on is not in the bundle (set ADDON_TAR and ADDON_SHA256)'
  for m in "$BUNDLE"/ADDON_*_MANIFEST.json; do
    [ -f "$m" ] || continue
    "$PY" -c "import json,hashlib,sys; b='$BUNDLE'; m=json.load(open('$m')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(b+'/'+p,'rb').read()).hexdigest()!=h]; print('$(basename "$m")', len(m['files']), 'files, bad', bad); sys.exit(1 if bad else 0)" \
      || fail "bundle files differ from $(basename "$m")"
  done
  for f in training_exports/train/action_sft.jsonl training_exports/train/consequence_v9.jsonl training_exports/validation/consequence_v4.jsonl \
           training_exports/validation/action_sft.jsonl eval_prompts/validation_V0.jsonl validation_tables/index.json \
           eval_prompts/test_V0.jsonl eval_prompts/test_variants_V0.jsonl eval_prompts/test_target_swap_V0.jsonl \
           eval_prompts/test_diagnostic_geom_predict.jsonl eval_prompts/test_diagnostic_consequence.jsonl eval_prompts/test_diagnostic_S.jsonl \
           eval_prompts/confirm_c1_direct_choice.jsonl eval_prompts/confirm_c2_reference.jsonl \
           eval_prompts/test_V0_renamed_heldout.jsonl eval_prompts/test_cons_panel.jsonl eval_prompts/validation_cons.jsonl \
           eval_prompts/test_V0_example_random.jsonl eval_prompts/test_V0_schema.jsonl eval_prompts/v9_think/test_V0_part-59.jsonl; do
    [ -f "$BUNDLE/$f" ] || fail "$BUNDLE/$f missing: the bundle of the v8 run plus the v9 add-on is needed"
  done
  "$PY" -c "import json,sys; r=json.loads(open('$BUNDLE/training_exports/train/action_sft.jsonl').readline()); sys.exit(0 if r.get('robust_set') else 1)" \
    || fail 'action_sft records carry no robust_set (A_set lists it)'
  n=$(wc -l < "$BUNDLE/training_exports/train/consequence_v9.jsonl" | tr -d " ")
  [ "$n" = "$N_CONS_V9" ] || fail "consequence_v9 has $n training records, A_cons_dense needs $N_CONS_V9"
  (git rev-parse HEAD 2>/dev/null && git status --short) > "$W/code_commit.txt" || echo 'no git metadata (source snapshot)' > "$W/code_commit.txt"
}

do_unit() {
  "$PY" -m unittest discover -s tests > "$W/unit.log" 2>&1 || { tail -30 "$W/unit.log"; fail 'unit tests failed'; }
  tail -3 "$W/unit.log"
  LCAM_CONFIRM_PROMPTS=$BUNDLE/eval_prompts "$PY" -m unittest tests.test_confirm || fail 'C2 prompt rebuild differs from the frozen prompts'
}

check_one() {   # check_one <name> <args...>: every record of a source encoded with the real processor, as the run builds it
  local name=$1; shift
  LCAM_ROUND=v9 LCAM_MODEL=$QWEN "$PY" scripts/check_data.py --config configs/v9/common.yaml --bundle "$BUNDLE" --split train --limit 0 "$@" > "$W/check_data-$name.json" \
    || fail "data check ($name) crashed"
  "$PY" -c "import json,sys; d=json.load(open('$W/check_data-$name.json')); print('$name', {k:(v['n'], v['n_errors'], v['tokens']['max']) for k,v in d['kinds'].items()}); sys.exit(0 if d['pass'] else 1)" \
    || fail "encoding errors in the $name training data (see $W/check_data-$name.json)"
}

do_check() {
  check_one A_set --kinds action_sft --target-choice robust_set_prefix
  check_one A_cons_dense --kinds action_sft consequence_v9
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

do_train() {    # the 6 runs, one per node at a time
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
  LCAM_ROUND=v9 LCAM_INFER_BATCH=1 "$PY" scripts/make_v2_tasks.py val > "$W/tasks-v9-val.txt"
  pool v9 "$W/tasks-v9-val.txt"
  local r f s
  while read -r r f s; do
    "$PY" scripts/select_checkpoint.py --bundle "$BUNDLE" --run "$LOCUS_ROOT/work/runs/$r/$f-s$s" || fail "checkpoint selection failed for $r/$f-s$s"
  done <<< "$RUNSPEC"
}

do_test() {     # the v2 suite: endpoint checkpoint of every run and B0
  local n want
  LCAM_ROUND=v9 LCAM_INFER_BATCH=1 "$PY" scripts/make_v2_tasks.py test > "$W/tasks-v9-test.txt"
  pool v9 "$W/tasks-v9-test.txt"
  want=$(( ($(grep -c "^v9 " <<< "$RUNSPEC") + 1) * 600 ))
  n=$(cat "$LOCUS_ROOT"/work/outputs/v9/*/test_V0.jsonl | wc -l | tr -d " ")
  [ "$n" = "$want" ] || fail "v9: expected $want test_V0 answers (every run and B0, 600 each), found $n"
  LCAM_ROUND=v9 LCAM_INFER_BATCH=1 "$PY" scripts/make_v2_tasks.py valbest --root "$LOCUS_ROOT" > "$W/tasks-v9-valbest.txt"
  pool v9 "$W/tasks-v9-valbest.txt"
}

do_confirm() {  # C1 / C2 (the frozen settings: batch 1, 2048 tokens); C2 is built from each run's own test_V0 answers
  LCAM_ROUND=v9 "$PY" scripts/make_confirm_tasks.py --root "$LOCUS_ROOT" --bundle "$BUNDLE" > "$LOCUS_ROOT/work/v9/tasks-v9-confirm.txt" || fail 'v9: confirm tasks'
  pool v9 "$LOCUS_ROOT/work/v9/tasks-v9-confirm.txt"
  echo "v9 confirm answers: $(cat "$LOCUS_ROOT"/work/outputs/v9-confirm/*/*.jsonl | wc -l)"
}

do_probe() {    # the v7 probes on the new checkpoints (scripts/make_v7_tasks.py): S, renamed objects, consequence probes for A_cons_dense
  LCAM_ROUND=v9 "$PY" scripts/make_v7_tasks.py --root "$LOCUS_ROOT" > "$W/tasks-v9-probe.txt" 2> "$W/probe-missing.txt" || fail 'probe tasks'
  [ -s "$W/probe-missing.txt" ] && fail "endpoint checkpoints missing: $(cat "$W/probe-missing.txt")"
  pool v9 "$W/tasks-v9-probe.txt"
}

do_base() {     # untrained bases under other prompts: Qwen with the v9 config, Gemma with the v3_gemma config
  local b n
  "$PY" scripts/make_v9_tasks.py --base qwen > "$W/tasks-v9-base-qwen.txt"
  "$PY" scripts/make_v9_tasks.py --base gemma > "$W/tasks-v9-base-gemma.txt"
  pool v9 "$W/tasks-v9-base-qwen.txt"
  pool v3_gemma "$W/tasks-v9-base-gemma.txt"
  for b in qwen gemma; do
    n=$(cat "$LOCUS_ROOT"/work/outputs/v9-base/$b/test_V0_example_random.jsonl "$LOCUS_ROOT"/work/outputs/v9-base/$b/test_V0_schema.jsonl \
            "$LOCUS_ROOT"/work/outputs/v9-base/$b/test_V0_thinking/part-*.jsonl | wc -l | tr -d " ")
    [ "$n" = 1800 ] || fail "v9 base $b: expected 1800 answers (600 per prompt condition), found $n"
  done
}

do_pack() {
  cd "$LOCUS_ROOT"
  local out=work/v9/locus-v9-results-$(date +%m%d).tar list
  : > work/v9/checkpoint_sha256.txt
  sha256sum work/runs/v9/*/checkpoints/ckpt-*/adapter_model.safetensors >> work/v9/checkpoint_sha256.txt
  (cd "$LCAM_CODE" && git diff > "$LOCUS_ROOT/work/v9/code_diff.patch" 2>/dev/null) || true
  list=$(ls -d work/outputs/v9 work/outputs/v9-valbest work/outputs/v9-confirm work/v9/confirm work/outputs/v7/v9 work/outputs/v9-base \
                work/runs/v9/*/run.json work/runs/v9/*/metrics.jsonl work/runs/v9/*/trainable_manifest.json work/runs/v9/*/selection.json \
                work/runs/v9/*/val work/runs/v9/*/checkpoints/*/exposures.json work/logs/v9-*.log \
                work/v9/*.txt work/v9/*.json work/v9/*.log work/v9/code_diff.patch 2>/dev/null || true)
  tar -cf "$out" --exclude='*.safetensors' --exclude='*.pt' $list
  sha256sum "$out" | tee "$out.sha256"
  echo "packed: $(tar -tf "$out" | grep -c 'test_V0.jsonl$') test_V0 files, $(tar -tf "$out" | grep -c '^work/outputs/v7/v9/.*jsonl$') probe files, $(tar -tf "$out" | grep -c '^work/outputs/v9-base/.*jsonl$') base-prompt files"
  cd "$LCAM_CODE"
}

once setup do_setup
once unit do_unit
once check do_check
once train do_train
once val do_val
once test do_test
once confirm do_confirm
once probe do_probe
once base do_base
once pack do_pack
echo "== v9 finished $(date '+%F %T'); send work/v9/locus-v9-results-*.tar and its .sha256 back"
