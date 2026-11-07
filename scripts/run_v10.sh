#!/bin/bash
# v10 (docs/v10/README.md): perceive, then compute. Every step of the round in one command, on the main split.
#   Training (Qwen3.5-35B-A3B, the v2 protocol, 3 seeds each), geometry answers only:
#     G_rand  5,053 geometry_rand_v10 records: random-structure scenes (targets and obstacles at random, training nouns only)
#     G_fam   5,053 geometry_sft_v2 records: the ten training families (the geometry source of v2 AG)
#     15,000 exposures each, the same prompt, answer format and hidden-field defaults; only the scene structures differ.
#   Then the geometry prompts of all 600 test tasks and the 120 diagnostic scenes on every endpoint; the fixed solver
#   acts on the predicted geometry when the results are scored (benchmark side).
#
#   LOCUS_ROOT=/shared/project NODES=/shared/project/work/v10/nodes.txt \
#     ADDON_TAR=/shared/project/data/locus-bundle-v1.7.0-addon-v10.tar ADDON_SHA256=<sha256> \
#     nohup scripts/run_v10.sh > /dev/null 2>&1 &
#
# Stages: setup, unit, check, train (6 runs, one per node at a time), test, pack.
# Finished stages and runs are skipped, so after a stop fix the cause and run the same command again.
set -euo pipefail
export LCAM_ROUND=v10
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
W=$LOCUS_ROOT/work/v10
ST=$W/stages
RUNSPEC=$(for f in G_rand G_fam; do for s in 0 1 2; do echo "v10 $f $s"; done; done)
N_RAND=5053
mkdir -p "$W" "$ST" "$LOCUS_ROOT/work/logs" "$LOCUS_ROOT/work/runs/v10"
exec > >(tee -a "$W/run_v10.log") 2>&1
for r in ${REDO:-}; do rm -f "${ST:?}/${r:?}.ok"; done
fail() { echo "STOPPED: $*"; echo "fix the cause and run the same command again (finished steps are skipped)"; exit 1; }
finished() { grep -q '"finished"' "$LOCUS_ROOT/work/runs/$1/run.json" 2>/dev/null; }
echo "== run_v10 $(date '+%F %T')  code $(git rev-parse --short HEAD 2>/dev/null || echo no-git)  model $LCAM_MODEL"

once() {   # once <stage> <command...>: run a stage unless its marker exists
  local s=$1; shift
  if [ -f "$ST/$s.ok" ]; then echo "-- $s done earlier"; return; fi
  echo "== $s $(date '+%F %T')"
  "$@"
  touch "$ST/$s.ok"
}

do_setup() {
  "$PY" -c "import sys, transformers, peft, torch; from transformers.models.auto.configuration_auto import CONFIG_MAPPING as M; print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__); sys.exit(0 if 'qwen3_5_moe' in M else 1)" \
    || fail 'this venv cannot load qwen3_5_moe (use the venv of the v9 run)'
  "$PY" -c "import json,sys; c=json.load(open('$LCAM_MODEL/config.json')); sys.exit(0 if c['model_type']=='qwen3_5_moe' else 1)" || fail "$LCAM_MODEL is not Qwen3.5-35B-A3B"
  if [ -n "${ADDON_TAR:-}" ]; then
    : "${ADDON_SHA256:?set ADDON_SHA256 with ADDON_TAR}"
    echo "$ADDON_SHA256  $ADDON_TAR" | sha256sum -c - || fail 'add-on sha256 differs'
    tar -xf "$ADDON_TAR" -C "$BUNDLE" || fail 'could not extract the add-on'
  fi
  [ -f "$BUNDLE/ADDON_V10_MANIFEST.json" ] || fail 'the v10 add-on is not in the bundle (set ADDON_TAR and ADDON_SHA256)'
  for m in "$BUNDLE"/ADDON_*_MANIFEST.json; do
    [ -f "$m" ] || continue
    "$PY" -c "import json,hashlib,sys; b='$BUNDLE'; m=json.load(open('$m')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(b+'/'+p,'rb').read()).hexdigest()!=h]; print('$(basename "$m")', len(m['files']), 'files, bad', bad[:5]); sys.exit(1 if bad else 0)" \
      || fail "bundle files differ from $(basename "$m")"
  done
  for f in training_exports/train/geometry_rand_v10.jsonl training_exports/train/geometry_sft_v2.jsonl training_exports/validation/geometry_sft_v2.jsonl \
           eval_prompts/test_geom_predict.jsonl eval_prompts/test_diagnostic_geom_predict.jsonl splits_index.json; do
    [ -f "$BUNDLE/$f" ] || fail "$BUNDLE/$f missing: the bundle of the v9 run plus the v10 add-on is needed"
  done
  n=$(wc -l < "$BUNDLE/training_exports/train/geometry_rand_v10.jsonl" | tr -d " ")
  [ "$n" = "$N_RAND" ] || fail "geometry_rand_v10 has $n records, G_rand needs $N_RAND"
  n=$(wc -l < "$BUNDLE/training_exports/train/geometry_sft_v2.jsonl" | tr -d " ")
  [ "$n" = "$N_RAND" ] || fail "geometry_sft_v2 has $n records, G_fam needs $N_RAND"
  "$PY" -c "import json,os,sys; b='$BUNDLE'; bad=[i for l in open(b+'/training_exports/train/geometry_rand_v10.jsonl') for i in json.loads(l)['images'] if not os.path.exists(b+'/'+i)]; print('random-scene images missing:', len(bad)); sys.exit(1 if bad else 0)" \
    || fail 'random-scene images are missing from the bundle'
  (git rev-parse HEAD 2>/dev/null && git status --short) > "$W/code_commit.txt" || echo 'no git metadata (source snapshot)' > "$W/code_commit.txt"
}

do_unit() {
  "$PY" -m unittest discover -s tests > "$W/unit.log" 2>&1 || { tail -30 "$W/unit.log"; fail 'unit tests failed'; }
  tail -3 "$W/unit.log"
}

check_one() {   # check_one <name> <kind>: every record of a source encoded with the real processor, as the run builds it
  local name=$1 kind=$2
  "$PY" scripts/check_data.py --config configs/v10/common.yaml --bundle "$BUNDLE" --split train --limit 0 --kinds "$kind" > "$W/check_data-$name.json" \
    || fail "data check ($name) crashed"
  "$PY" -c "import json,sys; d=json.load(open('$W/check_data-$name.json')); print('$name', {k:(v['n'], v['n_errors'], v['tokens']['max']) for k,v in d['kinds'].items()}); sys.exit(0 if d['pass'] else 1)" \
    || fail "encoding errors in the $name training data (see $W/check_data-$name.json)"
}

do_check() {
  check_one G_rand geometry_rand_v10
  check_one G_fam geometry_sft_v2
}

node_list() {
  if [ -n "${NODES:-}" ]; then grep -v '^#' "$NODES" | awk 'NF {print $1}'; else echo "$SELF_IP"; fi
}

one_run() {     # one_run <round> <family> <seed> <ip>: one training run on the 8 GPUs of a node; returns when it ends
  local r=$1 f=$2 s=$3 ip=$4 log cmd
  log=$LOCUS_ROOT/work/logs/$r-$f-s$s.log
  if [ "$ip" = "$SELF_IP" ]; then LCAM_ROUND=$r launch/launch_run_v2.sh "configs/$r/$f.yaml" "$s" 29640 > "$log" 2>&1; return; fi
  printf -v cmd 'cd %q && LOCUS_ROOT=%q LCAM_ROUND=%q LCAM_MODEL=%q LCAM_CODE=%q BUNDLE=%q PY=%q SELF_IP=%q launch/launch_run_v2.sh %q %q 29640 > %q 2>&1' \
    "$LCAM_CODE" "$LOCUS_ROOT" "$r" "$LCAM_MODEL" "$LCAM_CODE" "$BUNDLE" "$PY" "$ip" "configs/$r/$f.yaml" "$s" "$log"
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

pool() {        # pool <task file>: run it over the nodes until every job is done
  local tasks=$1 nodes=${NODES:-} left out
  [ -s "$tasks" ] || return 0
  if [ -z "$nodes" ]; then nodes=$W/nodes-self.txt; echo "$SELF_IP" > "$nodes"; fi
  rm -rf "$LOCUS_ROOT/work/claims/$(basename "$tasks" .txt)"
  local flog=$LOCUS_ROOT/work/logs/infer/failures.log f0=0            # only failures of this attempt count
  [ -f "$flog" ] && f0=$(wc -l < "$flog" | tr -d " ")
  launch/run_infer_pool.sh "$tasks" "$nodes"
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
    echo "$(date +%T) v10: $left jobs left in $(basename "$tasks")"; sleep 60
  done
}

do_test() {     # geometry prompts of the 600 test tasks and the 120 diagnostic scenes, endpoint checkpoint of every run
  local n
  "$PY" scripts/make_v10_tasks.py --root "$LOCUS_ROOT" > "$W/tasks-v10-test.txt" 2> "$W/test-missing.txt" || fail 'test tasks'
  [ -s "$W/test-missing.txt" ] && fail "endpoint checkpoints missing: $(cat "$W/test-missing.txt")"
  pool "$W/tasks-v10-test.txt"
  n=$(cat "$LOCUS_ROOT"/work/outputs/v10/*/test_geom_predict.jsonl | wc -l | tr -d " ")
  [ "$n" = 3600 ] || fail "expected 3600 test_geom_predict answers (6 runs x 600), found $n"
}

do_pack() {
  cd "$LOCUS_ROOT"
  local out=work/v10/locus-v10-results-$(date +%m%d).tar list
  : > work/v10/checkpoint_sha256.txt
  sha256sum work/runs/v10/*/checkpoints/ckpt-*/adapter_model.safetensors >> work/v10/checkpoint_sha256.txt
  (cd "$LCAM_CODE" && git diff > "$LOCUS_ROOT/work/v10/code_diff.patch" 2>/dev/null) || true
  list=$(ls -d work/outputs/v10 work/runs/v10/*/run.json work/runs/v10/*/metrics.jsonl work/runs/v10/*/trainable_manifest.json \
                work/runs/v10/*/checkpoints/*/exposures.json work/logs/v10-*.log \
                work/v10/*.txt work/v10/*.json work/v10/*.log work/v10/code_diff.patch 2>/dev/null || true)
  tar -cf "$out" --exclude='*.safetensors' --exclude='*.pt' $list
  sha256sum "$out" | tee "$out.sha256"
  echo "packed: $(tar -tf "$out" | grep -c 'test_geom_predict.jsonl$') test_geom_predict files"
  cd "$LCAM_CODE"
}

once setup do_setup
once unit do_unit
once check do_check
once train do_train
once test do_test
once pack do_pack
echo "== v10 finished $(date '+%F %T'); send work/v10/locus-v10-results-*.tar and its .sha256 back"
