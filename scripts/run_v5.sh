#!/bin/bash
# v5 method development (docs/v5/README.md): one method per call, on the development split (frozen test set untouched).
#
#   LOCUS_ROOT=/shared/project ADDON_TAR=.../locus-bundle-v1.7.0-addon-v5.tar ADDON_SHA256=<sha> scripts/run_v5.sh m1
#   then  scripts/run_v5.sh m2,  m3,  m4      (in this order; each call is independent of the later ones)
#
#   m1  RL with execution rewards: GRPO from the development baseline (configs/v5/m1_rl.yaml)
#   m2  imagine, then act: SFT (m2_imagact.yaml), then GRPO with an imagination reward (m2_rl.yaml)
#   m3  inverse imagination as an auxiliary source (m3_inverse.yaml)
#   m4  multi-solution action targets (m4_multi.yaml)
#
# Shared stages run once, at the first call: setup (add-on, files, model type), unit, check (real processor on the new
# sources), smoke (tiny random model through SFT, GRPO, inference and scoring on one GPU), base (the development
# baseline = A_base without the development families; every method is compared with it). NODES (a file with one IP per line)
# lists the free nodes, default this node only: the seeds of a run go to different nodes in parallel (8 GPUs each), and
# evaluation shares all of them. SEEDS (default "0 1 2") lists the seeds: differences between seeds of one method reach 8
# points, so one seed cannot tell methods apart. Base and method 2 are also evaluated at the checkpoint RL starts from.
# Finished runs and evaluations are skipped, so after a stop fix the cause and run the same command again.
# Results: $LOCUS_ROOT/work/v5/dev_summary.md (held-out development families and seen families, each method vs base).
set -euo pipefail
METHOD=${1:?give the method: m1 m2 m3 m4}
case "$METHOD" in m1|m2|m3|m4) ;; *) echo "unknown method $METHOD"; exit 1 ;; esac
export LCAM_ROUND=v5
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
W=$LOCUS_ROOT/work/v5
ST=$W/stages
RUNS=$LOCUS_ROOT/work/runs/v5
OUTS=work/outputs/v5
SEEDS=${SEEDS:-0 1 2}
mkdir -p "$W" "$ST" "$RUNS" "$LOCUS_ROOT/work/logs"
exec > >(tee -a "$W/run_$METHOD.log") 2>&1
for r in ${REDO:-}; do rm -f "${ST:?}/${r:?}.ok"; done
fail() { echo "STOPPED ($METHOD): $*"; echo "fix the cause and run the same command again (finished steps are skipped)"; exit 1; }
finished() { grep -q '"finished"' "$RUNS/$1/run.json" 2>/dev/null; }
last_ckpt() { ls -d "$RUNS/$1/checkpoints"/ckpt-* 2>/dev/null | sort -t- -k2 -n | tail -1; }
echo "== run_v5 $METHOD $(date '+%F %T')  code $(git rev-parse --short HEAD 2>/dev/null || echo no-git)  seeds $SEEDS  model $LCAM_MODEL"

once() {   # once <stage> <command...>: run a shared stage unless its marker exists
  local s=$1; shift
  if [ -f "$ST/$s.ok" ]; then echo "-- $s done earlier"; return; fi
  echo "== $s $(date '+%F %T')"
  "$@"
  touch "$ST/$s.ok"
}

do_setup() {
  "$PY" -c "import sys, transformers, peft, torch; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__); sys.exit(0 if 'qwen3_5_moe' in CONFIG_MAPPING else 1)" \
    || fail 'this venv cannot load qwen3_5_moe (use the venv of the v2 runs)'
  "$PY" -c "import json,sys; c=json.load(open('$LCAM_MODEL/config.json')); sys.exit(0 if c['model_type']=='qwen3_5_moe' else 1)" || fail "$LCAM_MODEL is not Qwen3.5-35B-A3B"
  if [ -n "${ADDON_TAR:-}" ]; then
    : "${ADDON_SHA256:?set ADDON_SHA256 with ADDON_TAR}"
    echo "$ADDON_SHA256  $ADDON_TAR" | sha256sum -c - || fail 'add-on sha256 differs'
    tar -xf "$ADDON_TAR" -C "$BUNDLE" || fail 'could not extract the add-on'     # its own files only: a newer version replaces an older one
    echo "$ADDON_SHA256" > "$W/addon.sha256"
  fi
  for m in "$BUNDLE"/ADDON_*_MANIFEST.json; do
    [ -f "$m" ] || continue
    "$PY" -c "import json,hashlib,sys; b='$BUNDLE'; m=json.load(open('$m')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(b+'/'+p,'rb').read()).hexdigest()!=h]; print('$(basename "$m")', len(m['files']), 'files, bad', bad); sys.exit(1 if bad else 0)" \
      || fail "bundle files differ from $(basename "$m")"
  done
  for f in training_exports/train/action_sft.jsonl training_exports/train/imagact_v5.jsonl training_exports/train/inverse_v5.jsonl \
           training_exports/validation/action_sft.jsonl training_exports/validation/imagact_v5.jsonl training_exports/validation/inverse_v5.jsonl \
           reward_tables/train.jsonl.gz eval_prompts/dev_heldfam_V0.jsonl eval_prompts/dev_seen_V0.jsonl validation_tables/index.json; do
    [ -f "$BUNDLE/$f" ] || fail "$BUNDLE/$f missing: extract locus-bundle-v1.7.0-addon-v5.tar (ADDON_TAR / ADDON_SHA256)"
  done
  (git rev-parse HEAD 2>/dev/null && git status --short) > "$W/code_commit.txt" || echo 'no git metadata (source snapshot)' > "$W/code_commit.txt"
}

do_unit() {
  "$PY" -m unittest discover -s tests > "$W/unit.log" 2>&1 || { tail -30 "$W/unit.log"; fail 'unit tests failed'; }
  tail -3 "$W/unit.log"
}

do_check() {
  for s in train validation; do
    lim=0; [ "$s" = train ] && lim=3000
    "$PY" scripts/check_data.py --config configs/v5/common.yaml --bundle "$BUNDLE" --split $s --kinds action_sft imagact_v5 inverse_v5 --limit $lim > "$W/check_data-$s.json" \
      || fail "data check ($s) crashed"
    "$PY" -c "import json,sys; d=json.load(open('$W/check_data-$s.json')); print('$s', {k:(v['n_errors'], v['tokens']['max']) for k,v in d['kinds'].items()}); sys.exit(0 if d['pass'] else 1)" \
      || fail "encoding errors in the $s data (see $W/check_data-$s.json)"
  done
}

do_smoke() {
  local S=$W/smoke T
  rm -rf "${S:?}"; mkdir -p "$S"
  T=(--micro-batch 1 --workers 0 --allow-optimizer-step)
  (
    set -e
    export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}
    "$PY" tests/mock/make_tiny_model.py "${SMOKE_SNAPSHOT:-$LCAM_MODEL}" "$S/tiny-model"
    "$PY" tests/mock/make_mini_bundle.py "$BUNDLE" "$S/mini-bundle"
    "$PY" tests/mock/make_smoke_config.py "$S/tiny-model" "$S/configs"
    C=$S/configs/v5; B=$S/mini-bundle
    # the mini bundle has a dozen training roots; the development families are not excluded there (unit tests cover it)
    "$PY" -c "import yaml; p='$C/common.yaml'; c=yaml.safe_load(open(p)); c['data']['exclude_families']=[]; yaml.safe_dump(c, open(p,'w'), allow_unicode=True, sort_keys=False)"
    for f in base m4_multi m3_inverse m2_imagact; do
      "$PY" scripts/train_v2.py --config "$C/$f.yaml" --seed 0 --bundle "$B" --out "$S/runs/$f" "${T[@]}"
    done
    for pair in m1_rl:base m2_rl:m2_imagact; do
      f=${pair%%:*}; init=$(ls -d "$S/runs/${pair##*:}/checkpoints"/ckpt-* | sort -t- -k2 -n | tail -1)
      "$PY" scripts/train_grpo.py --config "$C/$f.yaml" --seed 0 --bundle "$B" --out "$S/runs/$f" --init-adapter "$init" --max-steps 2 --smoke
    done
    for f in base m1_rl m2_rl; do
      ck=$(ls -d "$S/runs/$f/checkpoints"/ckpt-* | sort -t- -k2 -n | tail -1)
      for p in dev_heldfam dev_seen; do
        "$PY" scripts/infer.py --config "$C/common.yaml" --bundle "$B" --prompts eval_prompts/${p}_V0.jsonl --out "$S/out/$f-s0/$p.jsonl" --adapter "$ck" --max-new-tokens 8 --batch 2
      done
    done
    "$PY" scripts/score_dev.py --bundle "$B" --outputs "$S/out" --out "$S/dev_summary.json"
  ) > "$S/smoke.log" 2>&1 || { tail -40 "$S/smoke.log"; fail "smoke test failed ($S/smoke.log)"; }
  tail -6 "$S/smoke.log"
  echo "smoke passed"
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

run_seeds() {   # run_seeds <run family> <launcher: launch_one <seed> <ip>>: one run per node at a time
  local f=$1 launcher=$2 s p n=0 failed=""; local -a nodes todo pids     # pids: a plain `wait` would also wait for the log's tee
  mapfile -t nodes < <(node_list)
  for s in $SEEDS; do
    if finished "$f-s$s"; then echo "-- $f-s$s finished earlier"; else todo+=("$s"); fi
  done
  for s in ${todo[@]+"${todo[@]}"}; do
    echo "== train $f-s$s on ${nodes[n]} $(date '+%F %T')"
    "$launcher" "$s" "${nodes[n]}" &
    pids+=($!)
    n=$((n+1))
    if [ "$n" = "${#nodes[@]}" ]; then
      for p in "${pids[@]}"; do wait "$p" || true; done
      pids=(); n=0
    fi
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  for s in ${todo[@]+"${todo[@]}"}; do finished "$f-s$s" || failed="$failed $f-s$s"; done
  [ -z "$failed" ] || fail "training ended without finishing:$failed (work/logs/v5-<run>.log)"
}

train_sft() {   # train_sft <config name>
  SFT_F=$1
  run_seeds "$1" one_sft
}
one_sft() { on_node "$2" "$LOCUS_ROOT/work/logs/v5-$SFT_F-s$1.log" launch/launch_run_v2.sh "configs/v5/$SFT_F.yaml" "$1" 29640; }

train_rl() {    # train_rl <config name> <init run family>: starts from the checkpoint named by init_checkpoint in the config
  local s
  RL_F=$1; RL_INIT=$2
  RL_CK=$("$PY" -c "import sys; sys.path.insert(0, '.'); from lcam import config as C; print(C.load('configs/v5/$1.yaml')['init_checkpoint'])")
  for s in $SEEDS; do
    finished "$1-s$s" || [ -d "$RUNS/$2-s$s/checkpoints/$RL_CK" ] || fail "no $RL_CK of $2-s$s to start $1 from"
  done
  run_seeds "$1" one_rl
}
one_rl() { on_node "$2" "$LOCUS_ROOT/work/logs/v5-$RL_F-s$1.log" launch/launch_grpo.sh "configs/v5/$RL_F.yaml" "$1" "$RUNS/$RL_INIT-s$1/checkpoints/$RL_CK" 29740; }

shards() {      # development prompts cut into shards, one inference job each
  local p=$1 n=$2
  mkdir -p "$W/prompts"
  [ -f "$W/prompts/$p-0.jsonl" ] && return
  "$PY" -c "
import sys
lines = open('$BUNDLE/eval_prompts/${p}_V0.jsonl', encoding='utf-8').read().splitlines()
for k in range($n):
    open('$W/prompts/$p-%d.jsonl' % k, 'w', encoding='utf-8').write(''.join(l + '\n' for l in lines[k::$n]))"
}

evaluate() {    # evaluate <run family[@checkpoint]...>: final (or the named) checkpoint of every seed on both prompt sets
  local tasks=$W/tasks-eval-$METHOD.txt item f at label s ck k
  shards dev_heldfam 8; shards dev_seen 4
  : > "$tasks"
  for item in "$@"; do
    f=${item%@*}; at=""; label=$f
    if [ "$item" != "$f" ]; then at=${item#*@}; label=${f}_at_$at; fi
    for s in $SEEDS; do
      if [ -n "$at" ]; then ck=$RUNS/$f-s$s/checkpoints/$at; [ -d "$ck" ] || fail "no $at of $f-s$s"
      else ck=$(last_ckpt "$f-s$s"); [ -n "$ck" ] || fail "no checkpoint of $f-s$s"; fi
      mx=512; case $f in m2_*) mx=1024 ;; esac          # method 2 writes four imagined candidates before the move
      for k in 0 1 2 3 4 5 6 7; do printf '%s\t%s\t%s\t0\t1\t%s\n' "$OUTS/$label-s$s/dev_heldfam.part$k.jsonl" "$W/prompts/dev_heldfam-$k.jsonl" "${ck#$LOCUS_ROOT/}" "$mx" >> "$tasks"; done
      for k in 0 1 2 3; do printf '%s\t%s\t%s\t0\t1\t%s\n' "$OUTS/$label-s$s/dev_seen.part$k.jsonl" "$W/prompts/dev_seen-$k.jsonl" "${ck#$LOCUS_ROOT/}" "$mx" >> "$tasks"; done
    done
  done
  local left=0
  while IFS=$'\t' read -r out _; do [ -e "$LOCUS_ROOT/$out.done" ] || left=$((left+1)); done < "$tasks"
  [ "$left" = 0 ] && { echo "-- evaluation of $* done earlier"; return; }
  local nodes=${NODES:-}
  if [ -z "$nodes" ]; then nodes=$W/nodes-self.txt; echo "$SELF_IP" > "$nodes"; fi
  rm -rf "$LOCUS_ROOT/work/claims/$(basename "$tasks" .txt)"
  local flog=$LOCUS_ROOT/work/logs/infer/failures.log f0=0            # only failures of this attempt count
  [ -f "$flog" ] && f0=$(wc -l < "$flog")
  launch/run_infer_pool.sh "$tasks" "$nodes"
  while true; do
    left=0
    while IFS=$'\t' read -r out _; do
      [ -e "$LOCUS_ROOT/$out.done" ] && continue
      left=$((left+1))
      [ -f "$flog" ] && tail -n +$((f0+1)) "$flog" | grep -qF "FAILED $out " && fail "inference failed: $out (work/logs/infer/failures.log)"
    done < "$tasks"
    [ "$left" = 0 ] && break
    echo "$(date +%T) $left evaluation jobs left"; sleep 60
  done
}

score() {
  "$PY" scripts/score_dev.py --bundle "$BUNDLE" --outputs "$LOCUS_ROOT/$OUTS" --out "$W/dev_summary.json" || fail 'scoring failed'
}

# a new add-on must be extracted even if an earlier one already passed setup
if [ -n "${ADDON_SHA256:-}" ] && [ "$ADDON_SHA256" != "$(cat "$W/addon.sha256" 2>/dev/null)" ]; then rm -f "${ST:?}/setup.ok"; fi
once setup do_setup
once unit do_unit
once check do_check
once smoke do_smoke
train_sft base
evaluate base
case $METHOD in
  m1) evaluate base@ckpt-1; train_rl m1_rl base; evaluate m1_rl ;;          # base@ckpt-1: what RL starts from (an earlier stop on its own)
  m2) train_sft m2_imagact; evaluate m2_imagact m2_imagact@ckpt-1; train_rl m2_rl m2_imagact; evaluate m2_rl ;;
  m3) train_sft m3_inverse; evaluate m3_inverse ;;
  m4) train_sft m4_multi; evaluate m4_multi ;;
esac
score
echo "== $METHOD finished $(date '+%F %T'); results in $W/dev_summary.md"
