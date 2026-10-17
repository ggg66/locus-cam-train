#!/usr/bin/env bash
# Full pipeline on the training server, stage by stage. Every stage writes logs to $WORK/logs/<stage>.log and can be
# re-run (finished inference lines and training runs are kept).
#
#   export BUNDLE=/data/locus-bundle WORK=/data/locus-work          # data bundle (unpacked tar) and a work directory
#   scripts/run_pipeline.sh setup      # pip install, download Qwen3.5-35B-A3B, pin the revision in configs/common.yaml
#   scripts/run_pipeline.sh smoke      # tiny random model + mini bundle through every script (minutes, no real weights used)
#   scripts/run_pipeline.sh check      # real processor: lengths, visual tokens, masks (train + validation)
#   scripts/run_pipeline.sh forward    # real model, forward/loss only, no optimizer
#   ALLOW_BACKWARD_PROBE=1 scripts/run_pipeline.sh probe   # one backward pass for peak memory (separate authorisation)
#   scripts/run_pipeline.sh b0         # untrained model on validation + every test prompt file (thinking off), test_V0 with thinking on
#   ALLOW_TRAIN=1 scripts/run_pipeline.sh train            # B1/B2 x $BUDGETS x $SEEDS with optimizer steps (needs authorisation)
#   scripts/run_pipeline.sh select     # every checkpoint on validation, pick by validation success only
#   scripts/run_pipeline.sh test       # selected checkpoint of each run on every test prompt file (once)
#   scripts/run_pipeline.sh pack       # one tar with answers, run records, selections and selected adapters, to send back
#   scripts/run_pipeline.sh all        # setup smoke check forward b0; then train select test pack only if ALLOW_TRAIN=1
#
# Optional: ARMS="B1 B2"  BUDGETS="primary_fixed_total secondary_matched"  SEEDS="0 1 2"  MICRO_BATCH=4  PY=python
#           WORKERS=4  SMOKE_SNAPSHOT=<dir with the model's non-weight files>
#           ENGINE_B0=vllm|hf (default vllm)   ENGINE_ADAPTER=hf|vllm (default hf)   TP=1 (vLLM tensor parallel)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
STAGE="${1:?stage}"
: "${BUNDLE:?set BUNDLE to the unpacked data bundle}"
: "${WORK:?set WORK to a work directory}"
PY="${PY:-python}"
ARMS="${ARMS:-B1 B2}"
BUDGETS="${BUDGETS:-primary_fixed_total}"
SEEDS="${SEEDS:-0}"
MICRO_BATCH="${MICRO_BATCH:-4}"
WORKERS="${WORKERS:-4}"            # data-loading processes (0 on macOS)
ENGINE_B0="${ENGINE_B0:-vllm}"
ENGINE_ADAPTER="${ENGINE_ADAPTER:-hf}"
TP="${TP:-1}"
CFG="$REPO/configs"
mkdir -p "$WORK/logs" "$WORK/runs" "$WORK/outputs"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$WORK/logs/pipeline.log"; }
run() { log "$*"; "$@"; }

infer() {  # infer <engine> <config dir> <prompt file in bundle> <out> [extra args]
  local engine=$1 cfgdir=$2 prompts=$3 out=$4; shift 4
  if [ "$engine" = vllm ]; then
    run $PY scripts/infer_vllm.py --config "$cfgdir/common.yaml" --bundle "$BUNDLE" --prompts "$prompts" --out "$out" --tensor-parallel "$TP" "$@"
  else
    run $PY scripts/infer.py --config "$cfgdir/common.yaml" --bundle "$BUNDLE" --prompts "$prompts" --out "$out" "$@"
  fi
}

test_prompt_files() { (cd "$BUNDLE" && ls eval_prompts/test_*.jsonl); }

stage_setup() {
  run $PY -m pip install -r requirements.txt
  $PY -m pip install flash-linear-attention causal-conv1d || log "WARNING: fast Gated DeltaNet kernels not installed (training will be slow)"
  local rev
  rev=$($PY - <<'EOF'
from huggingface_hub import snapshot_download, HfApi
sha = HfApi().model_info('Qwen/Qwen3.5-35B-A3B').sha
snapshot_download('Qwen/Qwen3.5-35B-A3B', revision=sha)
print(sha)
EOF
)
  log "downloaded Qwen/Qwen3.5-35B-A3B revision $rev"
  $PY - "$rev" <<'EOF'
import sys, re
rev = sys.argv[1]
p = 'configs/common.yaml'; s = open(p).read()
s = re.sub(r'(\n  revision: )\S+', r'\g<1>' + rev, s, count=1)
open(p, 'w').write(s)
EOF
  grep -n "revision:" configs/common.yaml | head -1
  run $PY -m unittest discover -s tests
}

stage_smoke() {
  local S="$WORK/smoke" snap
  snap="${SMOKE_SNAPSHOT:-$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-35B-A3B', allow_patterns=['*.json','*.jinja','*.txt']))")}"
  rm -rf "$S"; mkdir -p "$S"
  run $PY tests/mock/make_tiny_model.py "$snap" "$S/tiny-model"
  run $PY tests/mock/make_mini_bundle.py "$BUNDLE" "$S/mini-bundle"
  run $PY tests/mock/make_smoke_config.py "$S/tiny-model" "$S/configs"
  local B="$S/mini-bundle" C="$S/configs"
  run $PY scripts/check_data.py --config "$C/common.yaml" --bundle "$B"
  run $PY scripts/forward_check.py --config "$C/common.yaml" --bundle "$B" --batches 2 --micro-batch 2
  LCAM_ALLOW_BACKWARD_PROBE=1 run $PY scripts/forward_check.py --config "$C/common.yaml" --bundle "$B" --batches 1 --micro-batch 2 --backward-probe
  for arm in B1 B2; do
    run $PY scripts/train.py --config "$C/$arm.yaml" --budget primary_fixed_total --seed 0 --bundle "$B" --out "$S/runs/$arm" --micro-batch 2 --workers "$WORKERS" --allow-optimizer-step
    for ck in "$S/runs/$arm"/checkpoints/ckpt-*; do
      run $PY scripts/infer.py --config "$C/common.yaml" --bundle "$B" --prompts eval_prompts/validation_V0.jsonl --out "$S/runs/$arm/val/$(basename "$ck").jsonl" --adapter "$ck" --max-new-tokens 8
    done
    run $PY scripts/select_checkpoint.py --bundle "$B" --run "$S/runs/$arm" --expect "$(wc -l < "$B/eval_prompts/validation_V0.jsonl")"
  done
  for f in test_V0 test_diagnostic_no_image test_diagnostic_geom_predict; do
    run $PY scripts/infer.py --config "$C/common.yaml" --bundle "$B" --prompts "eval_prompts/$f.jsonl" --out "$S/b0/$f.jsonl" --max-new-tokens 8
  done
  run $PY tests/mock/overfit_check.py --config "$C" --work "$S/overfit"
  log "SMOKE PASSED"
}

stage_check() {
  for split in train validation; do run $PY scripts/check_data.py --config "$CFG/common.yaml" --bundle "$BUNDLE" --split "$split"; done
}

stage_forward() { run $PY scripts/forward_check.py --config "$CFG/common.yaml" --bundle "$BUNDLE" --batches 4 --micro-batch "$MICRO_BATCH"; }

stage_probe() {
  [ "${ALLOW_BACKWARD_PROBE:-}" = 1 ] || { log "probe needs ALLOW_BACKWARD_PROBE=1"; exit 2; }
  LCAM_ALLOW_BACKWARD_PROBE=1 run $PY scripts/forward_check.py --config "$CFG/common.yaml" --bundle "$BUNDLE" --batches 2 --micro-batch "$MICRO_BATCH" --backward-probe
}

stage_b0() {
  infer "$ENGINE_B0" "$CFG" eval_prompts/validation_V0.jsonl "$WORK/outputs/B0-nothink/validation_V0.jsonl"
  for f in $(test_prompt_files); do infer "$ENGINE_B0" "$CFG" "$f" "$WORK/outputs/B0-nothink/$(basename "$f")"; done
  if [ "$ENGINE_B0" = vllm ]; then
    infer vllm "$CFG" eval_prompts/test_V0.jsonl "$WORK/outputs/B0-think/test_V0.jsonl" --thinking
  else
    infer hf "$CFG" eval_prompts/test_V0.jsonl "$WORK/outputs/B0-think/test_V0.jsonl" --thinking --max-new-tokens 32768 --batch 2
  fi
}

stage_train() {
  [ "${ALLOW_TRAIN:-}" = 1 ] || { log "training needs ALLOW_TRAIN=1 (optimizer steps update the model)"; exit 2; }
  for arm in $ARMS; do for budget in $BUDGETS; do for seed in $SEEDS; do
    local out="$WORK/runs/$arm-$budget-s$seed"
    if [ -f "$out/run.json" ] && grep -q '"finished"' "$out/run.json"; then log "skip finished $out"; continue; fi
    run $PY scripts/train.py --config "$CFG/$arm.yaml" --budget "$budget" --seed "$seed" --bundle "$BUNDLE" --out "$out" --micro-batch "$MICRO_BATCH" --workers "$WORKERS" --allow-optimizer-step
  done; done; done
}

stage_select() {
  for out in "$WORK"/runs/*/; do
    out=${out%/}
    [ -d "$out/checkpoints" ] || continue
    for ck in "$out"/checkpoints/ckpt-*; do
      infer "$ENGINE_ADAPTER" "$CFG" eval_prompts/validation_V0.jsonl "$out/val/$(basename "$ck").jsonl" --adapter "$ck"
    done
    run $PY scripts/select_checkpoint.py --bundle "$BUNDLE" --run "$out"
  done
}

stage_test() {
  for out in "$WORK"/runs/*/; do
    out=${out%/}
    [ -f "$out/selection.json" ] || { log "no selection for $out (run the select stage)"; continue; }
    local ck
    ck=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['selected'])" "$out/selection.json")
    for f in $(test_prompt_files); do
      infer "$ENGINE_ADAPTER" "$CFG" "$f" "$WORK/outputs/$(basename "$out")/$(basename "$f")" --adapter "$out/checkpoints/$ck"
    done
  done
}

stage_pack() {
  local tarf="$WORK/locus-results-$(date +%Y%m%d-%H%M).tar"
  (cd "$WORK" && {
    find outputs -name '*.jsonl'
    find runs -maxdepth 2 \( -name run.json -o -name metrics.jsonl -o -name selection.json \)
    find runs -path '*/val/*.jsonl'
    for s in runs/*/selection.json; do [ -f "$s" ] && echo "$(dirname "$s")/checkpoints/$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['selected'])" "$s")"; done
    find logs -type f
  } | tar -cf "$tarf" -T -)
  [ -d "$BUNDLE/checks" ] && tar -rf "$tarf" -C "$BUNDLE" checks
  cp "$REPO/configs/common.yaml" "$WORK/common.yaml.used" && tar -rf "$tarf" -C "$WORK" common.yaml.used
  log "packed $tarf ($(du -h "$tarf" | cut -f1)); send it back for scoring on the benchmark machine"
}

case "$STAGE" in
  setup|smoke|check|forward|probe|b0|train|select|test|pack) "stage_$STAGE" 2>&1 | tee -a "$WORK/logs/$STAGE.log" ;;
  all)
    for s in setup smoke check forward b0; do "stage_$s" 2>&1 | tee -a "$WORK/logs/$s.log"; done
    if [ "${ALLOW_TRAIN:-}" = 1 ]; then
      for s in train select test pack; do "stage_$s" 2>&1 | tee -a "$WORK/logs/$s.log"; done
    else
      log "stopping before training: set ALLOW_TRAIN=1 and run 'train', 'select', 'test', 'pack' when authorised"
    fi ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
