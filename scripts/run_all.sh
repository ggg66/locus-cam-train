#!/bin/bash
# Every experiment of one training round in one command: setup, all gates, the training runs, validation, the test
# suite (incl. B0 = the untrained base, zero-shot), the C1/C2 inference controls, and one result tar.
#
#   LOCUS_ROOT=/shared/project NODES=/shared/project/work/<round>/nodes.txt scripts/run_all.sh <round>
#   rounds: v3_gemma (Gemma 4 31B, docs/v3_gemma/README.md), v4 (supervision study on the v2 base, docs/v4/README.md)
#
# NODES: one node IP per line (8 nodes x 8 GPUs); the node running this script must be one of them.
# ADDON_TAR / ADDON_SHA256 (optional): a bundle add-on to check and extract first (v4: locus-bundle-v1.7.0-addon-v4.tar).
# Every stage writes work/<round>/stages/<stage>.ok when its checks pass; a rerun skips finished stages, so after a
# stop (a failed check, a dead node) fix the cause and run the same command again. REDO="stage ..." reruns stages.
# A failed check stops everything before training starts: the gates are the v2 gates with their pass criteria checked
# here instead of by hand. The compute calibration only needs token statistics, so it is frozen right after them.
# Stages: setup unit smoke check stats freeze audit equiv batchcheck short train val test confirm pack
set -euo pipefail
export LCAM_ROUND=${1:-${LCAM_ROUND:?give the round: scripts/run_all.sh v3_gemma|v4}}
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
[ -f "configs/$LCAM_ROUND/common.yaml" ] || { echo "no configs/$LCAM_ROUND/"; exit 1; }
: "${NODES:?set NODES to a file with one node IP per line}"
case "$NODES" in /*) ;; *) NODES=$LOCUS_ROOT/$NODES ;; esac
export NODES
W=$LOCUS_ROOT/work/$LCAM_ROUND
A=$W/audit
ST=$W/stages
mkdir -p "$W" "$A" "$ST" "$LOCUS_ROOT/work/logs"
exec > >(tee -a "$W/run_all.log") 2>&1
RI() { "$PY" scripts/round_info.py "configs/$LCAM_ROUND" "$1"; }
STAGES=(setup unit smoke check stats freeze audit equiv batchcheck short train val test confirm pack)

fail() { echo "STOPPED at $stage: $*"; echo "fix the cause and run the same command again (finished stages are skipped)"; exit 1; }
json_pass() { "$PY" -c "import json,sys; sys.exit(0 if all(json.load(open(f)).get('pass') for f in sys.argv[1:]) else 1)" "$@"; }
infer_batch() { cat "$W/infer_batch" 2>/dev/null || echo 8; }

for r in ${REDO:-}; do rm -f "${ST:?}/${r:?}.ok"; done
grep -qx "$SELF_IP" <(grep -v '^#' "$NODES" | awk 'NF {print $1}') || { echo "this node ($SELF_IP) is not in $NODES"; exit 1; }
echo "== run_all $LCAM_ROUND $(date '+%F %T')  code $(git rev-parse --short HEAD)  nodes $(grep -cv '^#' "$NODES")  model $LCAM_MODEL"
echo "   families: $(RI families)   training sources: $(RI kinds)"

for stage in "${STAGES[@]}"; do
  [ -f "$ST/$stage.ok" ] && { echo "-- $stage done earlier, skipped"; continue; }
  echo "== $stage $(date '+%F %T')"
  case $stage in
    setup)
      MT=$(RI model_type)
      "$PY" -c "import sys, transformers, peft, torch; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__); sys.exit(0 if '$MT' in CONFIG_MAPPING else 1)" \
        || fail "this venv cannot load model type $MT: install a transformers release that has it"
      if [ ! -f "$LCAM_MODEL/model.safetensors.index.json" ] && [ ! -f "$LCAM_MODEL/model.safetensors" ]; then
        "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$(RI hf_id)', revision='$(RI revision)', local_dir='$LCAM_MODEL')"
      fi
      "$PY" -c "import json,sys; c=json.load(open('$LCAM_MODEL/config.json')); sys.exit(0 if c['model_type']=='$MT' and (c['model_type']!='gemma4' or (c.get('text_config') or {}).get('enable_moe_block') is not True) else 1)" \
        || fail "$LCAM_MODEL is not a $MT model as configs/$LCAM_ROUND/common.yaml expects"
      if [ -n "${ADDON_TAR:-}" ]; then
        : "${ADDON_SHA256:?set ADDON_SHA256 with ADDON_TAR}"
        echo "$ADDON_SHA256  $ADDON_TAR" | sha256sum -c - || fail "add-on sha256 differs"
        tar -xf "$ADDON_TAR" -C "$BUNDLE" --keep-old-files 2>/dev/null || true     # files already present are kept, then checked below
      fi
      for m in "$BUNDLE"/ADDON_*_MANIFEST.json; do
        [ -f "$m" ] || continue
        "$PY" -c "import json,hashlib,sys; b='$BUNDLE'; m=json.load(open('$m')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open(b+'/'+p,'rb').read()).hexdigest()!=h]; print('$(basename "$m")', len(m['files']), 'files, bad', bad); sys.exit(1 if bad else 0)" \
          || fail "bundle files differ from $(basename "$m")"
      done
      for f in eval_prompts/test_V0.jsonl eval_prompts/validation_V0.jsonl eval_prompts/test_diagnostic_consequence.jsonl \
               eval_prompts/confirm_c1_direct_choice.jsonl eval_prompts/confirm_c2_reference.jsonl \
               $(for k in $(RI kinds); do echo training_exports/train/$k.jsonl training_exports/validation/$k.jsonl; done); do
        [ -f "$BUNDLE/$f" ] || fail "$BUNDLE/$f missing: the v1 bundle with the v2, confirm-1026 (and for v4 the v4) add-ons is needed"
      done
      (git rev-parse HEAD; git status --short) > "$W/code_commit.txt" ;;
    unit)
      "$PY" -m unittest discover -s tests > "$W/unit.log" 2>&1 || { tail -30 "$W/unit.log"; fail 'unit tests failed'; }
      tail -3 "$W/unit.log"
      LCAM_CONFIRM_PROMPTS=$BUNDLE/eval_prompts "$PY" -m unittest tests.test_confirm || fail 'C2 prompt rebuild differs from the frozen prompts' ;;
    smoke)
      SMOKE_SNAPSHOT=${SMOKE_SNAPSHOT:-$LCAM_MODEL} scripts/run_v2.sh smoke > "$W/smoke.log" 2>&1 || { tail -30 "$W/smoke.log"; fail "smoke test did not pass ($W/smoke.log)"; }
      tail -3 "$W/smoke.log"
      grep -q 'smoke passed' "$W/smoke.log" || fail 'smoke test did not pass' ;;
    check)
      scripts/run_v2.sh check
      json_pass "$A/check_data-train.json" "$A/check_data-validation.json" || fail 'data check failed (see audit/check_data-*.json)' ;;
    stats)
      scripts/run_v2.sh stats ;;
    freeze)
      scripts/run_v2.sh freeze || fail 'compute calibration outside tolerance; not frozen'
      cp "configs/$LCAM_ROUND/calibration.json" "$W/calibration.json" ;;
    audit)
      scripts/run_v2.sh audit || fail 'audit failed (batch traces: audit/batch_trace/summary-*.json)'
      "$PY" - "$W" <<'PYEOF' || fail 'learned tokens or images per sample are not as specified'
import glob, json, os, sys
w = sys.argv[1]
for s in glob.glob(os.path.join(w, 'audit', 'batch_trace', 'summary-*.json')):
    assert json.load(open(s))['pass'], s
print('batch traces pass')
PYEOF
      ;;
    equiv)
      scripts/run_v2.sh equiv || fail 'first-step gradients on 1 and 8 GPUs differ (audit/equiv.json)'
      "$PY" - "$A" "$(RI lora_on)" <<'PYEOF' || fail 'LoRA reached a frozen part, or is not on the configured layer kinds only'
import json, sys
m = json.load(open(sys.argv[1] + '/equiv-1gpu/trainable_manifest.json'))
cats = {x['category'] for x in m if 'lora_' in x['name']}
want = set(sys.argv[2].split())
print('LoRA categories', sorted(cats), 'expected', sorted(want), 'tensors', len(m))
sys.exit(0 if cats == want else 1)
PYEOF
      ;;
    batchcheck)
      scripts/run_v2.sh batchcheck
      if json_pass "$A/batchcheck.json"; then echo 8 > "$W/infer_batch"; else echo 1 > "$W/infer_batch"; fi
      echo "inference batch for val/test: $(infer_batch)" ;;
    short)
      scripts/run_v2.sh short || fail 'short runs or the adapter reload check failed (audit/reload_check.json)'
      "$PY" - "$A" $(RI short_families) <<'PYEOF'
import json, sys
for f in sys.argv[2:]:
    m = [json.loads(l) for l in open(f'{sys.argv[1]}/short-{f}/metrics.jsonl')]
    per = (m[-1]['train_seconds'] - m[2]['train_seconds']) / (len(m) - 3) if len(m) > 3 else m[-1]['train_seconds'] / len(m)
    print(f, 'steps', len(m), 'seconds/step (after warm-up)', round(per, 1))
PYEOF
      ;;
    train)
      "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/$LCAM_ROUND/calibration.json')).get('frozen') else 1)" || fail 'calibration not frozen'
      launch/train_queue.sh "$NODES" || fail "a training run failed twice (work/$LCAM_ROUND/queue/)" ;;
    val)
      LCAM_INFER_BATCH=$(infer_batch) scripts/run_v2.sh val || fail 'validation inference or checkpoint selection failed' ;;
    test)
      LCAM_INFER_BATCH=$(infer_batch) scripts/run_v2.sh test || fail 'test inference failed'
      want=$(( ($(RI families | wc -w) * $(RI seeds | wc -w) + 1) * 600 ))
      n=$(cat "$LOCUS_ROOT"/work/outputs/$LCAM_ROUND/*/test_V0.jsonl | wc -l)
      [ "$n" = "$want" ] || fail "expected $want test_V0 answers (every run and B0, 600 each), found $n" ;;
    confirm)
      "$PY" scripts/make_confirm_tasks.py --root "$LOCUS_ROOT" --bundle "$BUNDLE" > "$W/tasks-confirm.txt"
      cat "$W/confirm/build.json"
      launch/run_infer_pool.sh "$W/tasks-confirm.txt" "$NODES"
      while true; do
        left=0
        while IFS=$'\t' read -r out _; do [ -n "$out" ] && [ ! -e "$LOCUS_ROOT/$out.done" ] && left=$((left+1)); done < "$W/tasks-confirm.txt"
        grep -qF "work/outputs/$LCAM_ROUND-confirm" "$LOCUS_ROOT/work/logs/infer/failures.log" 2>/dev/null && fail 'a confirm job failed (work/logs/infer/failures.log)'
        [ "$left" = 0 ] && break
        echo "$(date +%T) $left confirm jobs left"; sleep 120
      done
      echo "confirm answers: $(cat "$LOCUS_ROOT"/work/outputs/$LCAM_ROUND-confirm/*/*.jsonl | wc -l)" ;;
    pack)
      cd "$LOCUS_ROOT"
      out=work/$LCAM_ROUND/locus-$LCAM_ROUND-results-$(date +%m%d).tar
      list=$(ls -d work/outputs/$LCAM_ROUND work/outputs/$LCAM_ROUND-valbest work/outputs/$LCAM_ROUND-confirm work/$LCAM_ROUND/audit \
                   work/$LCAM_ROUND/confirm work/$LCAM_ROUND/queue work/$LCAM_ROUND/*.txt work/$LCAM_ROUND/*.json work/$LCAM_ROUND/infer_batch \
                   work/$LCAM_ROUND/run_all.log work/runs/$LCAM_ROUND/*/run.json work/runs/$LCAM_ROUND/*/metrics.jsonl \
                   work/runs/$LCAM_ROUND/*/trainable_manifest.json work/runs/$LCAM_ROUND/*/selection.json work/runs/$LCAM_ROUND/*/val \
                   work/runs/$LCAM_ROUND/*/checkpoints/*/exposures.json work/logs/$LCAM_ROUND-*.log 2>/dev/null)
      sha256sum work/runs/$LCAM_ROUND/*/checkpoints/ckpt-4/adapter_model.safetensors > work/$LCAM_ROUND/checkpoint_sha256.txt
      tar -cf "$out" --exclude='*.safetensors' --exclude='*.pt' $list work/$LCAM_ROUND/checkpoint_sha256.txt
      sha256sum "$out" | tee "$out.sha256"
      cd "$LCAM_CODE" ;;
  esac
  touch "$ST/$stage.ok"
done
echo "== all stages finished $(date '+%F %T'); send work/$LCAM_ROUND/locus-$LCAM_ROUND-results-*.tar and its .sha256 back"
