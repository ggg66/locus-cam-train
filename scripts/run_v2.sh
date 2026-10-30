#!/bin/bash
# v2 batch on the training server, stage by stage (docs/v2/README.md). Every stage writes under $LOCUS_ROOT/work/$LCAM_ROUND/.
#   LOCUS_ROOT=... scripts/run_v2.sh <stage> [<stage> ...]
# Stages, in order:
#   addon      extract the v2 bundle add-on into $BUNDLE after checking its sha256   (ADDON_TAR, ADDON_SHA256)
#   unit       unit tests (stdlib + torch)
#   smoke      tiny random model + mini bundle through every v2 script, 1 and 2 GPUs (minutes; SMOKE_SNAPSHOT = a directory
#              with the model's config/tokenizer files, default $LCAM_MODEL)
#   check      real-processor data check of the v2 sources, train and validation
#   stats      per-record token counts (encode_stats.py), input of calibrate and audit
#   audit      v1 loss weighting (no GPU), v1 loss on the real model (1 GPU), batch traces of A_base and AG
#   equiv      first-step gradients of AG s0 on 1 GPU and on 8 GPUs (dropout off) must agree
#   batchcheck B0 on validation V0 with inference batch 1 and batch 8: if answers differ, evaluate with batch 1
#   short      12-step cost runs of A_base and AG on 8 GPUs (+ adapter save) and the reload check
#   calibrate  A-compute budget from padded tokens; prints the result, does not freeze
#   freeze     the same, frozen into configs/$LCAM_ROUND/calibration.json (commit that file before training)
#   train      the 9 runs; single node one after another, or NODES=<nodefile> to spread them   (ALLOW_TRAIN=1)
#   val        validation V0 at every checkpoint (NODES=<nodefile>), then checkpoint selection
#   test       the evaluation suite on every endpoint and on B0 (NODES=<nodefile>)
#   pack       one tar with everything the benchmark side needs to score (no adapters)
set -euo pipefail
source "$(dirname "$0")/../launch/env.sh"
cd "$LCAM_CODE"
W=$LOCUS_ROOT/work/$LCAM_ROUND
A=$W/audit
mkdir -p "$W" "$A" "$LOCUS_ROOT/work/logs"
SEEDS=$("$PY" -c "import sys; sys.path.insert(0, '.'); from lcam import config as C; print(*C.load('configs/$LCAM_ROUND/common.yaml')['optim']['seeds'])")

wait_pool() {   # until every job of a task file has its .done marker; stops when one of them failed
  local tasks=$1 fails=$LOCUS_ROOT/work/logs/infer/failures.log
  while true; do
    local left=0
    while IFS=$'\t' read -r out _; do
      [ -z "$out" ] && continue
      [ -e "$LOCUS_ROOT/$out.done" ] && continue
      left=$((left+1))
      if [ -f "$fails" ] && grep -qF "FAILED $out " "$fails"; then echo "job failed: $out (see $fails); fix and rerun the stage"; exit 1; fi
    done < "$tasks"
    [ "$left" = 0 ] && break
    echo "$(date +%T) $left jobs left in $(basename "$tasks")"; sleep 120
  done
}

run_pool() {    # task file -> results, on NODES if given, else on the 8 GPUs of this node
  local tasks=$1 nodes=${NODES:-}
  if [ -z "$nodes" ]; then nodes=$W/nodes-self.txt; echo "$SELF_IP" > "$nodes"; fi
  launch/run_infer_pool.sh "$tasks" "$nodes"
  wait_pool "$tasks"
}

for stage in "$@"; do
  echo "== $stage $(date '+%F %T')"
  case $stage in
    addon)
      : "${ADDON_TAR:?}" "${ADDON_SHA256:?}"
      echo "$ADDON_SHA256  $ADDON_TAR" | sha256sum -c -
      tar -xf "$ADDON_TAR" -C "$BUNDLE" --keep-old-files
      "$PY" -c "import json,hashlib,sys; m=json.load(open('$BUNDLE/ADDON_V2_MANIFEST.json')); bad=[p for p,h in m['files'].items() if hashlib.sha256(open('$BUNDLE/'+p,'rb').read()).hexdigest()!=h]; print('addon files', len(m['files']), 'bad', bad); sys.exit(1 if bad else 0)" ;;
    unit)
      "$PY" -m unittest discover -s tests -v 2>&1 | tail -30 ;;
    smoke)
      S=$W/smoke; rm -rf "$S"; mkdir -p "$S"
      "$PY" tests/mock/make_tiny_model.py "${SMOKE_SNAPSHOT:-$LCAM_MODEL}" "$S/tiny-model"
      "$PY" tests/mock/make_mini_bundle.py "$BUNDLE" "$S/mini-bundle"
      "$PY" tests/mock/make_smoke_config.py "$S/tiny-model" "$S/configs"
      B=$S/mini-bundle; C=$S/configs/$LCAM_ROUND
      T=(--bundle "$B" --micro-batch "${MICRO_BATCH:-1}" --workers 0 --allow-optimizer-step)
      "$PY" scripts/check_data.py --config "$C/common.yaml" --bundle "$B" --kinds action_sft geometry_sft_v2 > "$S/check.json"
      "$PY" scripts/encode_stats.py --config "$C/common.yaml" --bundle "$B" --kinds action_sft geometry_sft geometry_sft_v2 --out "$S/stats.json" --workers 2
      "$PY" scripts/trace_batch.py --config "$C/AG.yaml" --seed 0 --bundle "$B" --out "$S/trace"
      "$PY" scripts/audit_v1_weighting.py --stats "$S/stats.json" --world-size 1 > "$S/v1_weighting.json" || true   # v1 budgets exceed the mini bundle; only checks it runs
      CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_v2.py --config "$C/AG.yaml" --seed 0 --out "$S/g1" "${T[@]}" --max-steps 1 --disable-dropout --dump-first-grad "$S/g1.pt"
      CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --nproc-per-node 2 --master-port 29650 scripts/train_v2.py --config "$C/AG.yaml" --seed 0 \
        --out "$S/g2" "${T[@]}" --max-steps 1 --disable-dropout --dump-first-grad "$S/g2.pt"
      "$PY" scripts/compare_grads.py "$S/g1.pt" "$S/g2.pt"
      for f in A_base AG; do CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_v2.py --config "$C/$f.yaml" --seed 0 --out "$S/runs/$f" "${T[@]}"; done
      "$PY" scripts/calibrate_compute.py --stats "$S/stats.json" --micro-batch "${MICRO_BATCH:-1}" --world-size 1 --config-dir "$C" --freeze
      CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_v2.py --config "$C/A_compute.yaml" --seed 0 --out "$S/runs/A_compute" "${T[@]}"
      last=$(ls -d "$S/runs/AG/checkpoints"/ckpt-* | sort -t- -k2 -n | tail -1)
      CUDA_VISIBLE_DEVICES=0 "$PY" scripts/reload_check.py --config "$C/common.yaml" --bundle "$B" --ckpt "$last"
      for f in A_base AG A_compute; do
        for ck in "$S/runs/$f/checkpoints"/ckpt-*; do
          CUDA_VISIBLE_DEVICES=0 "$PY" scripts/infer.py --config "$C/common.yaml" --bundle "$B" --prompts eval_prompts/validation_V0.jsonl \
            --out "$S/runs/$f/val/$(basename "$ck").jsonl" --adapter "$ck" --max-new-tokens 8
        done
        "$PY" scripts/select_checkpoint.py --bundle "$B" --run "$S/runs/$f" --expect "$(wc -l < "$B/eval_prompts/validation_V0.jsonl")"
      done
      for p in test_diagnostic_consequence test_target_swap_V0 dev_V0_placeholder; do
        CUDA_VISIBLE_DEVICES=0 "$PY" scripts/infer.py --config "$C/common.yaml" --bundle "$B" --prompts eval_prompts/$p.jsonl --out "$S/out/$p.jsonl" --adapter "$last" --max-new-tokens 8
      done
      echo "smoke passed" ;;
    check)
      for s in train validation; do "$PY" scripts/check_data.py --config configs/$LCAM_ROUND/common.yaml --bundle "$BUNDLE" --split $s --kinds action_sft geometry_sft_v2 > "$A/check_data-$s.json"; done
      grep -h '"pass"' "$A"/check_data-*.json ;;
    stats)
      "$PY" scripts/encode_stats.py --config configs/$LCAM_ROUND/common.yaml --bundle "$BUNDLE" --kinds action_sft geometry_sft geometry_sft_v2 --out "$A/encode_stats-train.json" --workers 16 ;;
    audit)
      "$PY" scripts/audit_v1_weighting.py --stats "$A/encode_stats-train.json" > "$A/v1_weighting.json"
      CUDA_VISIBLE_DEVICES=0 "$PY" scripts/loss_audit.py --config configs/$LCAM_ROUND/common.yaml --bundle "$BUNDLE" --out "$A/loss_audit.json"
      for f in A_base AG; do "$PY" scripts/trace_batch.py --config configs/$LCAM_ROUND/$f.yaml --seed 0 --bundle "$BUNDLE" --out "$A/batch_trace"; done ;;
    equiv)
      CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_v2.py --config configs/$LCAM_ROUND/AG.yaml --seed 0 --bundle "$BUNDLE" --out "$A/equiv-1gpu" \
        --micro-batch "${MICRO_BATCH:-1}" --allow-optimizer-step --max-steps 1 --disable-dropout --dump-first-grad "$A/grad-1gpu.pt"
      OUT=$A/equiv-8gpu launch/launch_run_v2.sh configs/$LCAM_ROUND/AG.yaml 0 29610 --max-steps 1 --disable-dropout --dump-first-grad "$A/grad-8gpu.pt"
      "$PY" scripts/compare_grads.py "$A/grad-1gpu.pt" "$A/grad-8gpu.pt" | tee "$A/equiv.json" ;;
    batchcheck)
      for b in 1 8; do
        CUDA_VISIBLE_DEVICES=$((b == 1 ? 0 : 1)) "$PY" scripts/infer.py --config configs/$LCAM_ROUND/common.yaml --bundle "$BUNDLE" \
          --prompts eval_prompts/validation_V0.jsonl --out "$A/batch$b.jsonl" --batch $b > "$A/batch$b.log" 2>&1 &
      done
      wait
      "$PY" scripts/compare_outputs.py "$A/batch1.jsonl" "$A/batch8.jsonl" | tee "$A/batchcheck.json" || echo 'answers depend on batching: set LCAM_INFER_BATCH=1 for val/test' ;;
    short)
      OUT=$A/short-A_base launch/launch_run_v2.sh configs/$LCAM_ROUND/A_base.yaml 0 29611 --max-steps 12
      OUT=$A/short-AG launch/launch_run_v2.sh configs/$LCAM_ROUND/AG.yaml 0 29612 --max-steps 12 --save-at-end
      CUDA_VISIBLE_DEVICES=0 "$PY" scripts/reload_check.py --config configs/$LCAM_ROUND/common.yaml --bundle "$BUNDLE" --ckpt "$A/short-AG/checkpoints/ckpt-1" | tee "$A/reload_check.json" ;;
    calibrate|freeze)
      extra=(); [ "$stage" = freeze ] && extra=(--freeze)
      "$PY" scripts/calibrate_compute.py --stats "$A/encode_stats-train.json" --micro-batch "${MICRO_BATCH:-1}" --world-size 8 \
        --gpu-seconds "A_base=$A/short-A_base/metrics.jsonl" "AG=$A/short-AG/metrics.jsonl" "${extra[@]}"
      cp configs/$LCAM_ROUND/calibration.json "$A/calibration.json"
      [ "$stage" = freeze ] && "$PY" scripts/trace_batch.py --config configs/$LCAM_ROUND/A_compute.yaml --seed 0 --bundle "$BUNDLE" --out "$A/batch_trace" ;;
    train)
      [ "${ALLOW_TRAIN:-0}" = 1 ] || { echo 'set ALLOW_TRAIN=1 (after the audit has been reviewed and calibration.json is frozen)'; exit 1; }
      "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/$LCAM_ROUND/calibration.json')).get('frozen') else 1)" || { echo 'calibration.json not frozen'; exit 1; }
      if [ -n "${NODES:-}" ]; then
        "$PY" scripts/make_v2_tasks.py train --nodes "$NODES" > "$W/plan-train.txt"
        launch/launch_batch_v2.sh "$W/plan-train.txt"
      else
        for f in A_base AG A_compute; do for s in $SEEDS; do
          grep -q '"finished"' "$LOCUS_ROOT/work/runs/$LCAM_ROUND/$f-s$s/run.json" 2>/dev/null && continue
          launch/launch_run_v2.sh configs/$LCAM_ROUND/$f.yaml "$s" 29600 > "$LOCUS_ROOT/work/logs/$LCAM_ROUND-$f-s$s.log" 2>&1
        done; done
      fi ;;
    val)
      "$PY" scripts/make_v2_tasks.py val > "$W/tasks-val.txt"
      run_pool "$W/tasks-val.txt"
      for f in A_base AG A_compute; do for s in $SEEDS; do
        "$PY" scripts/select_checkpoint.py --bundle "$BUNDLE" --run "$LOCUS_ROOT/work/runs/$LCAM_ROUND/$f-s$s"
      done; done ;;
    test)
      "$PY" scripts/make_v2_tasks.py test > "$W/tasks-test.txt"
      run_pool "$W/tasks-test.txt"
      "$PY" scripts/make_v2_tasks.py valbest --root "$LOCUS_ROOT" > "$W/tasks-valbest.txt"
      [ -s "$W/tasks-valbest.txt" ] && run_pool "$W/tasks-valbest.txt" || true ;;
    pack)
      cd "$LOCUS_ROOT"
      out=work/$LCAM_ROUND/locus-$LCAM_ROUND-results-$(date +%m%d).tar
      list=$(ls -d work/outputs/$LCAM_ROUND work/outputs/$LCAM_ROUND-valbest work/$LCAM_ROUND/audit work/$LCAM_ROUND/*.txt work/runs/$LCAM_ROUND/*/run.json work/runs/$LCAM_ROUND/*/metrics.jsonl \
                   work/runs/$LCAM_ROUND/*/trainable_manifest.json work/runs/$LCAM_ROUND/*/selection.json work/runs/$LCAM_ROUND/*/val work/runs/$LCAM_ROUND/*/checkpoints/*/exposures.json 2>/dev/null)
      tar -cf "$out" --exclude='*.safetensors' --exclude='*.pt' $list
      sha256sum "$out" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
