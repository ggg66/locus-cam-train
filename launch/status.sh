#!/bin/bash
# Progress of every v2 run: last step, objective, per-kind loss, speed, ETA, checkpoints.
source "$(dirname "$0")/env.sh"
"$PY" - "$LOCUS_ROOT" <<'PYEOF'
import glob, json, os, sys, time
root = sys.argv[1]
for rj in sorted(glob.glob(os.path.join(root, 'work/runs/v2/*/run.json'))):
    d = os.path.dirname(rj); tag = os.path.basename(d)
    try:
        r = json.load(open(rj))
    except Exception:
        print(tag, 'run.json unreadable'); continue
    total = r.get('steps_this_run') or r.get('planned_optimizer_steps', 0)
    mf = os.path.join(d, 'metrics.jsonl')
    ms = [json.loads(l) for l in open(mf) if l.strip()] if os.path.exists(mf) else []
    if not ms:
        print(f'{tag:16s} loading            0/{total}'); continue
    last = ms[-1]
    n = min(10, len(ms) - 1)
    per = (last['train_seconds'] - ms[-1 - n]['train_seconds']) / n if n else last['train_seconds']
    age = time.time() - os.path.getmtime(mf)
    state = 'FINISHED' if 'finished' in r else ('running' if age < 300 else f'STALLED {age / 60:.0f}min')
    kinds = ' '.join(f"{k}={v['mean_ce']:.4f}" for k, v in last['per_kind'].items() if v['mean_ce'] is not None)
    ck = len(glob.glob(os.path.join(d, 'checkpoints', 'ckpt-*')))
    print(f"{tag:16s} {state:12s} {last['step']}/{total}  obj={last['objective']:.4f}  {kinds}  {per:.1f}s/step  "
          f"ETA {(total - last['step']) * per / 3600:.2f}h  ckpt={ck}")
PYEOF
