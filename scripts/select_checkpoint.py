"""Checkpoint selection on validation only (pre-registered: V0 end-to-end success on the 400 validation roots).

  python scripts/select_checkpoint.py --bundle /data/locus-bundle --run runs/B1-primary-s0
  # expects runs/<..>/val/ckpt-<k>.jsonl written by scripts/infer.py on eval_prompts/validation_V0.jsonl

Ties go to the earlier checkpoint. Writes <run>/selection.json; the test set is run once, afterwards, with the
selected adapter only.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam.parse import outcome  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--expect', type=int, default=400, help='validation answers required per checkpoint')
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.bundle, 'validation_tables', 'index.json')))  # task_id -> root_id
    rows = []
    for path in sorted(glob.glob(os.path.join(a.run, 'val', 'ckpt-*.jsonl')), key=lambda p: int(p.rsplit('-', 1)[1].split('.')[0])):
        outs = [json.loads(l) for l in open(path)]
        res = {}
        for o in outs:
            table = json.load(open(os.path.join(a.bundle, 'validation_tables', index[o['task_id']] + '.json')))
            r = outcome(o['raw'], table)
            res[r] = res.get(r, 0) + 1
        n = len(outs)
        rows.append({'checkpoint': os.path.basename(path)[:-6], 'n': n, 'success': round(res.get('satisfied', 0) / n, 4) if n else None, 'outcomes': res})
    if not rows:
        raise SystemExit('no validation outputs found')
    if any(r['n'] != a.expect for r in rows):
        raise SystemExit(f'every checkpoint needs all {a.expect} validation answers: {[(r["checkpoint"], r["n"]) for r in rows]}')
    best = max(rows, key=lambda r: (r['success'], -int(r['checkpoint'].split('-')[1])))
    json.dump({'rule': 'max validation V0 success; ties -> earlier checkpoint', 'checkpoints': rows, 'selected': best['checkpoint']},
              open(os.path.join(a.run, 'selection.json'), 'w'), indent=1)
    print(json.dumps(rows, indent=1), '\nselected', best['checkpoint'])


if __name__ == '__main__':
    main()
