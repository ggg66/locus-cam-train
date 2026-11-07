"""Inference tasks of v10 (docs/v10/README.md): the geometry route on the endpoint checkpoint of every run.

  LCAM_ROUND=v10 python scripts/make_v10_tasks.py --root $LOCUS_ROOT > work/v10/tasks-v10-test.txt

For each run of configs/v10 (endpoint checkpoint, pre-registered; no validation selection):
  test_geom_predict             all 600 test scenes, geometry output for the fixed solver (addon v7)
  test_diagnostic_geom_predict  the 120 diagnostic scenes, same instruction (as v2 AG, for the field-replacement analysis)
Greedy, thinking off, batch 1, 1024 new tokens (the v7 setting of test_geom_predict). Runs whose endpoint checkpoint is
missing are listed on stderr and skipped.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lcam import config as C  # noqa: E402

ROUND = 'v10'
PROMPTS = [('test_geom_predict', 1024), ('test_diagnostic_geom_predict', 1024)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('LOCUS_ROOT'))
    a = ap.parse_args()
    common = C.load(os.path.join(HERE, 'configs', ROUND, 'common.yaml'))
    fr = [float(x) for x in common['checkpoints']['at_fraction']]
    ep = f'ckpt-{fr.index(float(common["checkpoints"]["primary"])) + 1}'
    missing = []
    for f in sorted(C.families(os.path.join(HERE, 'configs', ROUND))):
        for s in common['optim']['seeds']:
            tag = f'{f}-s{s}'
            adapter = f'work/runs/{ROUND}/{tag}/checkpoints/{ep}'
            if not os.path.exists(os.path.join(a.root, adapter, 'adapter_model.safetensors')):
                missing.append(tag)
                continue
            for name, maxnew in PROMPTS:
                print('\t'.join([f'work/outputs/{ROUND}/{tag}/{name}.jsonl', f'eval_prompts/{name}.jsonl', adapter, '0', '1', str(maxnew)]))
    if missing:
        print(' '.join(missing), file=sys.stderr)


if __name__ == '__main__':
    main()
