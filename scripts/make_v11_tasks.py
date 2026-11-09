"""Inference tasks of v11 (docs/v11/README.md) on the endpoint checkpoint of every run (pre-registered; no validation selection).

  LCAM_ROUND=v11 python scripts/make_v11_tasks.py --root $LOCUS_ROOT > work/v11/tasks-v11-test.txt

  A_rand                 test_V0 (600 test tasks, the action prompt), 512 new tokens
  GA_rand                test_V0 with 1,024 new tokens (geometry, then the action); test_geomact_true / test_geomact_mirror
                         (the answer starts with the true or the left-right mirrored scene geometry in the training-answer
                         format; addon v7), 128 new tokens for the action
  G_rand_r, G_k10       test_geom_predict (600 test tasks, geometry for the fixed solver), 1,024 new tokens
  D_fam, D_rand         images withheld, geometry in the prompt (addon v11): test_blind_true (true geometry in the training-answer
                        format), test_blind_grand_s<k> and test_blind_gfam_s<k> (the geometry v10 G_rand-s<k> / G_fam-s<k>
                        predicted; seed k of the D run), 512 new tokens
  D_fam_anon            the same three prompts without object names: test_blind_anon_true, test_blind_anon_grand_s<k>,
                        test_blind_anon_gfam_s<k>
Greedy, thinking off, batch 1. Runs whose endpoint checkpoint is missing are listed on stderr and skipped.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lcam import config as C  # noqa: E402

ROUND = 'v11'
GEOM = [('test_geom_predict', 1024)]
PROMPTS = {'A_rand': [('test_V0', 512)],
           'GA_rand': [('test_V0', 1024), ('test_geomact_true', 128), ('test_geomact_mirror', 128)],
           'G_rand_r': GEOM, 'G_k10': GEOM, 'D_fam': 'blind', 'D_rand': 'blind', 'D_fam_anon': 'blind_anon'}


def prompts(f, seed):
    if PROMPTS[f] in ('blind', 'blind_anon'):
        x = 'test_blind_' if PROMPTS[f] == 'blind' else 'test_blind_anon_'
        return [(f'{x}true', 512), (f'{x}grand_s{seed}', 512), (f'{x}gfam_s{seed}', 512)]
    return PROMPTS[f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('LOCUS_ROOT'))
    a = ap.parse_args()
    common = C.load(os.path.join(HERE, 'configs', ROUND, 'common.yaml'))
    fr = [float(x) for x in common['checkpoints']['at_fraction']]
    ep = f'ckpt-{fr.index(float(common["checkpoints"]["primary"])) + 1}'
    fams = sorted(C.families(os.path.join(HERE, 'configs', ROUND)))
    if set(fams) != set(PROMPTS):
        raise SystemExit(f'configs/{ROUND} has {fams}, tasks are defined for {sorted(PROMPTS)}')
    missing = []
    for f in fams:
        for s in common['optim']['seeds']:
            tag = f'{f}-s{s}'
            adapter = f'work/runs/{ROUND}/{tag}/checkpoints/{ep}'
            if not os.path.exists(os.path.join(a.root, adapter, 'adapter_model.safetensors')):
                missing.append(tag)
                continue
            for name, maxnew in prompts(f, s):
                print('\t'.join([f'work/outputs/{ROUND}/{tag}/{name}.jsonl', f'eval_prompts/{name}.jsonl', adapter, '0', '1', str(maxnew)]))
    if missing:
        print(' '.join(missing), file=sys.stderr)


if __name__ == '__main__':
    main()
