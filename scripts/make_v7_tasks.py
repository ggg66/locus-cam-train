"""Inference tasks of v7 (docs/v7/README.md): finished checkpoints only, no training.

  LCAM_ROUND=v2 python scripts/make_v7_tasks.py --root $LOCUS_ROOT > work/v7/tasks-v2.txt

For each finished run of the round (endpoint checkpoint present), and for the round's untrained base:
  test_diagnostic_S        120 diagnostic scenes with the reference geometry written into the prompt (every run)
  test_geom_predict        all 600 test scenes, geometry output for the fixed solver (Action+Geometry runs)
  test_geomact_true        all 600 test scenes, the answer starts with the true scene geometry (Geometry->Action runs)
  test_geomact_mirror      the same with that geometry mirrored left-right (Geometry->Action runs)
  test_V0_renamed_heldout  the 200 held-out-family tasks with the four object names absent from training replaced by
                           training names (every run)
  test_cons_panel          the 12 panel candidates of the 120 diagnostic scenes, one prompt each in the Action+Consequence
                           training format; validation_cons: 2,330 validation prompts in that format (Action+Consequence runs)
Runs whose endpoint checkpoint is missing are listed on stderr and skipped.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lcam import config as C  # noqa: E402

ROUND = os.environ.get('LCAM_ROUND', 'v2')
RUNS = f'work/runs/{ROUND}'
OUTS = f'work/outputs/v7/{ROUND}'
CONS = [('test_cons_panel', 512), ('validation_cons', 512)]
PER_FAMILY = {'AG': [('test_geom_predict', 1024)], 'A_geomact': [('test_geomact_true', 128), ('test_geomact_mirror', 128)],
              'A_cons': CONS, 'A_cons_full': CONS, 'A_cons_dense': CONS}


def cfg(name):
    return C.load(os.path.join(HERE, 'configs', ROUND, f'{name}.yaml'))


def line(out, prompts, adapter, maxnew):
    return '\t'.join([out, f'eval_prompts/{prompts}.jsonl', adapter, '0', '1', str(maxnew)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('LOCUS_ROOT'))
    a = ap.parse_args()
    common = cfg('common')
    fr = [float(x) for x in common['checkpoints']['at_fraction']]
    ep = f'ckpt-{fr.index(float(common["checkpoints"]["primary"])) + 1}'
    out, missing = [line(f'{OUTS}/B0-nothink/test_diagnostic_S.jsonl', 'test_diagnostic_S', '-', 512)], []
    for f in C.families(os.path.join(HERE, 'configs', ROUND)):
        fc = cfg(f)
        if fc.get('reference_only'):
            continue
        floor = int(fc.get('eval_min_new_tokens', 0))
        for s in common['optim']['seeds']:
            tag = f'{f}-s{s}'
            adapter = f'{RUNS}/{tag}/checkpoints/{ep}'
            if not os.path.exists(os.path.join(a.root, adapter, 'adapter_model.safetensors')):
                missing.append(tag)
                continue
            out.append(line(f'{OUTS}/{tag}/test_diagnostic_S.jsonl', 'test_diagnostic_S', adapter, max(512, floor)))
            out.append(line(f'{OUTS}/{tag}/test_V0_renamed_heldout.jsonl', 'test_V0_renamed_heldout', adapter, max(512, floor)))
            for p, m in PER_FAMILY.get(f, []):
                out.append(line(f'{OUTS}/{tag}/{p}.jsonl', p, adapter, m))
    if missing:
        print(f'# {ROUND}: endpoint checkpoint missing, skipped: {" ".join(missing)}', file=sys.stderr)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
