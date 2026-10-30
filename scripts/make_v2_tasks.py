"""Plan and inference task files for the v2 batch (paths relative to $LOCUS_ROOT, the layout of launch/).

  python scripts/make_v2_tasks.py train   --nodes nodes.txt          > work/v2/plan-train.txt
  python scripts/make_v2_tasks.py val                                > work/v2/tasks-val.txt
  python scripts/make_v2_tasks.py test                               > work/v2/tasks-test.txt
  python scripts/make_v2_tasks.py valbest --root $LOCUS_ROOT         > work/v2/tasks-valbest.txt

train    one line per run "<node_ip> <config> <seed>": 3 families x the seeds of configs/v2/common.yaml, nodes in turn.
val      validation V0 for every checkpoint of every run (checkpoint selection, reported separately).
test     the evaluation suite on the pre-registered endpoint (the checkpoint at fraction 1.0) of every run, and on B0
         (no adapter, thinking off); plus the B0 dev diagnostic of the zero-action example in the V0 prompt.
valbest  the same suite on each run's validation-best checkpoint, for runs where it is not the endpoint
         (needs <run>/selection.json from scripts/select_checkpoint.py).
Task line (launch/run_infer_pool.sh): out<TAB>prompts<TAB>adapter|-<TAB>thinking<TAB>batch<TAB>max_new_tokens; batch is
$LCAM_INFER_BATCH (default 8; set 1 if scripts/run_v2.sh batchcheck finds answers that depend on batching).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lcam import config as C  # noqa: E402

FAMILIES = ['A_base', 'AG', 'A_compute']
ROUND = os.environ.get('LCAM_ROUND', 'v2')   # v3_gemma: the same suite on the Gemma 4 runs
RUNS = f'work/runs/{ROUND}'
OUTS = f'work/outputs/{ROUND}'
SUITE = [  # prompt file (bundle eval_prompts/), max new tokens
    ('test_V0', 512), ('test_variants_V0', 512), ('test_target_swap_V0', 512),
    ('test_diagnostic_geom_predict', 1024), ('test_diagnostic_consequence', 2048)]
B0_ONLY = [('dev_V0', 512), ('dev_V0_placeholder', 512)]


def common():
    return C.load(os.path.join(HERE, 'configs', ROUND, 'common.yaml'))


def runs():
    seeds = common()['optim']['seeds']
    return [(f, s, f'{f}-s{s}') for f in FAMILIES for s in seeds]


def endpoint_ckpt():
    fr = [float(x) for x in common()['checkpoints']['at_fraction']]
    return f'ckpt-{fr.index(float(common()["checkpoints"]["primary"])) + 1}'


def line(out, prompts, adapter, maxnew, batch=None):
    batch = batch or int(os.environ.get('LCAM_INFER_BATCH', 8))   # 1 when the batchcheck stage finds batch-dependent answers
    return '\t'.join([out, f'eval_prompts/{prompts}.jsonl', adapter, '0', str(batch), str(maxnew)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['train', 'val', 'test', 'valbest'])
    ap.add_argument('--nodes')
    ap.add_argument('--root', default=os.environ.get('LOCUS_ROOT'))
    a = ap.parse_args()
    out = []
    if a.what == 'train':
        nodes = [x.split()[0] for x in open(a.nodes) if x.strip() and not x.startswith('#')]
        for i, (f, s, _) in enumerate(runs()):
            out.append(f'{nodes[i % len(nodes)]} configs/{ROUND}/{f}.yaml {s}')
    elif a.what == 'val':
        n = len(common()['checkpoints']['at_fraction'])
        for _, _, tag in runs():
            for k in range(1, n + 1):
                out.append(line(f'{RUNS}/{tag}/val/ckpt-{k}.jsonl', 'validation_V0', f'{RUNS}/{tag}/checkpoints/ckpt-{k}', 512))
    elif a.what == 'test':
        ep = endpoint_ckpt()
        for _, _, tag in runs():
            for p, m in SUITE:
                out.append(line(f'{OUTS}/{tag}/{p}.jsonl', p, f'{RUNS}/{tag}/checkpoints/{ep}', m))
        for p, m in SUITE + B0_ONLY:
            out.append(line(f'{OUTS}/B0-nothink/{p}.jsonl', p, '-', m))
    else:
        ep = endpoint_ckpt()
        for _, _, tag in runs():
            sel = json.load(open(os.path.join(a.root, RUNS, tag, 'selection.json')))['selected']
            if sel == ep:
                print(f'# {tag}: validation-best is the endpoint {ep}; no extra run', file=sys.stderr)
                continue
            for p, m in SUITE:
                out.append(line(f'{OUTS}-valbest/{tag}/{p}.jsonl', p, f'{RUNS}/{tag}/checkpoints/{sel}', m))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
