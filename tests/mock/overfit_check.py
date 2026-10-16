"""End-to-end consistency check: train.py must be able to make a model reproduce two fixed answers, and infer.py
(a separate code path that builds the prompt again) must then decode exactly those answers. A mismatch between the
training sequence and the inference prompt, a label shift, or a broken mask shows up here as a wrong answer.

  python tests/mock/overfit_check.py --config <dir with common.yaml> --work <scratch dir> [--steps 100]

Uses short records without images so that it runs on a CPU in a few minutes. Exit code 0 = both answers exact.
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--work', required=True)
ap.add_argument('--steps', type=int, default=100)
ap.add_argument('--lr', type=float, default=5e-3)
a = ap.parse_args()
w = os.path.abspath(a.work)
bundle = os.path.join(w, 'bundle')
os.makedirs(os.path.join(bundle, 'training_exports', 'train'), exist_ok=True)
system = '你在一个三维白模场景里调整相机位置。只输出要求的 JSON。'
recs = [{'id': f'overfit-{i}:action', 'root_id': f'overfit-{i}', 'pair_group_id': f'overfit-{i}', 'task_id': f'tover{i}', 'source_split': 'train',
         'images': [], 'system': system, 'prompt': p, 'target': json.dumps({'move_m': dict(zip(('right', 'up', 'back'), act))})}
        for i, (p, act) in enumerate((('题目甲：人在柱子后面，往哪挪？', (1, 0, -2)), ('题目乙：人太小了，往哪挪？', (-2, 1, 0))))]
with open(os.path.join(bundle, 'training_exports', 'train', 'action_sft.jsonl'), 'w', encoding='utf-8') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
json.dump({}, open(os.path.join(bundle, 'splits_index.json'), 'w'))
with open(os.path.join(bundle, 'prompts.jsonl'), 'w', encoding='utf-8') as f:
    for r in recs:
        f.write(json.dumps({'task_id': r['task_id'], 'sample': 0, 'condition': 'V0', 'split': 'train', 'images': [], 'system': system,
                            'prompt': r['prompt']}, ensure_ascii=False) + '\n')
cfgdir = os.path.join(w, 'configs')
os.makedirs(cfgdir, exist_ok=True)
common = yaml.safe_load(open(os.path.join(a.config, 'common.yaml')))
common['optim'].update(lr=a.lr, warmup_ratio=0.0, effective_batch_size=2)
yaml.safe_dump(common, open(os.path.join(cfgdir, 'common.yaml'), 'w'), allow_unicode=True, sort_keys=False)
yaml.safe_dump({'base': 'common.yaml', 'arm': 'B1', 'mix': {'action_sft': 1.0}, 'budget': {'primary_fixed_total': {'samples': 2 * a.steps}}},
               open(os.path.join(cfgdir, 'B1.yaml'), 'w'), sort_keys=False)
run = os.path.join(w, 'run')
py = sys.executable
subprocess.run([py, os.path.join(REPO, 'scripts', 'train.py'), '--config', os.path.join(cfgdir, 'B1.yaml'), '--budget', 'primary_fixed_total',
                '--seed', '0', '--bundle', bundle, '--out', run, '--micro-batch', '2', '--workers', '0', '--allow-optimizer-step'], check=True)
ckpts = sorted(os.listdir(os.path.join(run, 'checkpoints')), key=lambda c: int(c.split('-')[1]))
out = os.path.join(w, 'answers.jsonl')
if os.path.exists(out):
    os.remove(out)
subprocess.run([py, os.path.join(REPO, 'scripts', 'infer.py'), '--config', os.path.join(cfgdir, 'common.yaml'), '--bundle', bundle,
                '--prompts', 'prompts.jsonl', '--out', out, '--adapter', os.path.join(run, 'checkpoints', ckpts[-1]), '--max-new-tokens', '40',
                '--batch', '2'], check=True)
answers = {json.loads(l)['task_id']: json.loads(l)['raw'] for l in open(out, encoding='utf-8')}
losses = [json.loads(l)['loss'] for l in open(os.path.join(run, 'metrics.jsonl'))]
ok = True
for r in recs:
    got = answers[r['task_id']].strip()
    print(('OK  ' if got == r['target'] else 'FAIL'), r['task_id'], 'want', r['target'], 'got', repr(got))
    ok &= got == r['target']
print('loss first/last', losses[0], losses[-1], '| checkpoint', ckpts[-1])
sys.exit(0 if ok else 1)
