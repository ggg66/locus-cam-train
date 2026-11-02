"""Explicit-prediction diagnostics of hosted models, after their V0 run: geometry route, consequence panel, C1 and C2.

  AIHUB_API_KEY=... python -u scripts/run_api_diagnostics.py --bundle $BUNDLE --v0 work/api --out work/api_diag --gateway aihub

For each model, 600 requests through scripts/run_api_models.py, with the settings of its V0 run (protocol.json of --v0: effort
and output budget) and the prompts the other models received: test_diagnostic_geom_predict, test_diagnostic_consequence,
confirm_c1_direct_choice and confirm_c2_reference (120 each), and confirm_c2_v0_<model>, built here from the model's own V0
actions with the frozen C2 template (scripts/make_confirm_tasks.py; a V0 answer without a valid action gets no request).
Models run in parallel, one process each; a first pass of one request per prompt file checks the route. Rerunning the same
command skips finished answers. Writes <out>/<model>/*.jsonl, c2_v0_prompts.jsonl, <out>/batch_status.json and, when every model
is complete, <out>.tar.gz with its .sha256. Test labels are not read here; scoring happens on the benchmark side.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'scripts'))
from lcam.parse import parse  # noqa: E402
from make_confirm_tasks import c2_record  # noqa: E402

MODELS = 'gpt-6-astra,gemini-4-argon,claude-opus-5-5'
FIXED = ('test_diagnostic_geom_predict', 'test_diagnostic_consequence', 'confirm_c1_direct_choice', 'confirm_c2_reference')


def read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def build_c2(c1_rows, v0_rows):
    """(C2 prompt records for the model's own V0 actions, task ids without a valid action)."""
    v0 = {o['task_id']: o for o in v0_rows if not o.get('sample', 0)}
    recs, no_action = [], []
    for r in c1_rows:
        if r['task_id'] not in v0:
            raise SystemExit(f'no V0 answer for {r["task_id"]}')
        act, err = parse(v0[r['task_id']].get('raw'))
        if err:
            no_action.append(r['task_id'])
        else:
            recs.append(c2_record(r, act))
    return recs, no_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--v0', required=True, help='output directory of the V0 run: <v0>/<model>/test_V0.jsonl, protocol.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default=MODELS)
    ap.add_argument('--gateway', default=None, help='passed to run_api_models.py')
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=1800)
    a = ap.parse_args()
    runner = os.path.join(HERE, 'scripts', 'run_api_models.py')
    models = a.models.split(',')
    for name in FIXED:
        if len(read(os.path.join(a.bundle, 'eval_prompts', f'{name}.jsonl'))) != 120:
            raise SystemExit(f'eval_prompts/{name}.jsonl must hold 120 prompts')
    c1 = read(os.path.join(a.bundle, 'eval_prompts', 'confirm_c1_direct_choice.jsonl'))
    os.makedirs(a.out, exist_ok=True)
    state, lock = {'state': 'running', 'models': {}}, threading.Lock()

    def update(model, **kw):
        with lock:
            state['models'].setdefault(model, {}).update(kw)
            json.dump(state, open(os.path.join(a.out, 'batch_status.json'), 'w'), indent=1)

    def run_model(m):
        proto = json.load(open(os.path.join(a.v0, m, 'protocol.json')))['protocol']
        recs, no_action = build_c2(c1, read(os.path.join(a.v0, m, 'test_V0.jsonl')))
        c2name = f'confirm_c2_v0_{m}'
        path = os.path.join(a.bundle, 'eval_prompts', f'{c2name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in recs)
        d = os.path.join(a.out, m)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(path, os.path.join(d, 'c2_v0_prompts.jsonl'))
        update(m, c2_requests=len(recs), no_action=no_action, c2_sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest())
        names = [*FIXED, c2name]
        cmd = [sys.executable, '-u', runner, '--bundle', a.bundle, '--out', a.out, '--models', m, '--prompts', ','.join(names),
               '--effort', proto['effort'], '--max-output-tokens', str(proto['max_output_tokens']),
               '--concurrency', str(a.concurrency), '--timeout', str(a.timeout)]
        if a.gateway:
            cmd += ['--gateway', a.gateway]
        with open(os.path.join(a.out, f'{m}.log'), 'ab', buffering=0) as log:
            for stage, extra in (('first request per file', ['--limit', '1']), ('full', [])):
                update(m, stage=stage, state='running')
                code = subprocess.call(cmd + extra, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
                if code:
                    update(m, state='stopped', exit_code=code)
                    return False
        counts = {n: len({o['task_id'] for o in read(os.path.join(d, f'{n}.jsonl'))}) for n in names}
        want = {**dict.fromkeys(FIXED, 120), c2name: len(recs)}
        update(m, counts=counts, state='complete' if counts == want else 'incomplete', finished_at=time.strftime('%F %T'))
        return counts == want

    with cf.ThreadPoolExecutor(len(models)) as ex:
        done = list(ex.map(run_model, models))
    state['state'] = 'complete' if all(done) else 'needs_attention'
    update(models[-1])
    if not all(done):
        print('see', os.path.join(a.out, 'batch_status.json'))
        return 2
    archive = a.out.rstrip('/') + '.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(a.out, arcname=os.path.basename(a.out.rstrip('/')))
    digest = hashlib.sha256(open(archive, 'rb').read()).hexdigest()
    open(archive + '.sha256', 'w').write(f'{digest}  {os.path.basename(archive)}\n')
    print('complete:', archive)
    return 0


if __name__ == '__main__':
    sys.exit(main())
