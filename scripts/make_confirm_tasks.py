"""C1 / C2 inference controls for a training round, built on the training server (no test labels needed).

  python scripts/make_confirm_tasks.py --root $LOCUS_ROOT --bundle $BUNDLE > work/<round>/tasks-confirm.txt

The 2026-10-26 add-on (bundle eval_prompts/confirm_c1_direct_choice.jsonl, confirm_c2_reference.jsonl) holds the
model-independent prompts. C2 also needs every run's own V0 action on the 120 diagnostic roots, which only exists
after the round's test stage; this script builds those prompts from work/outputs/<round>/<run>/test_V0.jsonl with the
frozen C2 template (benchmark reports/paper_confirm_1026/build_confirm.py c2_prompt). The template text is
rebuilt from the C1 prompt of the same root, whose task part is the same text; tests/test_confirm.py checks the result
byte for byte against the frozen v2 prompts. A V0 answer without a valid action (lcam.parse: format or domain error)
is NO_ACTION and gets no request, as in the frozen protocol.
Writes work/<round>/confirm/prompts/confirm_c2_v0_<run>.jsonl, work/<round>/confirm/build.json and prints the task
lines for launch/run_infer_pool.sh: C1 for B0 and every run, C2 (own V0 action + reference action) for every run;
batch 1, max_new_tokens 2048 (the frozen setting).
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lcam.parse import parse  # noqa: E402

ROUND = os.environ.get('LCAM_ROUND', 'v2')
from lcam import config as C  # noqa: E402
FAMILIES = C.families(os.path.join(HERE, 'configs', ROUND))
CANDIDATES = '\n\n下面是 12 个候选挪法（含义同上）：\n'
APART = '；apart：两个目标在画面上是否水平分开'
KEY = '%+d,%+d,%+d'
MAX_NEW = 2048


def definitions(two_targets):
    apart = APART if two_targets else ''
    return ('legal：终点是否合法（不在物体内、不出房间、高度合规）；在终点合法时再判断 in_frame：目标是否完整入画并留出要求的边距；'
            'size_ok：目标包围盒高度占比是否在要求范围内；unoccluded：按 3×3 采样点规则是否不被遮挡' + apart)


def c2_prompt(c1_prompt, action):
    """The frozen C2 text for one root and one action, from the C1 prompt of that root."""
    base = c1_prompt.split(CANDIDATES)[0]
    two = APART in c1_prompt
    apart_fill = '' if two else '（本题只有一个目标，apart 填 null）'
    return (base + f'\n\n待评估动作：right {action[0]}，up {action[1]}，back {action[2]}（含义同上）。\n\n'
            '请判断：如果相机按这个挪法移动，会发生什么。字段含义：' + definitions(two) + apart_fill + '。'
            '终点合法时，这几个字段每一个都要单独判断并填 true 或 false，即使前面已有某项不满足也不能留空；只有终点不合法时其余字段才填 null。\n\n'
            '只输出一个 JSON，格式如下（尖括号处换成你的判断，每个字段只能是 true、false 或 null）：\n'
            '{"legal": <…>, "in_frame": <…>, "size_ok": <…>, "unoccluded": <…>, "apart": <…>}')


def c2_record(c1, action):
    keep = ('task_id', 'root_id', 'split', 'sample', 'variant_task', 'images', 'system', 'runner_version')
    return {**{k: c1[k] for k in keep if k in c1}, 'condition': 'confirm_c2_judge', 'prompt': c2_prompt(c1['prompt'], action),
            'eval_action': KEY % tuple(action)}


def read(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('LOCUS_ROOT'))
    ap.add_argument('--bundle', default=os.environ.get('BUNDLE'))
    ap.add_argument('--seeds', default='0,1,2')
    a = ap.parse_args()
    c1 = read(os.path.join(a.bundle, 'eval_prompts', 'confirm_c1_direct_choice.jsonl'))
    ref = os.path.join(a.bundle, 'eval_prompts', 'confirm_c2_reference.jsonl')
    if len(c1) != 120 or not os.path.exists(ref):
        raise SystemExit('the 2026-10-26 confirm add-on is missing from the bundle (eval_prompts/confirm_c1_direct_choice.jsonl, confirm_c2_reference.jsonl)')
    out_dir = os.path.join(a.root, 'work', ROUND, 'confirm', 'prompts')
    os.makedirs(out_dir, exist_ok=True)
    runs = [f'{f}-s{s}' for f in FAMILIES for s in a.seeds.split(',')]
    build, lines = {'round': ROUND, 'runs': {}}, []
    outs = f'work/outputs/{ROUND}-confirm'
    line = lambda out, prompts, adapter: '\t'.join([out, prompts, adapter, '0', '1', str(MAX_NEW)])  # noqa: E731
    lines.append(line(f'{outs}/B0/confirm_c1_direct_choice.jsonl', 'eval_prompts/confirm_c1_direct_choice.jsonl', '-'))
    for run in runs:
        v0_path = os.path.join(a.root, 'work', 'outputs', ROUND, run, 'test_V0.jsonl')
        v0 = {(o['task_id'], o.get('sample', 0)): o for o in read(v0_path)}
        recs, no_action = [], []
        for r in c1:
            o = v0.get((r['task_id'], 0))
            if o is None:
                raise SystemExit(f'{v0_path} has no answer for {r["task_id"]}')
            act, err = parse(o.get('raw'))
            if err:
                no_action.append(r['task_id'])
                continue
            recs.append(c2_record(r, act))
        path = os.path.join(out_dir, f'confirm_c2_v0_{run}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for x in recs:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
        build['runs'][run] = {'requests': len(recs), 'no_action': no_action,
                              'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest()}
        adapter = f'work/runs/{ROUND}/{run}/checkpoints/ckpt-4'
        lines.append(line(f'{outs}/{run}/confirm_c1_direct_choice.jsonl', 'eval_prompts/confirm_c1_direct_choice.jsonl', adapter))
        lines.append(line(f'{outs}/{run}/confirm_c2_v0.jsonl', path, adapter))
        lines.append(line(f'{outs}/{run}/confirm_c2_reference.jsonl', 'eval_prompts/confirm_c2_reference.jsonl', adapter))
    json.dump(build, open(os.path.join(a.root, 'work', ROUND, 'confirm', 'build.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
