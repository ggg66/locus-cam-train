"""Score the v5 development runs (stdlib only): held-out development families and seen validation families.

  python scripts/score_dev.py --bundle $BUNDLE --outputs $LOCUS_ROOT/work/outputs/v5 --out $LOCUS_ROOT/work/v5/dev_summary.json

Each run directory <outputs>/<run> (e.g. base-s0, m1_rl-s0) holds dev_heldfam*.jsonl (V0 answers on the training roots of
the development families; scored against reward_tables/train.jsonl.gz) and dev_seen*.jsonl (validation V0 answers of the
other families; scored against validation_tables). The action is read with the frozen parser (lcam.parse).
For every method (run name without -s<seed>), success rates are averaged over its seeds and compared with `base`:
difference of means and a 95 % interval from resampling roots (seed-averaged per root). Writes JSON and a markdown table.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import rewards as R  # noqa: E402
from lcam.parse import outcome  # noqa: E402

KEY = '%+d,%+d,%+d'


def answers(paths):
    out = {}
    for p in paths:
        for line in open(p, encoding='utf-8'):
            if line.strip():
                o = json.loads(line)
                out.setdefault(o['task_id'], o)
    return out


def boot(diffs, n=10000, seed=0):
    rng = random.Random(seed)
    bs = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(n))
    return [round(bs[int(.025 * n)], 4), round(bs[int(.975 * n) - 1], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--baseline', default='base')
    a = ap.parse_args()
    held = [json.loads(l) for l in open(os.path.join(a.bundle, 'eval_prompts', 'dev_heldfam_V0.jsonl'), encoding='utf-8')]
    seen = [json.loads(l) for l in open(os.path.join(a.bundle, 'eval_prompts', 'dev_seen_V0.jsonl'), encoding='utf-8')]
    tables = R.load_tables(os.path.join(a.bundle, 'reward_tables', 'train.jsonl.gz'), roots={p['root_id'] for p in held})
    vindex = json.load(open(os.path.join(a.bundle, 'validation_tables', 'index.json')))
    runs = {}
    for d in sorted(glob.glob(os.path.join(a.outputs, '*'))):
        name = os.path.basename(d)
        h = answers(sorted(glob.glob(os.path.join(d, 'dev_heldfam*.jsonl'))))
        v = answers(sorted(glob.glob(os.path.join(d, 'dev_seen*.jsonl'))))
        if not h and not v:
            continue
        rec = {'held': {}, 'seen': {}, 'n_held': len(h), 'n_seen': len(v), 'outcomes_held': defaultdict(int), 'per_family_held': defaultdict(lambda: [0, 0])}
        for p in held:
            o = h.get(p['task_id'])
            act, err = R.parse(o['raw']) if o else (None, 'missing')
            st = err or tables[p['root_id']]['rows'][KEY % act]['s']
            rec['outcomes_held'][st] += 1
            rec['held'][p['root_id']] = int(st == 'satisfied')
            rec['per_family_held'][p['family_id']][0] += int(st == 'satisfied')
            rec['per_family_held'][p['family_id']][1] += 1
        for p in seen:
            o = v.get(p['task_id'])
            ok = 0
            if o:
                table = json.load(open(os.path.join(a.bundle, 'validation_tables', vindex[p['task_id']] + '.json')))
                ok = int(outcome(o['raw'], table) == 'satisfied')
            rec['seen'][p['root_id']] = ok
        runs[name] = rec
    methods = defaultdict(list)
    for name in runs:
        methods[re.sub(r'-s\d+$', '', name)].append(name)
    def per_root(names, part):
        acc = defaultdict(list)
        for n in names:
            for root, ok in runs[n][part].items():
                acc[root].append(ok)
        return {k: sum(v) / len(v) for k, v in acc.items()}
    out = {'baseline': a.baseline, 'n_held_prompts': len(held), 'n_seen_prompts': len(seen), 'runs': {}, 'methods': {}}
    for name, r in runs.items():
        out['runs'][name] = {'held_success': round(sum(r['held'].values()) / len(held), 4), 'seen_success': round(sum(r['seen'].values()) / len(seen), 4),
                             'answers_held': r['n_held'], 'answers_seen': r['n_seen'], 'outcomes_held': dict(r['outcomes_held']),
                             'per_family_held': {f: round(s / n, 4) for f, (s, n) in r['per_family_held'].items()}}
    base = methods.get(a.baseline, [])
    for m, names in sorted(methods.items()):
        e = {'runs': names}
        for part in ('held', 'seen'):
            pr = per_root(names, part)
            e[f'{part}_success'] = round(sum(pr.values()) / len(pr), 4)
            if base and m != a.baseline:
                pb = per_root(base, part)
                keys = sorted(set(pr) & set(pb))
                d = [pr[k] - pb[k] for k in keys]
                e[f'{part}_minus_{a.baseline}'] = round(sum(d) / len(d), 4)
                e[f'{part}_ci95'] = boot(d)
        out['methods'][m] = e
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    lines = [f'| method | runs | held-out dev families | vs {a.baseline} (95% CI) | seen families | vs {a.baseline} |', '|---|---:|---:|---|---:|---|']
    for m, e in out['methods'].items():
        ci = f'{100 * e["held_minus_" + a.baseline]:+.1f} [{100 * e["held_ci95"][0]:+.1f}, {100 * e["held_ci95"][1]:+.1f}]' if f'held_minus_{a.baseline}' in e else '—'
        sv = f'{100 * e["seen_minus_" + a.baseline]:+.1f}' if f'seen_minus_{a.baseline}' in e else '—'
        lines.append(f'| {m} | {len(e["runs"])} | {100 * e["held_success"]:.1f} | {ci} | {100 * e["seen_success"]:.1f} | {sv} |')
    open(os.path.splitext(a.out)[0] + '.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
