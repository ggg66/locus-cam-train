"""Rewards of the v5 RL methods, computed from the outcome tables of training roots (reward_tables/train.jsonl.gz of the
v5 add-on, built by the benchmark side's export_v5.py). Stdlib only.

A table line: {root_id, task_id, family_id, two, names, rows: {"+r,+u,+b": {s, p, n, f, t}}} with
  s  status: satisfied / spatial_unmet / illegal_endpoint
  p, n  passed / evaluated constraint checks (n = 0 at an illegal endpoint)
  f  [legal, in_frame, size_ok, unoccluded, apart] (null where a field does not apply)
  t  per target, in the order of `names`: [height, margin, blocked, hits]; null at an illegal endpoint

Action reward (method 1): the action is read with the frozen scorer's parser (lcam.parse, last JSON); satisfied ->
`satisfied`, spatially unmet -> `partial_max` x passed/evaluated checks, illegal endpoint -> `illegal`, no valid action ->
`invalid`.
Plan reward (method 2, "propose, imagine, choose"): the answer first lists candidate moves, each with its imagined outcome
(first JSON with a "candidates" list; consequence_v4 fields per candidate), then the chosen move. Reward = action reward +
`imagination_weight` x the mean share of correctly imagined fields over the candidates whose endpoint is legal (an illegal
endpoint's outcome is trivial to state, so it earns nothing). No imagination credit when the move is invalid, is not one of
the listed candidates, or fewer than `min_legal_candidates` legal ones are listed: the reward pays for imagination the decision
can have used to compare moves, not for text next to it.
"""
import gzip
import json
import math

from .parse import parse

KEY = '%+d,%+d,%+d'
FIELDS = ('legal', 'in_frame', 'size_ok', 'unoccluded', 'apart')
QUANT = (('height', 0), ('margin', 1), ('blocked', 2), ('hits', 3))
DEFAULT = {'type': 'action', 'satisfied': 1.0, 'partial_max': 0.3, 'illegal': 0.0, 'invalid': -0.5,
           'imagination_weight': 0.5, 'tolerance': {'height': 0.03, 'margin': 0.03, 'blocked': 1, 'hits': 1}}


def load_tables(path, roots=None):
    out = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            if roots is None or t['root_id'] in roots:
                out[t['root_id']] = t
    return out


def settings(cfg):
    out = dict(DEFAULT)
    out.update({k: v for k, v in (cfg or {}).items() if k != 'tolerance'})
    out['tolerance'] = dict(DEFAULT['tolerance'], **((cfg or {}).get('tolerance') or {}))
    return out


def action_reward(raw, table, cfg):
    s = settings(cfg)
    act, err = parse(raw)
    if err:
        return s['invalid'], {'outcome': err, 'key': None}
    row = table['rows'][KEY % act]
    if row['s'] == 'satisfied':
        r = s['satisfied']
    elif row['s'] == 'illegal_endpoint':
        r = s['illegal']
    else:
        r = s['partial_max'] * row['p'] / max(row['n'], 1)
    return float(r), {'outcome': row['s'], 'key': KEY % act}


def top_level_json(text):
    """Every top-level {...} block that parses, in order."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text or ''):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    return out


def first_plan(raw):
    return next((o for o in top_level_json(raw) if isinstance(o.get('candidates'), list)), None)


def candidate_key(c):
    m = c.get('move') if isinstance(c, dict) else None
    if not (isinstance(m, list) and len(m) == 3 and all(isinstance(x, int) and not isinstance(x, bool) for x in m)):
        return None
    if m[0] not in (-2, -1, 0, 1, 2) or m[1] not in (-1, 0, 1, 2) or m[2] not in (-3, -2, -1, 0, 1, 2):
        return None
    return KEY % tuple(m)


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def imagination_accuracy(pred, row, names, tol):
    """Share of correctly imagined fields of one action's outcome."""
    legal = row['f'][0]
    hits = [pred.get('legal') is legal]
    if not legal:
        hits.append(pred.get('targets') == [])
        return sum(hits) / len(hits)
    for name, truth in zip(FIELDS[1:], row['f'][1:]):
        if truth is not None:
            hits.append(pred.get(name) is truth)
    targets = pred.get('targets') if isinstance(pred.get('targets'), list) else []
    targets = [t for t in targets if isinstance(t, dict)]
    by_name = {t.get('name'): t for t in targets}
    for j, (name, truth) in enumerate(zip(names, row['t'])):
        p = by_name.get(name) or (targets[j] if j < len(targets) else {})
        for field, idx in QUANT:
            v = p.get(field)
            hits.append(is_num(v) and abs(v - truth[idx]) <= tol[field] + 1e-9)
    return sum(hits) / len(hits)


def reward(raw, table, cfg):
    """(reward, info) for one sampled answer."""
    s = settings(cfg)
    r, info = action_reward(raw, table, s)
    if s['type'] == 'action':
        return r, info
    if s['type'] != 'imagact':
        raise ValueError(f'unknown reward type {s["type"]}')
    info['imagination'] = None
    if info['key'] is None:
        return r, info
    plan = first_plan(raw)
    cands = {}
    for c in (plan or {}).get('candidates', [])[:int(s.get('max_candidates', 6))]:
        k = candidate_key(c)
        if k is not None and k not in cands:
            cands[k] = c
    info['n_candidates'] = len(cands)
    info['chosen_in_candidates'] = info['key'] in cands
    legal = [k for k in cands if table['rows'][k]['f'][0]]
    acc = 0.0
    if info['chosen_in_candidates'] and len(legal) >= int(s.get('min_legal_candidates', 2)):
        acc = sum(imagination_accuracy(cands[k], table['rows'][k], table['names'], s['tolerance']) for k in legal) / len(legal)
    info['imagination'] = acc
    return r + s['imagination_weight'] * acc, info


def group_advantages(rewards, normalize_std=True, eps=1e-4):
    """GRPO advantages of one prompt's samples; all zero when every sample got the same reward."""
    n = len(rewards)
    mean = sum(rewards) / n
    if max(rewards) - min(rewards) < 1e-12:
        return [0.0] * n
    if not normalize_std:
        return [r - mean for r in rewards]
    sd = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    return [(r - mean) / (sd + eps) for r in rewards]
