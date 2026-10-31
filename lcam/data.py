"""Training data: explicit export files (never a glob), the B1/B2 mixtures and their exposure accounting.

Export records come from benchmark-v1-pretrain/training_exports/<split>/*.jsonl (see its export.py). Each record
has `images` (paths relative to the data bundle root), `system`, `prompt`, `target`, and for geometry records
`loss_mask_paths`. Outcome records and the privileged geometry file are never used.
geometry_sft_v2 (export_v2.py): fields the images cannot support carry fixed defaults instead of their true values,
`hidden_paths` lists them and `loss_mask_paths` is empty, so every target token is learned.
"""
import json
import os
import random

ALLOWED = {'action_sft', 'geometry_sft', 'geometry_sft_v2',     # never outcome_records or geometry_sft_privileged
           'consequence_v4', 'geomact_v4', 'nextview_v4'}   # v4 supervision study (training_exports/export_v4.py)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_exports(bundle, split, kinds):
    out = {}
    for k in kinds:
        if k not in ALLOWED:
            raise ValueError(f'{k} is not a B1/B2 training source')
        recs = read_jsonl(os.path.join(bundle, 'training_exports', split, f'{k}.jsonl'))
        for r in recs:
            if r['source_split'] != split:
                raise ValueError(f'{r["id"]}: source_split {r["source_split"]} != {split}')
        out[k] = recs
    return out


def check_isolation(bundle, split, recs):
    """No root / task / pair group of the training records may appear in another split's manifest."""
    path = os.path.join(bundle, 'splits_index.json')
    others = json.load(open(path))  # {"root_or_task_id": "split", ...} built by the bundle packer for every split
    bad = [r['id'] for rs in recs.values() for r in rs
           for k in (r['root_id'], r['task_id'], r['pair_group_id']) if others.get(k, split) != split]
    if bad:
        raise ValueError(f'{len(bad)} records leak across splits, e.g. {bad[:3]}')


def schedule(recs, mix, n_samples, seed):
    """Sample order for one run: each source is consumed in reshuffled passes (no repeats inside a pass), and the
    sources are interleaved by a seeded shuffle. Returns a list of (kind, record_index) and the exposure counts."""
    rng = random.Random(seed)
    kinds = sorted(mix)
    counts = {k: int(round(mix[k] * n_samples)) for k in kinds}
    counts[kinds[0]] += n_samples - sum(counts.values())
    seq = []
    for k in kinds:
        n, pool = counts[k], []
        while len(pool) < n:
            order = list(range(len(recs[k])))
            rng.shuffle(order)
            pool.extend(order)
        seq.extend((k, i) for i in pool[:n])
    rng.shuffle(seq)
    return seq, counts


def kind_pool(n_records, n, seed, kind):
    """n record indices of one source in reshuffled passes (no repeat inside a pass), from its own seeded stream."""
    rng = random.Random(f'{seed}|{kind}')
    pool = []
    while len(pool) < n:
        order = list(range(n_records))
        rng.shuffle(order)
        pool.extend(order)
    return pool[:n]


def schedule_v2(recs, exposures, seed, anchor='action_sft', streams=None):
    """Sample order of one v2 run: (kind, record_index) list and exposure counts.
    Every source draws from its own seeded stream, so the anchor (action) sequence of a seed does not depend on
    what else is mixed in: AG's action samples are, in order, exactly A-base's, and A-compute's begin with them.
    The other sources go to seeded positions between them.
    `streams` maps a kind to the stream it draws from (default: its own name). A source whose records are aligned
    one-to-one with action_sft (geomact_v4) can draw from the action_sft stream, so its roots come in A-base's order."""
    streams = streams or {}
    pools = {k: kind_pool(len(recs[k]), int(n), seed, streams.get(k, k)) for k, n in exposures.items()}
    seq = [(anchor, i) for i in pools[anchor]]
    others = [(k, i) for k in sorted(exposures) if k != anchor for i in pools[k]]
    if others:
        rng = random.Random(f'{seed}|mix|' + ','.join(sorted(exposures)))
        rng.shuffle(others)
        total = len(seq) + len(others)
        slots = set(rng.sample(range(total), len(others)))
        a, o = iter(seq), iter(others)
        seq = [next(o) if j in slots else next(a) for j in range(total)]
    return seq, {k: len(v) for k, v in pools.items()}


def step_layout(n, eff, micro, ws):
    """Who processes which schedule position. Optimizer step s owns positions [s*eff, (s+1)*eff) whatever the world
    size; inside a step rank r takes every ws-th position from r and cuts them into micro-batches. So the samples of
    every step, and hence the v2 objective, are the same on 1 GPU and on 8. The last step may be short (a rank may
    then get no micro-batch, and contributes a zero gradient)."""
    if eff % (micro * ws):
        raise ValueError(f'effective batch {eff} is not a multiple of micro-batch x world size ({micro} x {ws})')
    steps = []
    for s0 in range(0, n, eff):
        block = list(range(s0, min(s0 + eff, n)))
        ranks = []
        for r in range(ws):
            mine = block[r::ws]
            ranks.append([mine[i:i + micro] for i in range(0, len(mine), micro)])
        steps.append({'members': block, 'ranks': ranks})
    return steps


def messages(rec):
    """Chat messages (Qwen chat format): system, user (images in fixed order, then the prompt), assistant target."""
    user = [{'type': 'image', 'image': p} for p in rec['images']] + [{'type': 'text', 'text': rec['prompt']}]
    return [{'role': 'system', 'content': [{'type': 'text', 'text': rec['system']}]},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': rec['target']}]}]
