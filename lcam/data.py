"""Training data: explicit export files (never a glob), the B1/B2 mixtures and their exposure accounting.

Export records come from benchmark-v1-pretrain/training_exports/<split>/*.jsonl (see its export.py). Each record
has `images` (paths relative to the data bundle root), `system`, `prompt`, `target`, and for geometry records
`loss_mask_paths`. Outcome records and the privileged geometry file are never used by B1/B2.
"""
import json
import os
import random

ALLOWED = {'action_sft', 'geometry_sft'}          # B1/B2 never read outcome_records or geometry_sft_privileged


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


def messages(rec):
    """Qwen2.5-VL chat messages: system, user (images in fixed order, then the prompt), assistant target."""
    user = [{'type': 'image', 'image': p} for p in rec['images']] + [{'type': 'text', 'text': rec['prompt']}]
    return [{'role': 'system', 'content': [{'type': 'text', 'text': rec['system']}]},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': rec['target']}]}]
