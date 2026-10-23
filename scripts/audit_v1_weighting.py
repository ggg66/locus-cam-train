"""How much weight the v1 objective actually gave each kind of sample (audit of the v1 runs; stdlib only).

  python scripts/audit_v1_weighting.py --stats $WORK/audit/encode_stats-train.json [--world-size 8 --micro-batch 2]

Replays the v1 schedules (lcam.data.schedule, configs/B1.yaml and B2.yaml, both budgets, every seed) with the
layout of the v1 server code (rank r takes seq[r::ws], consecutive micro-batches, accumulation eff/(micro*ws)) and
the v1 loss (token mean per micro-batch, averaged over the micro-batches of a step). In that objective sample i's
own mean token loss enters with weight learned_tokens_i / learned_tokens_of_its_micro-batch / micro-batches_per_step.
Reported per run: each kind's share of the total weight, the total action weight relative to B1-primary (same
seed), and the spread of single-sample weights; with the v2 objective every sample weighs 1/64 of its step.
Output: printed JSON (keep it with the audit, docs/v2/training_audit.md).
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lcam import config as C  # noqa: E402
from lcam.data import schedule  # noqa: E402


def v1_weights(seq, stats, eff, micro, ws):
    """Total objective weight per kind and the list of single-sample weights (v1 layout and loss)."""
    accum = eff // (micro * ws)
    per_rank = len(seq) // ws
    slices = [seq[r::ws][:per_rank] for r in range(ws)]
    per_step = accum * ws
    tot, single = {}, []
    for r in range(ws):
        mine = slices[r]
        for i in range(0, len(mine), micro):
            mb = mine[i:i + micro]
            ls = [stats[k][j][1] for k, j in mb]
            for (k, j), l in zip(mb, ls):
                w = l / sum(ls) / per_step
                tot[k] = tot.get(k, 0.0) + w
                single.append((k, w))
    return tot, single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', required=True)
    ap.add_argument('--world-size', type=int, default=8)
    ap.add_argument('--micro-batch', type=int, default=2)
    a = ap.parse_args()
    stats = json.load(open(a.stats))['kinds']
    out = {}
    base = {}
    for arm in ('B1', 'B2'):
        cfg = C.load(os.path.join(HERE, 'configs', f'{arm}.yaml'))
        eff = int(cfg['optim']['effective_batch_size'])
        for budget in ('primary_fixed_total', 'secondary_matched'):
            mix, n = C.mix_and_samples(cfg, budget)
            for seed in cfg['optim']['seeds']:
                seq, counts = schedule({k: stats[k] for k in mix}, mix, n, seed)
                tot, single = v1_weights(seq, stats, eff, a.micro_batch, a.world_size)
                total = sum(tot.values())
                key = f'{arm}-{budget}-s{seed}'
                if arm == 'B1' and budget == 'primary_fixed_total':
                    base[seed] = tot['action_sft']
                act = [w for k, w in single if k == 'action_sft']
                out[key] = {'exposures': counts, 'weight_share': {k: round(v / total, 4) for k, v in tot.items()},
                            'action_weight_vs_B1_primary': None, '_action_total': tot['action_sft'],
                            'action_sample_weight_x64': {'median': round(statistics.median(act) * 64, 3), 'min': round(min(act) * 64, 3),
                                                         'max': round(max(act) * 64, 3)}}
    for key, v in out.items():
        seed = int(key.rsplit('-s', 1)[1])
        v['action_weight_vs_B1_primary'] = round(v.pop('_action_total') / base[seed], 4)
    print(json.dumps({'layout': {'world_size': a.world_size, 'micro_batch': a.micro_batch},
                      'note': 'action_sample_weight_x64 = 1.0 means the v2 weight (1/64 of a step)', 'runs': out}, indent=1))


if __name__ == '__main__':
    main()
