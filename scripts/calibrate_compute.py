"""Fix A-compute's action budget so that its training compute matches AG's (before any v2 run). Stdlib only.

  python scripts/calibrate_compute.py --stats $WORK/audit/encode_stats-train.json --micro-batch 2 --world-size 8 \
      [--gpu-seconds AG=<metrics.jsonl> A_base=<metrics.jsonl> ...] [--freeze]

For each seed of configs/v2/common.yaml: padded tokens of the AG schedule (v2 layout, the given micro-batch and
world size); the smallest action-only budget whose schedule costs at least as much; the seeds' budgets are averaged
and rounded up to a multiple of the effective batch, and one number is used for every seed. Also reported, not used:
the budget a real-token (no padding) match would give, and - if short-run metrics are given - GPU seconds per
optimizer step of each family, so the deviation of the chosen measure from wall-clock cost is on record.
Writes <config dir>/calibration.json (default configs/v2/); --freeze marks it frozen, after which train_v2.py accepts A_compute.yaml.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lcam import config as C  # noqa: E402
from lcam.costs import match_actions, schedule_cost  # noqa: E402
from lcam.data import schedule_v2  # noqa: E402


def sec_per_step(path):
    ms = [json.loads(l) for l in open(path) if l.strip()]
    if len(ms) < 3:
        raise SystemExit(f'{path}: need at least 3 steps')
    ms = ms[1:]                      # the first step includes warm-up work
    return round((ms[-1]['train_seconds'] - ms[0]['train_seconds']) / (len(ms) - 1), 3), round(sum(m['tokens_padded'] for m in ms) / len(ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', required=True)
    ap.add_argument('--micro-batch', type=int, required=True)
    ap.add_argument('--world-size', type=int, required=True)
    ap.add_argument('--gpu-seconds', nargs='*', default=[])
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--config-dir', default=os.path.join(HERE, 'configs', os.environ.get('LCAM_ROUND', 'v2')))
    a = ap.parse_args()
    cdir = a.config_dir
    ag, acomp = C.load(os.path.join(cdir, 'AG.yaml')), C.load(os.path.join(cdir, 'A_compute.yaml'))
    stats = json.load(open(a.stats))['kinds']
    for k in ag['exposures']:
        if k not in stats:
            raise SystemExit(f'{a.stats} has no {k}')
    eff = int(ag['optim']['effective_batch_size'])
    metric = acomp['calibration']['metric'].replace('_tokens', '')
    per_seed = {}
    for seed in ag['optim']['seeds']:
        seq, _ = schedule_v2({k: stats[k] for k in ag['exposures']}, ag['exposures'], seed)
        c_ag = schedule_cost(seq, stats, eff, a.micro_batch, a.world_size)
        n_pad, c_pad = match_actions(len(stats['action_sft']), stats, c_ag['padded'], seed, eff, a.micro_batch, a.world_size, 'padded')
        n_real, _ = match_actions(len(stats['action_sft']), stats, c_ag['real'], seed, eff, a.micro_batch, a.world_size, 'real')
        per_seed[seed] = {'AG': c_ag, 'actions_padded_match': n_pad, 'actions_real_match': n_real}
    key = 'actions_padded_match' if metric == 'padded' else 'actions_real_match'
    n = int(math.ceil(sum(v[key] for v in per_seed.values()) / len(per_seed) / eff) * eff)
    check = {}
    for seed, v in per_seed.items():
        seq, _ = schedule_v2({'action_sft': stats['action_sft']}, {'action_sft': n}, seed)
        c = schedule_cost(seq, stats, eff, a.micro_batch, a.world_size)
        check[seed] = {'A_compute': c, 'deviation_padded': round(c['padded'] / v['AG']['padded'] - 1, 4),
                       'deviation_real': round(c['real'] / v['AG']['real'] - 1, 4)}
    tol = float(acomp['calibration']['tolerance'])
    worst = max(abs(x[f'deviation_{metric}']) for x in check.values())
    gpu = {}
    for item in a.gpu_seconds:
        fam, path = item.split('=', 1)
        gpu[fam] = dict(zip(('seconds_per_step', 'padded_tokens_per_step'), sec_per_step(path)))
    out = {'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'frozen': bool(a.freeze), 'metric': f'{metric}_tokens',
           'micro_batch': a.micro_batch, 'world_size': a.world_size, 'effective_batch_size': eff, 'tolerance': tol,
           'within_tolerance': worst <= tol, 'worst_deviation': worst,
           'exposures': {'A_compute': {'action_sft': n}}, 'per_seed': per_seed, 'check': check, 'gpu_seconds_short_runs': gpu,
           'note': 'one action budget for all seeds; the real-token match and GPU seconds are reported, not used'}
    if a.freeze and not out['within_tolerance']:
        raise SystemExit(f'deviation {worst} > tolerance {tol}; not frozen')
    json.dump(out, open(os.path.join(cdir, 'calibration.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_seed', 'check')}, indent=1, default=str))


if __name__ == '__main__':
    main()
