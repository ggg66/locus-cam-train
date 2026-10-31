"""Fix the budget of every calibrated family so that its training compute matches AG's (before any run). Stdlib only.

v2 / v3_gemma: A_compute's action exposures. v4: the auxiliary exposures of A_cons and A_nextview (15000 actions fixed)
and the geometry-then-action exposures of A_geomact. Any config whose exposures contain `from_calibration` is calibrated.

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
from lcam.costs import match_kind, schedule_cost  # noqa: E402
from lcam.data import schedule_v2  # noqa: E402


def sec_per_step(path):
    ms = [json.loads(l) for l in open(path) if l.strip()]
    if len(ms) < 3:
        raise SystemExit(f'{path}: need at least 3 steps')
    ms = ms[1:]                      # the first step includes warm-up work
    return round((ms[-1]['train_seconds'] - ms[0]['train_seconds']) / (len(ms) - 1), 3), round(sum(m['tokens_padded'] for m in ms) / len(ms))


def calibrated_families(cdir):
    """{family: (config, vary kind, fixed exposures)} for every family with a `from_calibration` exposure."""
    out = {}
    for f in C.families(cdir):
        c = C.load(os.path.join(cdir, f'{f}.yaml'))
        vary = [k for k, v in (c.get('exposures') or {}).items() if v == 'from_calibration']
        if len(vary) > 1:
            raise SystemExit(f'{f}: only one exposure can come from calibration')
        if vary:
            out[f] = (c, vary[0], {k: int(v) for k, v in c['exposures'].items() if k != vary[0]})
    return out


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
    ag = C.load(os.path.join(cdir, 'AG.yaml'))     # the compute every calibrated family matches (reference_only in v4)
    stats = json.load(open(a.stats))['kinds']
    fams = calibrated_families(cdir)
    for k in set(ag['exposures']) | {k for c, v, fx in fams.values() for k in [v, *fx]}:
        if k not in stats:
            raise SystemExit(f'{a.stats} has no {k}')
    eff = int(ag['optim']['effective_batch_size'])
    ag_seq = lambda seed: schedule_v2({k: stats[k] for k in ag['exposures']}, ag['exposures'], seed,  # noqa: E731
                                      anchor=ag.get('anchor', 'action_sft'), streams=ag.get('pool_streams'))[0]
    exposures, per_seed, check, worst, tols = {}, {}, {}, {}, {}
    for fam, (cfg, vary, fixed) in fams.items():
        if cfg['calibration'].get('match', 'AG') != 'AG':
            raise SystemExit(f'{fam}: calibration can only match AG')
        metric = cfg['calibration']['metric'].replace('_tokens', '')
        kw = {'anchor': cfg.get('anchor', 'action_sft'), 'streams': cfg.get('pool_streams')}
        ps = {}
        for seed in ag['optim']['seeds']:
            c_ag = schedule_cost(ag_seq(seed), stats, eff, a.micro_batch, a.world_size)
            n_pad, _ = match_kind(stats, fixed, vary, c_ag['padded'], seed, eff, a.micro_batch, a.world_size, 'padded', **kw)
            n_real, _ = match_kind(stats, fixed, vary, c_ag['real'], seed, eff, a.micro_batch, a.world_size, 'real', **kw)
            ps[seed] = {'AG': c_ag, f'{vary}_padded_match': n_pad, f'{vary}_real_match': n_real}
        key = f'{vary}_{metric}_match'
        total = sum(fixed.values())
        n = int(math.ceil((sum(v[key] for v in ps.values()) / len(ps) + total) / eff) * eff) - total   # whole optimizer steps
        ck = {}
        for seed, v in ps.items():
            ex = dict(fixed, **{vary: n})
            seq, _ = schedule_v2({k: stats[k] for k in ex}, ex, seed, **kw)
            c = schedule_cost(seq, stats, eff, a.micro_batch, a.world_size)
            ck[seed] = {fam: c, 'deviation_padded': round(c['padded'] / v['AG']['padded'] - 1, 4),
                        'deviation_real': round(c['real'] / v['AG']['real'] - 1, 4)}
        exposures[fam], per_seed[fam], check[fam] = {vary: n}, ps, ck
        tols[fam] = float(cfg['calibration']['tolerance'])
        worst[fam] = max(abs(x[f'deviation_{metric}']) for x in ck.values())
    gpu = {}
    for item in a.gpu_seconds:
        fam, path = item.split('=', 1)
        if os.path.exists(path):
            gpu[fam] = dict(zip(('seconds_per_step', 'padded_tokens_per_step'), sec_per_step(path)))
    ok = all(worst[f] <= tols[f] for f in fams)
    out = {'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'frozen': bool(a.freeze), 'metric': 'per family (configs)',
           'micro_batch': a.micro_batch, 'world_size': a.world_size, 'effective_batch_size': eff, 'tolerance': tols,
           'within_tolerance': ok, 'worst_deviation': worst, 'exposures': exposures, 'per_seed': per_seed, 'check': check,
           'gpu_seconds_short_runs': gpu, 'note': 'one budget per family for all seeds, matched to AG; real-token match and GPU seconds reported, not used'}
    if a.freeze and not ok:
        raise SystemExit(f'deviation {worst} > tolerance {tols}; not frozen')
    json.dump(out, open(os.path.join(cdir, 'calibration.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_seed', 'check')}, indent=1, default=str))


if __name__ == '__main__':
    main()
