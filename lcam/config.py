"""Config loading: an arm file (B0/B1/B2.yaml, or v2/A_base|AG|A_compute.yaml) with `base: common.yaml` merged over
the common settings. Strings may name environment variables (`${LCAM_MODEL}`); they are expanded on load."""
import json
import os

import yaml


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def expand(x):
    if isinstance(x, dict):
        return {k: expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [expand(v) for v in x]
    return os.path.expandvars(x) if isinstance(x, str) else x


def load(path):
    with open(path, encoding='utf-8') as f:
        arm = yaml.safe_load(f)
    base = arm.pop('base', None)
    if base:
        with open(os.path.join(os.path.dirname(path), base), encoding='utf-8') as f:
            arm = merge(yaml.safe_load(f), arm)
    return expand(arm)


def families(config_dir):
    """Trained run families of a round directory, in a fixed order: every <family>.yaml with a run_family that is not
    marked reference_only (a reference such as v4's AG.yaml only defines the compute to match)."""
    order = ['A_base', 'AG', 'A_compute']
    out = []
    for f in sorted(os.listdir(config_dir)):
        if f.endswith('.yaml') and f != 'common.yaml':
            c = load(os.path.join(config_dir, f))
            if c.get('run_family') and not c.get('reference_only'):
                out.append(c['run_family'])
    return sorted(out, key=lambda x: (order.index(x) if x in order else len(order), x))


def mix_and_samples(cfg, budget):
    """(mixture {kind: share}, total samples) for budget 'primary_fixed_total' or 'secondary_matched'."""
    mix = cfg.get('mix')
    if not mix:
        raise ValueError(f'arm {cfg.get("arm")} has no training mixture')
    samples = int(cfg['budget'][budget]['samples'])
    if budget == 'secondary_matched' and cfg['arm'] == 'B2':
        b = cfg['budget'][budget]
        mix = {'action_sft': b['action_exposures'] / samples, 'geometry_sft': b['geometry_exposures'] / samples}
    return mix, samples


def exposures(cfg, config_path):
    """v2 budget: {kind: exposures}. A value `from_calibration` is read from calibration.json next to the config,
    which must be frozen (written by scripts/calibrate_compute.py --freeze before any v2 training run)."""
    ex = cfg.get('exposures')
    if not ex:
        raise ValueError(f'{config_path}: v2 configs give `exposures`; v1 configs use --budget with scripts/train.py')
    out = {}
    for k, v in ex.items():
        if v == 'from_calibration':
            path = os.path.join(os.path.dirname(config_path), 'calibration.json')
            if not os.path.exists(path):
                raise SystemExit(f'{path} missing: run scripts/calibrate_compute.py first')
            cal = json.load(open(path))
            if not cal.get('frozen'):
                raise SystemExit(f'{path} is not frozen')
            v = cal['exposures'][cfg['run_family']][k]
        out[k] = int(v)
    return out
