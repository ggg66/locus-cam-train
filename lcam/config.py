"""Config loading: an arm file (B0/B1/B2.yaml) with `base: common.yaml` merged over the common settings."""
import os

import yaml


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path):
    with open(path, encoding='utf-8') as f:
        arm = yaml.safe_load(f)
    base = arm.pop('base', None)
    if base:
        with open(os.path.join(os.path.dirname(path), base), encoding='utf-8') as f:
            arm = merge(yaml.safe_load(f), arm)
    return arm


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
