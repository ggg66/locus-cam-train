"""Configs for the smoke test: the real configs with the model pointed at the tiny mock model and tiny budgets.

  python tests/mock/make_smoke_config.py <tiny model dir> <out config dir>
Writes the v1 configs into <out>/ and the configs of the round $LCAM_ROUND (default v2) into <out>/<round>/
(A_compute is calibrated by the smoke stage).
"""
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tiny, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
c = yaml.safe_load(open(os.path.join(REPO, 'configs', 'common.yaml')))
c['model'].update(id=os.path.abspath(tiny), revision=None)
c['trainable'].update(lora_r=8, lora_alpha=16)      # hidden 64 in the tiny model; rank 8 keeps the same layers eligible
c['optim']['effective_batch_size'] = 4
yaml.safe_dump(c, open(os.path.join(out, 'common.yaml'), 'w'), allow_unicode=True, sort_keys=False)
budgets = {'B1': {'primary_fixed_total': {'samples': 8}, 'secondary_matched': {'samples': 12}},
           'B2': {'primary_fixed_total': {'samples': 12}, 'secondary_matched': {'samples': 12, 'action_exposures': 8, 'geometry_exposures': 4}}}
for arm in ('B0', 'B1', 'B2'):
    a = yaml.safe_load(open(os.path.join(REPO, 'configs', f'{arm}.yaml')))
    if arm in budgets:
        a['budget'] = budgets[arm]
    yaml.safe_dump(a, open(os.path.join(out, f'{arm}.yaml'), 'w'), allow_unicode=True, sort_keys=False)
rnd = os.environ.get('LCAM_ROUND', 'v2')
v2 = os.path.join(out, rnd)
os.makedirs(v2, exist_ok=True)
c = yaml.safe_load(open(os.path.join(REPO, 'configs', rnd, 'common.yaml')))
c['model'].update(id=os.path.abspath(tiny), revision=None)
c['trainable'].update(lora_r=8, lora_alpha=16)
c['optim']['effective_batch_size'] = 4
yaml.safe_dump(c, open(os.path.join(v2, 'common.yaml'), 'w'), allow_unicode=True, sort_keys=False)
sys.path.insert(0, REPO)
from lcam.config import families  # noqa: E402
src = os.path.join(REPO, 'configs', rnd)
for f in sorted(x for x in os.listdir(src) if x.endswith('.yaml') and x not in ('common.yaml',)):
    c2 = yaml.safe_load(open(os.path.join(src, f)))
    if c2.get('exposures'):
        # tiny budgets: actions 8, any other numeric source 4; calibrated sources stay from_calibration
        c2['exposures'] = {k: v if v == 'from_calibration' else (8 if k in ('action_sft', 'geomact_v4') else 4) for k, v in c2['exposures'].items()}
    if c2.get('calibration'):
        # 12 records and an effective batch of 4: one micro-batch more or less is ~a quarter of AG's compute, so the real
        # 5 % tolerance cannot be met at this scale (at 64 x 15000 one batch is ~0.3 %). The smoke run only checks that
        # calibration, freezing and the calibrated families' training work; the real configs keep 0.05.
        c2['calibration']['tolerance'] = 1.0
    yaml.safe_dump(c2, open(os.path.join(v2, f), 'w'), allow_unicode=True, sort_keys=False)
print('families', families(v2))
print('smoke configs ->', out)
