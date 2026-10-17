"""Configs for the smoke test: the real configs with the model pointed at the tiny mock model and tiny budgets.

  python tests/mock/make_smoke_config.py <tiny model dir> <out config dir>
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
print('smoke configs ->', out)
