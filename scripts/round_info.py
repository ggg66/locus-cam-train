"""What a training round consists of, for the shell scripts (no torch).

  python scripts/round_info.py configs/<round> <key>

Keys: families (trained run families), calibrated (families with a from_calibration exposure), plain (the others),
kinds (every training source named by the round's configs, reference configs included), equiv_family (AG if trained,
else the first family), short_families (the first two families), lora_on, seeds, hf_id, revision, model_type.
Prints space-separated values.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402


def info(cdir):
    fams = C.families(cdir)
    cfgs = {f[:-5]: C.load(os.path.join(cdir, f)) for f in sorted(os.listdir(cdir)) if f.endswith('.yaml') and f != 'common.yaml'}
    cal = [f for f in fams if 'from_calibration' in (cfgs[f].get('exposures') or {}).values()]
    common = C.load(os.path.join(cdir, 'common.yaml'))
    kinds = sorted({k for c in cfgs.values() for k in (c.get('exposures') or {})})
    return {'families': fams, 'calibrated': cal, 'plain': [f for f in fams if f not in cal], 'kinds': kinds,
            'equiv_family': ['AG' if 'AG' in fams else fams[0]], 'short_families': fams[:2],
            'lora_on': list(common['trainable']['lora_on']), 'seeds': [str(s) for s in common['optim']['seeds']],
            'hf_id': [common['model'].get('hf_id', '')], 'revision': [str(common['model'].get('revision', ''))],
            'model_type': [common['model'].get('expect_model_type', '')]}


if __name__ == '__main__':
    print(' '.join(info(sys.argv[1])[sys.argv[2]]))
