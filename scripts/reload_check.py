"""A saved v2 adapter reloads into the same model: recompute the checkpoint's validation-probe losses.

  python scripts/reload_check.py --config configs/v2/common.yaml --bundle /data/locus-bundle \
      --ckpt runs/v2-AG-s0-short/checkpoints/ckpt-1 [--max-abs 2e-3]

train_v2.py stores, at every checkpoint, the eval-mode mean CE of a fixed set of validation records (probe_ids,
probe_ce in exposures.json). This loads base weights + adapter from disk, recomputes them and compares.
Exit code 1 when any probe differs by more than --max-abs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam.data import load_exports  # noqa: E402
from lcam.model import load_model, load_processor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--max-abs', type=float, default=2e-3)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from train_v2 import probe_losses
    cfg = C.load(a.config)
    ex = json.load(open(os.path.join(a.ckpt, 'exposures.json')))
    recs = load_exports(a.bundle, 'validation', ['action_sft', 'geometry_sft_v2'])
    by_id = {r['id']: r for rs in recs.values() for r in rs}
    probe = [by_id[i] for i in ex['probe_ids']]
    processor = load_processor(cfg)
    model, _ = load_model(cfg, adapter=a.ckpt)
    now = probe_losses(model, processor, a.bundle, probe, cfg['data']['max_length'], next(model.parameters()).device)
    diff = [abs(x - y) for x, y in zip(now, ex['probe_ce'])]
    res = {'ckpt': a.ckpt, 'saved': ex['probe_ce'], 'reloaded': now, 'max_abs_diff': max(diff), 'pass': max(diff) <= a.max_abs}
    print(json.dumps(res, indent=1))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
