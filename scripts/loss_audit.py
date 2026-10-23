"""What the v1 loss was, measured on the real model (one GPU, no training, no LoRA).

  python scripts/loss_audit.py --config configs/v2/common.yaml --bundle /data/locus-bundle --out $WORK/audit/loss_audit.json

Builds two micro-batches from train records, [action, geometry_sft] and [action, action], runs the model once with
labels (what v1 optimised: `out.loss`) and once without, and recomputes the loss from the logits with lcam.loss:
  - out.loss == token mean over the micro-batch      -> v1 objective as documented, no extra term
  - out.loss != that                                 -> something else was in v1's loss (e.g. a router aux loss);
                                                        the difference is reported
Also reports each sample's mean CE and learned tokens, and the weight its own loss had inside the v1 micro-batch.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam.data import load_exports  # noqa: E402
from lcam.encode import collate, encode  # noqa: E402
from lcam.loss import per_sample_ce  # noqa: E402
from lcam.model import load_model, load_processor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = C.load(a.config)
    processor = load_processor(cfg)
    model, info = load_model(cfg)
    model.eval()
    recs = load_exports(a.bundle, 'train', ['action_sft', 'geometry_sft'])
    groups = {'action+geometry_v1': [recs['action_sft'][0], recs['geometry_sft'][0]], 'action+action': recs['action_sft'][1:3]}
    dev = next(model.parameters()).device
    res = {'model_class': info['class'], 'output_router_logits': getattr(model.config, 'output_router_logits', None),
           'router_aux_loss_coef': getattr(model.config, 'router_aux_loss_coef', getattr(getattr(model.config, 'text_config', None), 'router_aux_loss_coef', None)),
           'groups': {}}
    for name, rs in groups.items():
        b = collate([encode(processor, a.bundle, r, cfg['data']['max_length']) for r in rs], processor.tokenizer.pad_token_id)
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.no_grad():
            out_l = model(**b)
            labels = b.pop('labels')
            out = model(**b)
        sums, counts = per_sample_ce(out.logits, labels)
        tok_mean = float(sums.sum() / counts.sum())
        res['groups'][name] = {'ids': [r['id'] for r in rs], 'hf_out_loss': float(out_l.loss), 'token_mean_recomputed': tok_mean,
                               'difference': float(out_l.loss) - tok_mean, 'aux_loss_returned': getattr(out_l, 'aux_loss', None) is not None,
                               'per_sample_mean_ce': [float(s / c) for s, c in zip(sums, counts)], 'learned_tokens': counts.tolist(),
                               'v1_weight_of_own_mean_in_microbatch': [round(int(c) / int(counts.sum()), 4) for c in counts]}
    res['v1_loss_is_token_mean'] = all(abs(g['difference']) < 1e-3 for g in res['groups'].values())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
