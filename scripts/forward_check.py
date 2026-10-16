"""Forward / loss check on real batches with the real model. No optimizer is created and no step is taken.

  python scripts/forward_check.py --config configs/common.yaml --bundle /data/locus-bundle --batches 4 --micro-batch 4
  # optional, separately authorised: one backward pass to measure peak memory (gradients are discarded)
  LCAM_ALLOW_BACKWARD_PROBE=1 python scripts/forward_check.py ... --backward-probe

Reports per batch: loss, learned-token count, sequence length, peak GPU memory; with the probe also the peak
memory after backward under gradient checkpointing (this is what tells whether training fits).
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
from lcam.model import load_model, load_processor, trainable_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--micro-batch', type=int, default=4)
    ap.add_argument('--backward-probe', action='store_true')
    a = ap.parse_args()
    probe = a.backward_probe
    if probe and os.environ.get('LCAM_ALLOW_BACKWARD_PROBE') != '1':
        raise SystemExit('backward probe needs LCAM_ALLOW_BACKWARD_PROBE=1 (separate authorisation)')
    cfg = C.load(a.config)
    processor = load_processor(cfg)
    model, attn = load_model(cfg, for_training=probe)
    recs = load_exports(a.bundle, 'train', ['action_sft', 'geometry_sft'])
    pool = [r for pair in zip(recs['action_sft'], recs['geometry_sft']) for r in pair]
    res = {'attention': attn, 'params': trainable_summary(model), 'micro_batch': a.micro_batch, 'batches': []}
    pad = processor.tokenizer.pad_token_id
    for b in range(a.batches):
        chunk = pool[b * a.micro_batch:(b + 1) * a.micro_batch]
        enc = [encode(processor, a.bundle, r, cfg['data']['max_length']) for r in chunk]
        batch = {k: v.cuda() for k, v in collate(enc, pad).items()}
        torch.cuda.reset_peak_memory_stats()
        if probe:
            model.train()
            out = model(**batch)
            out.loss.backward()
            model.zero_grad(set_to_none=True)  # gradients discarded; no optimizer exists
        else:
            model.eval()
            with torch.no_grad():
                out = model(**batch)
        res['batches'].append({'ids': [r['id'] for r in chunk], 'loss': round(out.loss.item(), 4), 'seq_len': int(batch['input_ids'].shape[1]),
                               'learned_tokens': int((batch['labels'] != -100).sum()), 'peak_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)})
        print(json.dumps(res['batches'][-1], ensure_ascii=False), flush=True)
    res['mode'] = 'forward+backward probe (no optimizer)' if probe else 'forward only'
    os.makedirs(os.path.join(a.bundle, 'checks'), exist_ok=True)
    json.dump(res, open(os.path.join(a.bundle, 'checks', f'forward_check{"-backward" if probe else ""}.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
