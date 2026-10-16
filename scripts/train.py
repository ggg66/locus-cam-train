"""LoRA SFT for B1 / B2. Updating weights needs --allow-optimizer-step (off by default; the run otherwise stops
after building the schedule and encoding the first batch).

  python scripts/train.py --config configs/B1.yaml --budget primary_fixed_total --seed 0 \
      --bundle /data/locus-bundle --out runs/B1-primary-s0 --micro-batch 4 [--allow-optimizer-step]

Records in <out>/: run.json (config, schedule counts, versions), metrics.jsonl (per optimizer step: loss, lr,
exposures so far), checkpoints/ckpt-<k>/ (adapter + exposure counts) every 0.5 epoch-equivalents of action exposure.
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam.data import check_isolation, load_exports, schedule  # noqa: E402
from lcam.encode import collate, encode  # noqa: E402
from lcam.model import load_model, load_processor, trainable_summary  # noqa: E402


class Scheduled(Dataset):
    def __init__(self, seq, recs, processor, bundle, max_length):
        self.seq, self.recs, self.processor, self.bundle, self.max_length = seq, recs, processor, bundle, max_length

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        kind, j = self.seq[i]
        e = encode(self.processor, self.bundle, self.recs[kind][j], self.max_length)
        e['kind'] = kind
        return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--budget', choices=['primary_fixed_total', 'secondary_matched'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--micro-batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--allow-optimizer-step', action='store_true')
    a = ap.parse_args()
    cfg = C.load(a.config)
    if cfg['arm'] == 'B0':
        raise SystemExit('B0 is the untrained model; use scripts/infer.py')
    mix, n_samples = C.mix_and_samples(cfg, a.budget)
    recs = load_exports(a.bundle, 'train', sorted(mix))
    check_isolation(a.bundle, 'train', recs)
    seq, counts = schedule(recs, mix, n_samples, a.seed)
    os.makedirs(a.out, exist_ok=True)
    o = cfg['optim']
    eff = int(o['effective_batch_size'])
    if eff % a.micro_batch:
        raise SystemExit('effective_batch_size must be a multiple of --micro-batch')
    accum = eff // a.micro_batch
    steps = math.ceil(len(seq) / eff)
    ckpt_every = 0.5 * len(recs['action_sft'])
    import transformers
    import peft
    run = {'arm': cfg['arm'], 'budget': a.budget, 'seed': a.seed, 'config': cfg, 'mix': mix, 'n_samples': n_samples,
           'planned_exposures': counts, 'optimizer_steps': steps, 'micro_batch': a.micro_batch, 'grad_accum': accum,
           'checkpoint_every_action_exposures': ckpt_every, 'n_train_records': {k: len(v) for k, v in recs.items()},
           'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
                        'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
           'optimizer_step_allowed': a.allow_optimizer_step, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    processor = load_processor(cfg)
    torch.manual_seed(a.seed)
    model, attn = load_model(cfg, for_training=True)
    run['attention'] = attn
    run['params'] = trainable_summary(model)
    json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    pad = processor.tokenizer.pad_token_id
    dl = DataLoader(Scheduled(seq, recs, processor, a.bundle, cfg['data']['max_length']), batch_size=a.micro_batch, shuffle=False,
                    num_workers=a.workers, collate_fn=lambda b: (collate(b, pad), [x['kind'] for x in b], [x['stats'] for x in b]))
    if not a.allow_optimizer_step:
        batch, kinds, stats = next(iter(dl))
        print(json.dumps({'note': 'optimizer step not allowed: schedule built and first batch encoded, nothing trained',
                          'first_batch': stats, 'planned_exposures': counts, 'optimizer_steps': steps}, ensure_ascii=False, indent=1))
        return
    from transformers import get_cosine_schedule_with_warmup
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(o['lr']), weight_decay=float(o['weight_decay']))
    sched = get_cosine_schedule_with_warmup(opt, int(o['warmup_ratio'] * steps), steps)
    seen = {k: 0 for k in counts}
    tokens = 0
    next_ckpt, k_ckpt = ckpt_every, 0
    metrics = open(os.path.join(a.out, 'metrics.jsonl'), 'a')
    model.train()
    micro, loss_acc, t0 = 0, 0.0, time.time()
    for batch, kinds, stats in dl:
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / accum  # mean of micro-batch token means over the effective batch
        loss.backward()
        loss_acc += loss.item()
        for kd in kinds:
            seen[kd] += 1
        tokens += sum(s['tokens'] for s in stats)
        micro += 1
        if micro % accum == 0 or micro == len(dl):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = math.ceil(micro / accum)
            metrics.write(json.dumps({'step': step, 'loss': round(loss_acc, 5), 'lr': sched.get_last_lr()[0], 'exposures': dict(seen),
                                      'tokens': tokens, 'seconds': round(time.time() - t0, 1)}) + '\n')
            metrics.flush()
            loss_acc = 0.0
            if seen.get('action_sft', 0) >= next_ckpt or micro == len(dl):
                k_ckpt += 1
                d = os.path.join(a.out, 'checkpoints', f'ckpt-{k_ckpt}')
                model.save_pretrained(d)
                json.dump({'step': step, 'exposures': dict(seen), 'tokens': tokens}, open(os.path.join(d, 'exposures.json'), 'w'))
                next_ckpt += ckpt_every
    run['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    run['actual_exposures'] = seen
    run['tokens_seen'] = tokens
    json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), ensure_ascii=False, indent=1, default=str)


if __name__ == '__main__':
    main()
