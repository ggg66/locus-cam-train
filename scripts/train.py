"""LoRA SFT for B1 / B2. Updating weights needs --allow-optimizer-step (off by default; the run otherwise stops
after building the schedule and encoding the first batch).

  python scripts/train.py --config configs/B1.yaml --budget primary_fixed_total --seed 0 \
      --bundle /data/locus-bundle --out runs/B1-primary-s0 --micro-batch 4 [--allow-optimizer-step]

Data-parallel over the GPUs of one node (one process per GPU, a full model replica each; only the LoRA
gradients are exchanged). effective_batch_size must stay a multiple of --micro-batch x world_size:

  torchrun --nproc-per-node 8 scripts/train.py ... --micro-batch 2    # 8 x 2 x accum 4 = 64 as on one GPU

Each rank walks a strided slice of the same seeded schedule, gradients are averaged across ranks before the
clip and the step, and exposures/tokens/loss are reduced to global values. Rank 0 alone writes the records.

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

# One process per GPU (torchrun): expose only this rank's GPU, so `device_map: auto` keeps the whole model on it.
# Must happen before torch initialises CUDA.
if 'LOCAL_RANK' in os.environ:
    _vis = os.environ.get('CUDA_VISIBLE_DEVICES')
    _lr = int(os.environ['LOCAL_RANK'])
    os.environ['CUDA_VISIBLE_DEVICES'] = _vis.split(',')[_lr] if _vis else str(_lr)

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam.data import check_isolation, load_exports, schedule  # noqa: E402
from lcam.encode import collate, encode  # noqa: E402
from lcam.model import load_model, load_processor, trainable_summary  # noqa: E402


def dist_setup():
    """(rank, world_size). Under torchrun one process per GPU; a plain run keeps world_size 1 and the old path."""
    ws = int(os.environ.get('WORLD_SIZE', 1))
    if ws == 1:
        return 0, 1
    import torch.distributed as dist
    torch.cuda.set_device(0)                      # only this rank's GPU is visible
    dist.init_process_group('nccl')
    return dist.get_rank(), ws


def grad_average(params, ws):
    """Average the accumulated LoRA gradients over ranks: same update as DDP with gradient accumulation."""
    import torch.distributed as dist
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= ws


def sum_across(d, ws, dev):
    """Global totals of per-rank counters (exposures, tokens)."""
    if ws == 1:
        return dict(d)
    import torch.distributed as dist
    keys = sorted(d)
    t = torch.tensor([d[k] for k in keys], dtype=torch.long, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: int(v) for k, v in zip(keys, t.tolist())}


def mean_across(x, ws, dev):
    if ws == 1:
        return x
    import torch.distributed as dist
    t = torch.tensor([x], dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / ws


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
    rank, ws = dist_setup()
    rank0 = rank == 0
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
    if eff % (a.micro_batch * ws):
        raise SystemExit(f'effective_batch_size {eff} must be a multiple of --micro-batch x world_size ({a.micro_batch} x {ws})')
    accum = eff // (a.micro_batch * ws)
    # Every rank walks a strided slice of the one seeded schedule, all slices the same length so the
    # collectives stay aligned; the union over ranks is the schedule itself.
    per_rank = len(seq) // ws
    my_seq = seq[rank::ws][:per_rank] if ws > 1 else seq
    dropped = len(seq) - per_rank * ws
    micro_per_rank = math.ceil(len(my_seq) / a.micro_batch)
    steps = math.ceil(micro_per_rank / accum)
    ckpt_every = 0.5 * len(recs['action_sft'])
    import transformers
    import peft
    run = {'arm': cfg['arm'], 'budget': a.budget, 'seed': a.seed, 'config': cfg, 'mix': mix, 'n_samples': n_samples,
           'planned_exposures': counts, 'optimizer_steps': steps, 'micro_batch': a.micro_batch, 'grad_accum': accum,
           'checkpoint_every_action_exposures': ckpt_every, 'n_train_records': {k: len(v) for k, v in recs.items()},
           'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
                        'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
           'world_size': ws, 'samples_per_rank': per_rank, 'samples_dropped_for_even_split': dropped,
           'optimizer_step_allowed': a.allow_optimizer_step, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    if rank0:
        json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    processor = load_processor(cfg)
    torch.manual_seed(a.seed)          # identical on every rank, so the LoRA init is identical
    model, info = load_model(cfg, for_training=True)
    run['model_info'] = info
    run['params'] = trainable_summary(model)
    if rank0:
        json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    pad = processor.tokenizer.pad_token_id
    dev = next(model.parameters()).device
    dl = DataLoader(Scheduled(my_seq, recs, processor, a.bundle, cfg['data']['max_length']), batch_size=a.micro_batch, shuffle=False,
                    num_workers=a.workers, collate_fn=lambda b: (collate(b, pad), [x['kind'] for x in b], [x['stats'] for x in b]))
    if not a.allow_optimizer_step:
        batch, kinds, stats = next(iter(dl))
        if rank0:
            print(json.dumps({'note': 'optimizer step not allowed: schedule built and first batch encoded, nothing trained',
                              'first_batch': stats, 'planned_exposures': counts, 'optimizer_steps': steps,
                              'world_size': ws, 'samples_per_rank': per_rank}, ensure_ascii=False, indent=1))
        return
    from transformers import get_cosine_schedule_with_warmup
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(o['lr']), weight_decay=float(o['weight_decay']))
    sched = get_cosine_schedule_with_warmup(opt, int(o['warmup_ratio'] * steps), steps)
    seen = {k: 0 for k in counts}
    tokens = 0
    seen_global, tokens_global = {k: 0 for k in counts}, 0
    next_ckpt, k_ckpt = ckpt_every, 0
    metrics = open(os.path.join(a.out, 'metrics.jsonl'), 'a') if rank0 else None
    model.train()
    micro, loss_acc, t0 = 0, 0.0, time.time()
    for batch, kinds, stats in dl:
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / accum  # mean of micro-batch token means over this rank's share of the effective batch
        loss.backward()
        loss_acc += loss.item()
        for kd in kinds:
            seen[kd] += 1
        tokens += sum(s['tokens'] for s in stats)
        micro += 1
        if micro % accum == 0 or micro == len(dl):
            if ws > 1:
                grad_average(params, ws)       # clip and step on the globally averaged gradient
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = math.ceil(micro / accum)
            g_seen = sum_across(seen, ws, dev)
            g_tokens = sum_across({'t': tokens}, ws, dev)['t']
            g_loss = mean_across(loss_acc, ws, dev)
            if rank0:
                metrics.write(json.dumps({'step': step, 'loss': round(g_loss, 5), 'lr': sched.get_last_lr()[0], 'exposures': g_seen,
                                          'tokens': g_tokens, 'seconds': round(time.time() - t0, 1)}) + '\n')
                metrics.flush()
            seen_global, tokens_global = g_seen, g_tokens
            loss_acc = 0.0
            # checkpoint cadence is counted in global action exposures (0.5 epoch-equivalents)
            if g_seen.get('action_sft', 0) >= next_ckpt or micro == len(dl):
                k_ckpt += 1
                d = os.path.join(a.out, 'checkpoints', f'ckpt-{k_ckpt}')
                if rank0:
                    model.save_pretrained(d)
                    json.dump({'step': step, 'exposures': g_seen, 'tokens': g_tokens}, open(os.path.join(d, 'exposures.json'), 'w'))
                if ws > 1:
                    import torch.distributed as dist
                    dist.barrier()
                while next_ckpt <= g_seen.get('action_sft', 0):
                    next_ckpt += ckpt_every
    run['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    run['actual_exposures'] = seen_global
    run['tokens_seen'] = tokens_global
    if rank0:
        json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    if ws > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
