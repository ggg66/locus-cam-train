"""v2 LoRA SFT: A-base / AG / A-compute (configs/v2/, docs/v2/protocol_v2.yaml). scripts/train.py stays the v1 code.

  torchrun --nproc-per-node 8 scripts/train_v2.py --config configs/v2/AG.yaml --seed 0 \
      --bundle /data/locus-bundle --out runs/v2-AG-s0 --micro-batch 1 [--allow-optimizer-step]

What differs from v1 (each is recorded in run.json):
  schedule     lcam.data.schedule_v2: per-source seeded streams, so the action sequence of a seed is the same in
               A-base, AG and (as a prefix) A-compute.
  layout       lcam.data.step_layout: optimizer step s always owns schedule positions [64s, 64s+64), on any number
               of GPUs; ranks take interleaved micro-batches of it.
  loss         lcam.loss sample_mean: mean CE per sample over its learned tokens, weighted mean over the samples of
               the optimizer step (all ranks); rank gradients are summed. Per-kind losses and tokens are logged.
  geometry     geometry_sft_v2: hidden fields carry fixed defaults (no true value in the teacher-forced prefix).
  checkpoints  at fixed fractions of the planned optimizer steps (0.25 / 0.5 / 0.75 / 1.0); 1.0 is the primary
               endpoint. Each checkpoint stores the eval-mode loss on a fixed validation probe (scripts/reload_check.py).
Without --allow-optimizer-step the run stops after the schedule, the layout and the first micro-batch are built.
Checks written to run.json: the router aux loss is not part of the objective (and whether the model returns one);
every sample has learned tokens; at step 1 only lora_A gradients may be zero (lora_B starts at zero); once one update
with lr > 0 has happened (warm-up starts at lr 0) every trainable tensor gets a gradient and the sampled LoRA tensors
change.
Records in <out>/: run.json, trainable_manifest.json, metrics.jsonl (per step), checkpoints/ckpt-<k>/ (adapter +
exposures.json). Options for the audit: --max-steps N (short run), --save-at-end, --disable-dropout,
--dump-first-grad PATH (gradients of step 1 after the rank combination, for scripts/compare_grads.py).
"""
import argparse
import hashlib
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
from lcam.data import check_isolation, load_exports, schedule_v2, step_layout  # noqa: E402
from lcam.encode import collate, encode  # noqa: E402
from lcam.loss import microbatch_objective, per_sample_ce, rank_grad_op  # noqa: E402
from lcam.model import load_model, load_processor, trainable_summary  # noqa: E402
from lcam.targets import classify  # noqa: E402

A_BASE_ACTIONS = 15000   # the A-base action budget; the sha of this prefix of the action sequence is recorded


def sha(obj):
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()


def dist_setup():
    ws = int(os.environ.get('WORLD_SIZE', 1))
    if ws == 1:
        return 0, 1
    import torch.distributed as dist
    torch.cuda.set_device(0)                      # only this rank's GPU is visible
    dist.init_process_group('nccl')
    return dist.get_rank(), ws


def combine_grads(params, ws, op):
    """Sum (v2) or average (v1) the accumulated LoRA gradients over ranks. A rank without samples in this step
    contributes zeros."""
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    if ws == 1:
        return
    import torch.distributed as dist
    for p in params:
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if op == 'mean':
            p.grad /= ws


def sum_across(values, ws, dev):
    if ws == 1:
        return list(values)
    import torch.distributed as dist
    t = torch.tensor(values, dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def barrier(ws):
    if ws > 1:
        import torch.distributed as dist
        dist.barrier()


class Items(Dataset):
    def __init__(self, seq, recs, processor, bundle, max_length):
        self.seq, self.recs, self.processor, self.bundle, self.max_length = seq, recs, processor, bundle, max_length

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, pos):
        kind, j = self.seq[pos]
        e = encode(self.processor, self.bundle, self.recs[kind][j], self.max_length)
        e['kind'] = kind
        return e


def probe_losses(model, processor, bundle, probe, max_length, dev):
    """Eval-mode per-sample CE on the fixed validation probe (same records for every run)."""
    pad = processor.tokenizer.pad_token_id
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(probe), 2):
            enc = [encode(processor, bundle, r, max_length) for r in probe[i:i + 2]]
            b = collate(enc, pad)
            b = {k: v.to(dev) for k, v in b.items()}
            labels = b.pop('labels')
            sums, counts = per_sample_ce(model(**b).logits, labels)
            out += [round(float(s / c), 6) for s, c in zip(sums, counts)]
    if was_training:
        model.train()
    return out


def load_probe(bundle, kinds):
    recs = load_exports(bundle, 'validation', kinds)
    return [r for k in kinds for r in recs[k][:4]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--micro-batch', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--allow-optimizer-step', action='store_true')
    ap.add_argument('--max-steps', type=int, default=0, help='audit / cost measurement: stop after N optimizer steps')
    ap.add_argument('--save-at-end', action='store_true', help='with --max-steps: save the adapter at the last step')
    ap.add_argument('--disable-dropout', action='store_true', help='audit only: makes 1-GPU and 8-GPU gradients comparable')
    ap.add_argument('--dump-first-grad', default=None)
    a = ap.parse_args()
    rank, ws = dist_setup()
    rank0 = rank == 0
    cfg = C.load(a.config)
    if cfg.get('protocol') != 'v2':
        raise SystemExit(f'{a.config} is not a v2 config; v1 configs run with scripts/train.py')
    ex = C.exposures(cfg, a.config)
    kinds = sorted(ex)
    recs = load_exports(a.bundle, 'train', kinds)
    check_isolation(a.bundle, 'train', recs)
    n_img = int(cfg['processor']['images_per_sample'])
    bad = [r['id'] for rs in recs.values() for r in rs if len(r['images']) != n_img]
    if bad:
        raise SystemExit(f'{len(bad)} records do not have {n_img} images, e.g. {bad[:3]}')
    if 'geometry_sft' in recs:
        raise SystemExit('v2 trains on geometry_sft_v2 (hidden fields as defaults), not the v1 geometry_sft')
    anchor_kind = cfg.get('anchor', 'action_sft')
    seq, counts = schedule_v2(recs, ex, a.seed, anchor=anchor_kind, streams=cfg.get('pool_streams'))
    o, L = cfg['optim'], cfg['loss']
    if 'micro_batch' in o and a.micro_batch != int(o['micro_batch']):
        raise SystemExit(f'--micro-batch {a.micro_batch} differs from optim.micro_batch {o["micro_batch"]} of {a.config}')
    eff = int(o['effective_batch_size'])
    layout = step_layout(len(seq), eff, a.micro_batch, ws)
    planned_steps = len(layout)
    accum = eff // (a.micro_batch * ws)
    reduction = L['reduction']
    weight = {k: float((L.get('sample_weight') or {}).get(k, 1.0)) for k in kinds}
    step_weight = [sum(weight[seq[p][0]] for p in st['members']) for st in layout]
    fractions = [float(f) for f in cfg['checkpoints']['at_fraction']]
    ckpt_steps = {max(1, math.ceil(f * planned_steps)): f for f in fractions}
    if a.max_steps:
        layout = layout[:a.max_steps]
        ckpt_steps = {len(layout): len(layout) / planned_steps} if a.save_at_end else {}
    anchor = [i for k, i in seq if k == anchor_kind]
    os.makedirs(a.out, exist_ok=True)
    import transformers
    import peft
    run = {'protocol': 'v2', 'run_family': cfg['run_family'], 'config_path': a.config, 'config': cfg, 'seed': a.seed,
           'planned_exposures': counts, 'n_train_records': {k: len(v) for k, v in recs.items()},
           'schedule_sha256': sha(seq), 'action_sequence': {'n': len(anchor), 'sha256': sha(anchor),
                                                            f'prefix_{A_BASE_ACTIONS}_sha256': sha(anchor[:A_BASE_ACTIONS]) if len(anchor) >= A_BASE_ACTIONS else None},
           'effective_batch_size': eff, 'micro_batch': a.micro_batch, 'world_size': ws, 'grad_accum': accum,
           'planned_optimizer_steps': planned_steps, 'steps_this_run': len(layout), 'short_run': bool(a.max_steps),
           'loss': {'reduction': reduction, 'sample_weight': weight, 'rank_grad_op': rank_grad_op(reduction), 'router_aux_loss_in_objective': False},
           'checkpoint_steps': {str(k): v for k, v in sorted(ckpt_steps.items())}, 'dropout_disabled': a.disable_dropout,
           'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
                        'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
           'optimizer_step_allowed': a.allow_optimizer_step, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}

    def dump():
        if rank0:
            json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    dump()
    processor = load_processor(cfg)
    torch.manual_seed(a.seed)          # identical on every rank, so the LoRA init is identical
    model, info = load_model(cfg, for_training=True)
    if a.disable_dropout:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    manifest = [{'name': n, 'shape': list(p.shape), 'numel': p.numel(), 'category': classify(n)} for n, p in named]
    run['model_info'] = info
    run['params'] = trainable_summary(model)
    run['trainable_manifest_sha256'] = sha(manifest)
    if rank0:
        json.dump(manifest, open(os.path.join(a.out, 'trainable_manifest.json'), 'w'), indent=0)
    dump()
    pad = processor.tokenizer.pad_token_id
    dev = next(model.parameters()).device
    mine = [mb for st in layout for mb in st['ranks'][rank]]
    dl = DataLoader(Items(seq, recs, processor, a.bundle, cfg['data']['max_length']), batch_sampler=mine, num_workers=a.workers,
                    collate_fn=lambda b: (collate(b, pad), [x['kind'] for x in b], [x['stats'] for x in b]))
    it = iter(dl)
    if not a.allow_optimizer_step:
        batch, kinds_b, stats = next(it)
        if rank0:
            print(json.dumps({'note': 'optimizer step not allowed: schedule, layout and first micro-batch built, nothing trained',
                              'first_micro_batch': stats, 'planned_exposures': counts, 'planned_optimizer_steps': planned_steps,
                              'world_size': ws, 'grad_accum': accum, 'checkpoint_steps': run['checkpoint_steps']}, ensure_ascii=False, indent=1))
        return
    probe = load_probe(a.bundle, list(cfg.get('probe_kinds', ['action_sft', 'geometry_sft_v2'])))
    from transformers import get_cosine_schedule_with_warmup
    opt = torch.optim.AdamW(params, lr=float(o['lr']), weight_decay=float(o['weight_decay']))
    sched = get_cosine_schedule_with_warmup(opt, int(o['warmup_ratio'] * planned_steps), planned_steps)
    watch = [n for n, _ in named if 'lora_A' in n][:4] + [n for n, _ in named if 'lora_B' in n][:4]
    before = {n: p.detach().clone() for n, p in named if n in watch}
    keys = ['objective', 'tokens_real', 'tokens_padded'] + [f'{k}:{x}' for k in kinds for x in ('n', 'ce_sum', 'learned_tokens')]
    cum = {k: 0 for k in kinds}
    cum_tokens = [0.0, 0.0]
    train_seconds, k_ckpt, updates = 0.0, 0, 0
    metrics = open(os.path.join(a.out, 'metrics.jsonl'), 'a') if rank0 else None
    model.train()
    for s, st in enumerate(layout):
        t0 = time.time()
        acc = dict.fromkeys(keys, 0.0)
        for _ in st['ranks'][rank]:
            batch, kinds_b, stats = next(it)
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop('labels')
            out = model(**batch)
            if s == 0 and 'router_aux_loss_returned' not in run:
                run['router_aux_loss_returned'] = getattr(out, 'aux_loss', None) is not None
            sums, cnt = per_sample_ce(out.logits, labels)
            if (cnt == 0).any():
                raise RuntimeError(f'a sample has no learned tokens: {[x["id"] for x, c in zip(stats, cnt.tolist()) if c == 0]}')
            w = torch.tensor([weight[k] for k in kinds_b], device=sums.device)
            obj = microbatch_objective(sums, cnt, reduction, accum=accum, weights=w, step_weight=step_weight[s])
            obj.backward()
            acc['objective'] += obj.item()
            acc['tokens_real'] += sum(x['tokens'] for x in stats)
            acc['tokens_padded'] += labels.numel()
            for k, sm, c in zip(kinds_b, sums.tolist(), cnt.tolist()):
                acc[f'{k}:n'] += 1
                acc[f'{k}:ce_sum'] += sm / c
                acc[f'{k}:learned_tokens'] += c
        combine_grads(params, ws, rank_grad_op(reduction))
        # lora_B starts at zero, so lora_A has no gradient until the first update with lr > 0 (warm-up starts at lr 0)
        if s == 0:
            zero = [n for n, p in named if not p.grad.abs().sum().item()]
            run['zero_grad_params_step1'] = {'n': len(zero), 'non_lora_A': [n for n in zero if 'lora_A' not in n][:10]}
            if run['zero_grad_params_step1']['non_lora_A']:
                raise RuntimeError(f'tensors other than lora_A get no gradient at step 1: {run["zero_grad_params_step1"]["non_lora_A"][:3]}')
        check_change = updates >= 1 and 'zero_grad_params_after_first_update' not in run
        if check_change:
            zero = [n for n, p in named if not p.grad.abs().sum().item()]
            run['zero_grad_params_after_first_update'] = {'step': s + 1, 'n': len(zero), 'examples': zero[:10]}
            if zero:
                raise RuntimeError(f'{len(zero)} trainable tensors get no gradient after the first update, e.g. {zero[:3]}')
        if s == 0 and a.dump_first_grad:
            g_obj = sum_across([acc['objective']], ws, dev)[0]
            if rank0:
                torch.save({'objective': g_obj if rank_grad_op(reduction) == 'sum' else g_obj / ws, 'world_size': ws, 'micro_batch': a.micro_batch,
                            'members': st['members'], 'norms': {n: float(p.grad.float().norm()) for n, p in named},
                            'tensors': {n: p.grad.float().cpu() for n, p in [x for x in named if 'lora_B' in x[0]][:8]}}, a.dump_first_grad)
        lr_now = opt.param_groups[0]['lr']
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if lr_now > 0:
            updates += 1
        if check_change:
            run['watched_params_changed'] = {'step': s + 1, **{n: bool((p.detach() != before[n]).any().item()) for n, p in named if n in watch}}
            if not all(v for k, v in run['watched_params_changed'].items() if k != 'step'):
                raise RuntimeError(f'LoRA tensors did not change: {run["watched_params_changed"]}')
            dump()
        g = dict(zip(keys, sum_across([acc[k] for k in keys], ws, dev)))
        train_seconds += time.time() - t0
        for k in kinds:
            cum[k] += int(g[f'{k}:n'])
        cum_tokens[0] += g['tokens_real']
        cum_tokens[1] += g['tokens_padded']
        if rank0:
            rec = {'step': s + 1, 'objective': round(g['objective'] if rank_grad_op(reduction) == 'sum' else g['objective'] / ws, 6),
                   'lr': lr_now, 'grad_norm': round(float(gn), 5), 'exposures': dict(cum),
                   'per_kind': {k: {'n': int(g[f'{k}:n']), 'mean_ce': round(g[f'{k}:ce_sum'] / g[f'{k}:n'], 6) if g[f'{k}:n'] else None,
                                    'learned_tokens': int(g[f'{k}:learned_tokens'])} for k in kinds},
                   'tokens_real': int(g['tokens_real']), 'tokens_padded': int(g['tokens_padded']),
                   'train_seconds': round(train_seconds, 1)}
            metrics.write(json.dumps(rec) + '\n')
            metrics.flush()
        if s + 1 in ckpt_steps:
            k_ckpt += 1
            d = os.path.join(a.out, 'checkpoints', f'ckpt-{k_ckpt}')
            if rank0:
                model.save_pretrained(d)
                json.dump({'step': s + 1, 'fraction': ckpt_steps[s + 1], 'exposures': dict(cum), 'tokens_real': int(cum_tokens[0]),
                           'tokens_padded': int(cum_tokens[1]), 'train_seconds': round(train_seconds, 1),
                           'probe_ids': [r['id'] for r in probe], 'probe_ce': probe_losses(model, processor, a.bundle, probe, cfg['data']['max_length'], dev)},
                          open(os.path.join(d, 'exposures.json'), 'w'), indent=1)
            barrier(ws)
    run['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    run['actual_exposures'] = cum
    run['tokens_real'], run['tokens_padded'] = int(cum_tokens[0]), int(cum_tokens[1])
    run['train_seconds'] = round(train_seconds, 1)
    dump()
    if ws > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
