"""GRPO with execution rewards on the camera-move task (v5 methods 1 and 2), continuing an SFT LoRA adapter.

  torchrun --nproc-per-node 8 scripts/train_grpo.py --config configs/v5/m1_rl.yaml --seed 0 --bundle /data/locus-bundle \
      --out work/runs/v5/m1_rl-s0 --init-adapter work/runs/v5/base-s0/checkpoints/ckpt-4

Each optimizer step: every rank takes `rl.prompts_per_rank` training prompts (the V0 prompt of `rl.prompt_kind` records,
development families excluded), samples `rl.group_size` answers per prompt from the current policy, scores them with
lcam.rewards against the root's outcome table, and turns the rewards of each prompt's group into advantages. The update is
on-policy and single-pass, so the policy ratio is 1 and the loss is  -sum_i A_i * mean_t log p(token_t) / N  with N = all
samples of the step on all ranks (samples of groups without reward differences have A = 0 and are not forwarded); rank
gradients are summed. Sampling uses temperature 1, top-p 1, no top-k, so the samples come from exactly the policy whose
log-probabilities are trained. No KL term and no reference model (rl.kl is not supported); LoRA dropout is switched off so
that the sampling and the trained policy are the same function.
Records in <out>/: run.json, metrics.jsonl (per step: reward, success, outcome counts, imagination accuracy, share of
groups without signal, answer length), checkpoints/ckpt-<k>/ at rl.checkpoints.at_fraction of the steps.
"""
import argparse
import json
import math
import os
import platform
import random
import sys
import time

if 'LOCAL_RANK' in os.environ:      # one process per GPU, as in train_v2.py
    _vis = os.environ.get('CUDA_VISIBLE_DEVICES')
    _lr = int(os.environ['LOCAL_RANK'])
    os.environ['CUDA_VISIBLE_DEVICES'] = _vis.split(',')[_lr] if _vis else str(_lr)

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam import rewards as R  # noqa: E402
from lcam.data import load_exports  # noqa: E402
from lcam.encode import IGNORE, load_images, prompt_text  # noqa: E402
from lcam.loss import per_sample_ce  # noqa: E402
from lcam.model import load_model, load_processor, trainable_summary  # noqa: E402

PER_TOKEN_SKIP = ('input_ids', 'attention_mask')


def dist_setup():
    ws = int(os.environ.get('WORLD_SIZE', 1))
    if ws == 1:
        return 0, 1
    import torch.distributed as dist
    torch.cuda.set_device(0)
    dist.init_process_group('nccl')
    return dist.get_rank(), ws


def sum_across(values, ws, dev):
    if ws == 1:
        return list(values)
    import torch.distributed as dist
    t = torch.tensor(values, dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def inputs(processor, bundle, rec, n, nested):
    """Processor output for n copies of one prompt (the batch path of scripts/infer.py, so generation sees exactly what
    evaluation sees)."""
    user = [{'type': 'image', 'image': p} for p in rec['images']] + [{'type': 'text', 'text': rec['prompt']}]
    msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': rec['system']}]}, {'role': 'user', 'content': user}]
    text = prompt_text(processor, msgs)
    imgs = load_images(bundle, rec)
    images = [imgs] * n if nested else imgs * n
    return processor(text=[text] * n, images=images, return_tensors='pt', padding=True)


def cut(row, stop_ids, pad_id):
    """Generated ids up to and including the first stop token (all of them if none), pads removed."""
    out = []
    for t in row.tolist():
        if t == pad_id and t not in stop_ids:
            continue
        out.append(t)
        if t in stop_ids:
            break
    return out


def with_completion(enc1, comp, dev):
    """Model inputs and labels for prompt + sampled completion: only the completion tokens are learned."""
    p = enc1['input_ids'].shape[1]
    c = torch.tensor(comp, dtype=torch.long)
    batch = {'input_ids': torch.cat([enc1['input_ids'][0], c])[None], 'attention_mask': torch.ones((1, p + len(comp)), dtype=torch.long)}
    for k, v in enc1.items():
        if k in PER_TOKEN_SKIP or not hasattr(v, 'shape'):
            continue
        if v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == p:      # per-token (e.g. mm_token_type_ids): text tokens are 0
            batch[k] = torch.cat([v, torch.zeros((1, len(comp)), dtype=v.dtype)], dim=1)
        else:                                                           # per-image tensors
            batch[k] = v
    labels = torch.full((1, p + len(comp)), IGNORE, dtype=torch.long)
    labels[0, p:] = c
    return {k: v.to(dev) for k, v in batch.items()}, labels.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--init-adapter', required=True, help='SFT checkpoint the policy starts from')
    ap.add_argument('--max-steps', type=int, default=0, help='smoke test: stop after N steps')
    ap.add_argument('--smoke', action='store_true', help='smoke test only: groups without reward differences get +1/-1 '
                    'advantages, so that the update path runs on a random tiny model')
    a = ap.parse_args()
    rank, ws = dist_setup()
    rank0 = rank == 0
    cfg = C.load(a.config)
    rl = cfg['rl']
    if rl.get('kl'):
        raise SystemExit('rl.kl is not supported (no reference model)')
    if float(rl.get('temperature', 1.0)) != 1.0 or float(rl.get('top_p', 1.0)) != 1.0:
        raise SystemExit('rl.temperature and rl.top_p must be 1: the policy gradient uses the unmodified log-probabilities')
    kind = rl['prompt_kind']
    recs = load_exports(a.bundle, 'train', [kind], exclude_families=cfg['data'].get('exclude_families'))[kind]
    tables = R.load_tables(os.path.join(a.bundle, 'reward_tables', 'train.jsonl.gz'), roots={r['root_id'] for r in recs})
    missing = [r['root_id'] for r in recs if r['root_id'] not in tables]
    if missing:
        raise SystemExit(f'{len(missing)} training roots have no reward table, e.g. {missing[:3]}')
    steps = a.max_steps or int(rl['steps'])
    P, G = int(rl['prompts_per_rank']), int(rl['group_size'])
    perm = list(range(len(recs)))
    random.Random(f'{a.seed}|grpo').shuffle(perm)
    fractions = [float(f) for f in rl['checkpoints']['at_fraction']]
    ckpt_steps = {max(1, math.ceil(f * steps)): f for f in fractions}
    os.makedirs(a.out, exist_ok=True)
    import peft
    import transformers
    run = {'protocol': 'v5-grpo', 'run_family': cfg['run_family'], 'config_path': a.config, 'config': cfg, 'seed': a.seed,
           'init_adapter': a.init_adapter, 'n_prompts': len(recs), 'steps': steps, 'world_size': ws,
           'samples_per_step': ws * P * G, 'reward': R.settings(rl.get('reward')), 'checkpoint_steps': {str(k): v for k, v in sorted(ckpt_steps.items())},
           'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
                        'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
           'started': time.strftime('%Y-%m-%d %H:%M:%S')}

    def dump():
        if rank0:
            json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    dump()
    processor = load_processor(cfg)
    processor.tokenizer.padding_side = 'left'
    model, info = load_model(cfg, adapter=a.init_adapter, for_training=True)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit('the init adapter loaded without trainable parameters')
    run['model_info'], run['params'] = info, trainable_summary(model)
    dump()
    tok = processor.tokenizer
    stop_ids = sorted({tok.convert_tokens_to_ids(processor.lcam_end_of_turn), tok.eos_token_id} - {None})
    pad = tok.pad_token_id
    nested = bool((cfg.get('processor') or {}).get('images_per_prompt_list'))
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(params, lr=float(rl['lr']), weight_decay=0.0)
    warm = int(rl.get('warmup_steps', 0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)
    metrics = open(os.path.join(a.out, 'metrics.jsonl'), 'a') if rank0 else None
    k_ckpt, t_total = 0, 0.0
    rcfg = rl.get('reward')
    for s in range(steps):
        t0 = time.time()
        model.eval()
        samples, st = [], {'reward': 0.0, 'n': 0, 'satisfied': 0, 'invalid': 0, 'illegal': 0, 'unmet': 0, 'imag': 0.0, 'imag_n': 0,
                           'groups': 0, 'flat_groups': 0, 'tokens': 0}
        for j in range(P):
            rec = recs[perm[((s * ws + rank) * P + j) % len(perm)]]
            enc = inputs(processor, a.bundle, rec, G, nested).to(dev)
            torch.manual_seed(hash((a.seed, s, rank, j)) % (2 ** 31))
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     max_new_tokens=int(rl['max_new_tokens']), eos_token_id=stop_ids, pad_token_id=pad, use_cache=True)
            comps = [cut(row, stop_ids, pad) for row in gen[:, enc['input_ids'].shape[1]:]]
            scored = [R.reward(tok.decode(c, skip_special_tokens=True), tables[rec['root_id']], rcfg) for c in comps]
            rs = [x[0] for x in scored]
            adv = R.group_advantages(rs, normalize_std=bool(rl.get('normalize_std', True)))
            st['groups'] += 1
            st['flat_groups'] += all(x == 0 for x in adv)
            if a.smoke and all(x == 0 for x in adv):
                adv = [1.0 if i % 2 == 0 else -1.0 for i in range(len(adv))]
            for c, (r, inf), ad in zip(comps, scored, adv):
                st['reward'] += r
                st['n'] += 1
                st['tokens'] += len(c)
                o = inf['outcome']
                st['satisfied' if o == 'satisfied' else 'illegal' if o == 'illegal_endpoint' else 'unmet' if o == 'spatial_unmet' else 'invalid'] += 1
                if inf.get('imagination') is not None:
                    st['imag'] += inf['imagination']
                    st['imag_n'] += 1
                if ad != 0 and c:
                    samples.append((rec, c, ad))
        keys = list(st)
        tot = sum_across([st[k] for k in keys] + [len(samples)], ws, dev)
        g, n_global = dict(zip(keys, tot)), int(tot[-1])
        n_all = ws * P * G
        model.train()
        for rec, comp, ad in samples:
            batch, labels = with_completion(inputs(processor, a.bundle, rec, 1, nested), comp, dev)
            sums, cnt = per_sample_ce(model(**batch).logits, labels)
            (ad * sums[0] / cnt[0].clamp(min=1) / n_all).backward()
        gn = 0.0
        if n_global:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            if ws > 1:
                import torch.distributed as dist
                for p in params:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(rl.get('grad_clip', 1.0))))
            opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        t_total += time.time() - t0
        if rank0:
            n = max(g['n'], 1)
            metrics.write(json.dumps({'step': s + 1, 'lr': opt.param_groups[0]['lr'], 'reward': round(g['reward'] / n, 4),
                                      'success': round(g['satisfied'] / n, 4), 'outcomes': {k: int(g[k]) for k in ('satisfied', 'unmet', 'illegal', 'invalid')},
                                      'imagination': round(g['imag'] / g['imag_n'], 4) if g['imag_n'] else None,
                                      'flat_group_share': round(g['flat_groups'] / max(g['groups'], 1), 4), 'trained_samples': n_global,
                                      'mean_answer_tokens': round(g['tokens'] / n, 1), 'grad_norm': round(gn, 5), 'seconds': round(t_total, 1)}) + '\n')
            metrics.flush()
        if s + 1 in ckpt_steps:
            k_ckpt += 1
            if rank0:
                d = os.path.join(a.out, 'checkpoints', f'ckpt-{k_ckpt}')
                model.save_pretrained(d)
                json.dump({'step': s + 1, 'fraction': ckpt_steps[s + 1], 'seconds': round(t_total, 1)}, open(os.path.join(d, 'progress.json'), 'w'))
            if ws > 1:
                import torch.distributed as dist
                dist.barrier()
    run['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    run['train_seconds'] = round(t_total, 1)
    dump()
    if ws > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
