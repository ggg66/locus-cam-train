"""Model and processor loading for Qwen3.5-35B-A3B (multimodal MoE: 40 layers = 30 Gated DeltaNet + 10 gated
full-attention layers, each followed by an MoE block with 256 routed experts, 8 active + 1 shared expert).

LoRA placement (configs/common.yaml `trainable.lora_on`): the linear layers of both attention kinds and of the
shared expert. Routed experts, the router, embeddings, lm_head and the whole vision tower stay frozen. Module
names are discovered from the loaded model, not hard-coded, and the chosen list is written to run.json so it can
be checked before any training.
"""
import torch

from .targets import classify


def load_processor(cfg):
    from transformers import AutoProcessor
    m, p = cfg['model'], cfg.get('processor') or {}
    kw = {}
    if p.get('min_pixels') or p.get('max_pixels'):
        kw = {'size': {'shortest_edge': p['min_pixels'], 'longest_edge': p['max_pixels']}}
    import os
    return AutoProcessor.from_pretrained(m['id'], revision=None if os.path.isdir(m['id']) else m['revision'], **kw)


def _model_class():
    import transformers
    for name in ('AutoModelForMultimodalLM', 'AutoModelForImageTextToText'):
        if hasattr(transformers, name):
            return getattr(transformers, name)
    raise SystemExit('this transformers version has no multimodal auto class; install a version that supports qwen3_5_moe')


def lora_targets(model, lora_on, rank):
    """Full module names of nn.Linear layers whose category is in `lora_on` and whose smaller side is at least the
    LoRA rank (this skips e.g. the per-head decay/beta projections of Gated DeltaNet), plus counts per category."""
    names, counts, skipped = [], {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            c = classify(name)
            counts[c] = counts.get(c, 0) + 1
            if c in lora_on:
                if min(mod.in_features, mod.out_features) >= rank:
                    names.append(name)
                else:
                    skipped.append(name)
    return names, counts, skipped


def load_model(cfg, adapter=None, for_training=False):
    m = cfg['model']
    import os
    local = os.path.isdir(m['id'])
    if not local and m.get('revision') in (None, 'TBD'):
        raise SystemExit('pin model.revision in configs/common.yaml to the downloaded commit hash first')
    model = _model_class().from_pretrained(m['id'], revision=None if local else m['revision'], dtype=torch.bfloat16, device_map=m.get('device_map', 'auto'))
    info = {'class': type(model).__name__}
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=for_training)
    elif for_training:
        from peft import LoraConfig, get_peft_model
        t = cfg['trainable']
        for p in model.parameters():
            p.requires_grad_(False)
        targets, counts, skipped = lora_targets(model, set(t['lora_on']), t['lora_r'])
        if not targets:
            raise SystemExit(f'no LoRA targets found; linear layers by category: {counts}')
        info['linear_layers_by_category'] = counts
        info['lora_target_count'] = len(targets)
        info['lora_target_examples'] = targets[:12]
        info['lora_skipped_smaller_than_rank'] = skipped[:12]
        info['lora_skipped_count'] = len(skipped)
        model = get_peft_model(model, LoraConfig(r=t['lora_r'], lora_alpha=t['lora_alpha'], lora_dropout=t['lora_dropout'],
                                                 target_modules=targets, bias='none'))
    if for_training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        bad = [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
        if bad:
            raise SystemExit(f'unexpected trainable parameters: {bad[:5]}')
        frozen_bad = [n for n, p in model.named_parameters() if p.requires_grad and classify(n) in ('vision', 'routed_expert', 'router', 'embedding')]
        if frozen_bad:
            raise SystemExit(f'LoRA reached frozen components: {frozen_bad[:5]}')
    return model, info


def trainable_summary(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total, 'trainable_params': train, 'trainable_share': round(train / total, 6)}
