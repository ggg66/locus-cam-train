"""Model and processor loading for Qwen2.5-VL-7B-Instruct with LoRA on the language model only."""
import torch

LM_TARGET_REGEX = r'^(?!.*visual).*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$'


def load_processor(cfg):
    from transformers import AutoProcessor
    m, p = cfg['model'], cfg['processor']
    return AutoProcessor.from_pretrained(m['id'], revision=m['revision'], min_pixels=p['min_pixels'], max_pixels=p['max_pixels'])


def load_model(cfg, adapter=None, for_training=False):
    from transformers import Qwen2_5_VLForConditionalGeneration
    m = cfg['model']
    if m.get('revision') in (None, 'TBD'):
        raise SystemExit('pin model.revision in configs/common.yaml to the downloaded commit hash first')
    try:
        import flash_attn  # noqa: F401
        attn = 'flash_attention_2'
    except ImportError:
        attn = 'sdpa'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(m['id'], revision=m['revision'], torch_dtype=torch.bfloat16,
                                                               attn_implementation=attn, device_map='cuda')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=for_training)
    elif for_training:
        from peft import LoraConfig, get_peft_model
        t = cfg['trainable']
        for p in model.parameters():
            p.requires_grad_(False)
        model = get_peft_model(model, LoraConfig(r=t['lora_r'], lora_alpha=t['lora_alpha'], lora_dropout=t['lora_dropout'],
                                                 target_modules=LM_TARGET_REGEX, bias='none', task_type='CAUSAL_LM'))
    if for_training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        bad = [n for n, p in model.named_parameters() if p.requires_grad and ('visual' in n or 'lora_' not in n)]
        if bad:
            raise SystemExit(f'unexpected trainable parameters (vision tower / merger / base weights): {bad[:5]}')
    return model, attn


def trainable_summary(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total, 'trainable_params': train, 'trainable_share': round(train / total, 6)}
