"""A randomly initialised, shrunken copy of the base model with the real tokenizer, processor and chat template, for
smoke tests of the whole pipeline on any machine (no weights are downloaded). The model type is read from config.json.

  python tests/mock/make_tiny_model.py <dir with the real non-weight files> <out dir>

qwen3_5_moe (v2): kept the model class, layer pattern (3 Gated DeltaNet + 1 gated full attention), MoE with a shared
expert, attention output gate, vision patch 16 / merge 2, special token ids, tokenizer and template.
Shrunk: 4 layers, hidden 64, 8 experts (2 active), vision depth 2.
gemma4 (v3_gemma): kept the model class, sliding / full attention pattern (3 + 1), dense MLP, shared K=V on full
attention, vision patch 16 / pooling 3 / 280 soft tokens, special token ids, tokenizer and template.
Shrunk: 4 layers, hidden 64, vision depth 2 with hidden 64.
"""
import json
import os
import shutil
import sys

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
WEIGHTS = ('.safetensors', '.bin', '.pt', '.gguf')
for f in os.listdir(src):  # src may be a full model snapshot: never copy weight shards
    if f not in ('config.json', 'model.safetensors.index.json') and not f.endswith(WEIGHTS) and os.path.isfile(os.path.join(src, f)):
        shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
cfg = json.load(open(os.path.join(src, 'config.json')))
t = cfg['text_config']
if cfg['model_type'] == 'gemma4':
    t.update(num_hidden_layers=4, layer_types=['sliding_attention'] * 3 + ['full_attention'], hidden_size=64, intermediate_size=128,
             num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1, head_dim=32, global_head_dim=32,
             sliding_window=64, initializer_range=0.2)
    cfg['vision_config'].update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=4,
                                head_dim=16, global_head_dim=16)
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(out)).to(torch.bfloat16)
    model.save_pretrained(out)
    shutil.copyfile(os.path.join(src, 'generation_config.json'), os.path.join(out, 'generation_config.json'))
    print('tiny model', sum(p.numel() for p in model.parameters()) / 1e6, 'M params ->', out)
    sys.exit(0)
t.update(num_hidden_layers=4, layer_types=['linear_attention'] * 3 + ['full_attention'], hidden_size=64, num_attention_heads=4,
         num_key_value_heads=2, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
         linear_num_value_heads=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)
t['rope_parameters']['mrope_section'] = [2, 1, 1]          # sums to head_dim * partial_rotary_factor / 2 = 4
# Larger init than the real 0.02: with a frozen random lm_head of width 64, logits from a 0.02 init are almost flat and
# LoRA cannot make the model confident, which made the overfit check pass or fail by chance.
t['initializer_range'] = 0.2
cfg['vision_config'].update(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64)
json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1)
torch.manual_seed(0)
from transformers import Qwen3_5MoeForConditionalGeneration  # noqa: E402
model = Qwen3_5MoeForConditionalGeneration(AutoConfig.from_pretrained(out)).to(torch.bfloat16)
model.save_pretrained(out)
shutil.copyfile(os.path.join(src, 'generation_config.json'), os.path.join(out, 'generation_config.json'))  # keep the real stop tokens
print('tiny model', sum(p.numel() for p in model.parameters()) / 1e6, 'M params ->', out)
