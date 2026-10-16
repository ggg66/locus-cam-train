"""A randomly initialised, shrunken Qwen3.5-MoE with the real tokenizer, processor and chat template, for smoke
tests of the whole pipeline on any machine (no weights are downloaded).

  python tests/mock/make_tiny_model.py <dir with the real non-weight files> <out dir>

Kept from the real config: model class, layer pattern (3 Gated DeltaNet + 1 gated full attention), MoE with a
shared expert, attention output gate, vision patch 16 / merge 2, special token ids, tokenizer and template.
Shrunk: 4 layers, hidden 64, 8 experts (2 active), vision depth 2.
"""
import json
import os
import shutil
import sys

import torch
from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for f in os.listdir(src):
    if f not in ('config.json', 'model.safetensors.index.json') and os.path.isfile(os.path.join(src, f)):
        shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
cfg = json.load(open(os.path.join(src, 'config.json')))
t = cfg['text_config']
t.update(num_hidden_layers=4, layer_types=['linear_attention'] * 3 + ['full_attention'], hidden_size=64, num_attention_heads=4,
         num_key_value_heads=2, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
         linear_num_value_heads=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)
t['rope_parameters']['mrope_section'] = [2, 1, 1]          # sums to head_dim * partial_rotary_factor / 2 = 4
cfg['vision_config'].update(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64)
json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1)
torch.manual_seed(0)
model = Qwen3_5MoeForConditionalGeneration(AutoConfig.from_pretrained(out)).to(torch.bfloat16)
model.save_pretrained(out)
shutil.copyfile(os.path.join(src, 'generation_config.json'), os.path.join(out, 'generation_config.json'))  # keep the real stop tokens
print('tiny model', sum(p.numel() for p in model.parameters()) / 1e6, 'M params ->', out)
