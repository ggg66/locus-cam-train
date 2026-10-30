"""LoRA target categories from module paths (torch-free so it can be unit-tested anywhere)."""
import re


def classify(name):
    """Category of a linear layer by its module path."""
    n = name.lower()
    if 'visual' in n or 'vision' in n:
        return 'vision'
    if 'lm_head' in n or 'embed' in n:
        return 'embedding'
    if 'shared_expert' in n:
        return 'shared_expert_gate' if re.search(r'shared_expert_gate$', n) else 'shared_expert'
    if '.experts.' in n or n.endswith('.experts'):
        return 'routed_expert'
    if re.search(r'mlp\.(gate|router)$', n):
        return 'router'
    if re.search(r'\.mlp\.(gate_proj|up_proj|down_proj)(\.|$)', n):   # module, or its LoRA / weight parameters
        return 'mlp'           # dense feed-forward (Gemma 4); an MoE's shared expert and experts are matched above
    if 'self_attn' in n or 'linear_attn' in n or 'attention' in n:
        return 'attention'
    return 'other'
