"""Fast generation with vLLM (B0 in both modes, and adapters if the installed vLLM supports LoRA for qwen3_5_moe).

  python scripts/infer_vllm.py --config configs/common.yaml --bundle /data/locus-bundle \
      --prompts eval_prompts/test_V0.jsonl --out outputs/B0-nothink/test_V0.jsonl [--thinking] [--adapter DIR]

Non-thinking: greedy, max_tokens 512 (as for trained arms). Thinking (B0 only): the model card's recommended
thinking sampling (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5) and max_tokens 32768; the
sampling settings are written into every output line. Output lines use the benchmark submission format.
"""
import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402


def data_url(path):
    return 'data:image/png;base64,' + base64.b64encode(open(path, 'rb').read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thinking', action='store_true')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--tensor-parallel', type=int, default=1)
    ap.add_argument('--max-model-len', type=int, default=40960)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    cfg = C.load(a.config)
    m = cfg['model']
    if a.thinking:
        sp = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768)
    else:
        sp = dict(temperature=0.0, max_tokens=512)
    llm = LLM(model=m['id'], revision=m['revision'], tensor_parallel_size=a.tensor_parallel, max_model_len=a.max_model_len,
              limit_mm_per_prompt={'image': 2}, enable_lora=bool(a.adapter), max_lora_rank=cfg['trainable']['lora_r'])
    lora = None
    if a.adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest('adapter', 1, a.adapter)
    items = [json.loads(l) for l in open(os.path.join(a.bundle, a.prompts), encoding='utf-8') if l.strip()]
    convs = []
    for it in items:
        user = [{'type': 'image_url', 'image_url': {'url': data_url(os.path.join(a.bundle, p))}} for p in it['images']]
        user.append({'type': 'text', 'text': it['prompt']})
        convs.append([{'role': 'system', 'content': it['system']}, {'role': 'user', 'content': user}])
    outs = llm.chat(convs, SamplingParams(**sp), lora_request=lora, chat_template_kwargs={'enable_thinking': a.thinking})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            f.write(json.dumps({'task_id': it['task_id'], 'sample': it['sample'], 'condition': it['condition'], 'split': it['split'],
                                'raw': c.text, 'infra_error': None, 'new_tokens': len(c.token_ids), 'finish': c.finish_reason,
                                'thinking': a.thinking, 'sampling': sp, 'adapter': a.adapter, 'engine': 'vllm'}, ensure_ascii=False) + '\n')
    print(len(items), 'answers ->', a.out)


if __name__ == '__main__':
    main()
