"""Generation on prepared prompt files (B0 baseline, validation checkpoint selection, final test answers).

  python scripts/infer.py --config configs/common.yaml --bundle /data/locus-bundle \
      --prompts eval_prompts/test_V0.jsonl --out outputs/B0/test_V0.jsonl [--adapter runs/B1.../checkpoints/ckpt-3] \
      [--max-new-tokens 512] [--batch 8]

Each output line: {"task_id", "sample", "condition", "split", "raw", "infra_error", "new_tokens", "finish"} - the
submission format of the benchmark scorer. Test answers are scored on the benchmark machine (no test labels
here); validation answers can be scored here with scripts/select_checkpoint.py. Decoding is greedy, so repeated
samples of the same prompt are identical; that is recorded, not counted as independent samples.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam.encode import load_images, prompt_text  # noqa: E402
from lcam.model import load_model, load_processor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--max-new-tokens', type=int, default=None)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--thinking', action='store_true', help='B0 only: keep the model thinking mode (default off, as in training)')
    a = ap.parse_args()
    cfg = C.load(a.config)
    max_new = a.max_new_tokens or cfg['evaluation'].get('max_new_tokens', 512)
    processor = load_processor(cfg)
    processor.tokenizer.padding_side = 'left'
    model, _ = load_model(cfg, adapter=a.adapter)
    model.eval()
    items = [json.loads(l) for l in open(os.path.join(a.bundle, a.prompts), encoding='utf-8') if l.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)['task_id'], json.loads(l)['sample'], json.loads(l)['condition']) for l in open(a.out)}
    todo = [it for it in items if (it['task_id'], it['sample'], it['condition']) not in done]
    # stop explicitly at <|im_end|> (the end of an assistant turn) instead of relying on the checkpoint's generation_config
    tok = processor.tokenizer
    stop_ids = sorted({tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id} - {None})
    with open(a.out, 'a', encoding='utf-8') as f:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            texts, images = [], []
            for it in chunk:
                user = [{'type': 'image', 'image': p} for p in it['images']] + [{'type': 'text', 'text': it['prompt']}]
                msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': it['system']}]}, {'role': 'user', 'content': user}]
                texts.append(prompt_text(processor, msgs, enable_thinking=a.thinking))
                images.extend(load_images(a.bundle, it))
            enc = processor(text=texts, images=images or None, return_tensors='pt', padding=True).to(next(model.parameters()).device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stop_ids, pad_token_id=processor.tokenizer.pad_token_id)
            new = gen[:, enc['input_ids'].shape[1]:]
            for it, row in zip(chunk, new):
                ids = row.tolist()
                n = len([t for t in ids if t != processor.tokenizer.pad_token_id])
                f.write(json.dumps({'task_id': it['task_id'], 'sample': it['sample'], 'condition': it['condition'], 'split': it['split'],
                                    'raw': processor.tokenizer.decode(row, skip_special_tokens=True), 'infra_error': None, 'new_tokens': n,
                                    'finish': 'stop' if any(t in stop_ids for t in ids) else 'length',
                                    'adapter': a.adapter, 'decoding': 'greedy', 'thinking': a.thinking}, ensure_ascii=False) + '\n')
            f.flush()
            print(f'{min(i + a.batch, len(todo))}/{len(todo)}', flush=True)


if __name__ == '__main__':
    main()
