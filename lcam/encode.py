"""Encoding for Qwen3.5 (chat template with thinking switched off): one export record -> input_ids, labels,
pixel_values, image_grid_thw.

The sequence is built as  template(system, user; add_generation_prompt, enable_thinking=False) + target + <|im_end|>
so the prompt part is exactly what inference sees (including the empty think block the template inserts when
thinking is off). Labels: every token of the system/user turn and the images is -100; the target text and the
closing <|im_end|> are learned. Geometry records: tokens overlapping a field in `loss_mask_paths` are -100 too.
Every encoding checks that the prompt-only tokenisation is a prefix of the full one and that the assistant part
tokenises the same on its own; a mismatch or a sequence longer than max_length raises instead of truncating.
"""
import os

import torch
from PIL import Image

from .data import messages
from .spans import masked_char_spans, token_mask

IGNORE = -100


class EncodeError(RuntimeError):
    pass


def load_images(bundle, rec):
    return [Image.open(os.path.join(bundle, p)).convert('RGB') for p in rec['images']]


END = '<|im_end|>'


def prompt_text(processor, msgs, enable_thinking=False):
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def encode(processor, bundle, rec, max_length):
    msgs = messages(rec)
    prefix_text = prompt_text(processor, msgs[:-1])
    if END not in processor.tokenizer.get_vocab():
        raise EncodeError(f'{END} is not a single token of this tokenizer')
    full_text = prefix_text + rec['target'] + END
    images = load_images(bundle, rec)
    full = processor(text=[full_text], images=images, return_tensors='pt')
    prefix = processor(text=[prefix_text], images=images, return_tensors='pt')
    ids = full['input_ids'][0]
    p = prefix['input_ids'].shape[1]
    if not torch.equal(ids[:p], prefix['input_ids'][0]):
        raise EncodeError(f'{rec["id"]}: prompt tokens are not a prefix of the full sequence')
    if ids.shape[0] > max_length:
        raise EncodeError(f'{rec["id"]}: {ids.shape[0]} tokens > max_length {max_length} (no silent truncation)')
    labels = ids.clone()
    labels[:p] = IGNORE
    n_masked = 0
    paths = rec.get('loss_mask_paths') or []
    if paths:
        suffix = full_text[len(prefix_text):]
        tok = processor.tokenizer(suffix, add_special_tokens=False, return_offsets_mapping=True)
        if tok['input_ids'] != ids[p:].tolist():
            raise EncodeError(f'{rec["id"]}: assistant part tokenises differently in context')
        start = suffix.find(rec['target'])
        if start < 0:
            raise EncodeError(f'{rec["id"]}: target text not found in the assistant turn')
        spans = [(a + start, b + start) for a, b in masked_char_spans(rec['target'], paths)]
        for j, hit in enumerate(token_mask(tok['offset_mapping'], spans)):
            if hit:
                labels[p + j] = IGNORE
                n_masked += 1
    grid = full['image_grid_thw']
    merge = getattr(processor.image_processor, 'merge_size', 2)
    n_visual = int((grid.prod(dim=1) // (merge * merge)).sum())
    return {'input_ids': ids, 'labels': labels, 'pixel_values': full['pixel_values'], 'image_grid_thw': grid,
            'stats': {'id': rec['id'], 'tokens': int(ids.shape[0]), 'prompt_tokens': p, 'visual_tokens': n_visual,
                      'target_tokens': int(ids.shape[0]) - p, 'masked_target_tokens': n_masked}}


def collate(batch, pad_id):
    n = max(b['input_ids'].shape[0] for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = b['input_ids'].shape[0]
        ids[i, :k] = b['input_ids']
        labels[i, :k] = b['labels']
        att[i, :k] = 1
    return {'input_ids': ids, 'labels': labels, 'attention_mask': att,
            'pixel_values': torch.cat([b['pixel_values'] for b in batch]), 'image_grid_thw': torch.cat([b['image_grid_thw'] for b in batch])}
