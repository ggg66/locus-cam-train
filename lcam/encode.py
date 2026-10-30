"""Encoding (chat template with thinking switched off): one export record -> input_ids, labels and the processor's
image tensors.

The sequence is built as  template(system, user; add_generation_prompt, enable_thinking=False) + target + END, where
END is the token that closes an assistant turn (processor.lcam_end_of_turn from the config: <|im_end|> for Qwen3.5,
<turn|> for Gemma 4), so the prompt part is exactly what inference sees (including the empty think block / thought
channel the template inserts when thinking is off). Labels: every token of the system/user turn and the images is
-100; the target text and the closing END are learned. Geometry records: tokens overlapping a field in
`loss_mask_paths` are -100 too.
Processor outputs other than input_ids / attention_mask are passed to the model unchanged: per-token tensors
(e.g. mm_token_type_ids) are padded in collate, per-image tensors (pixel_values, image_grid_thw for Qwen;
pixel_values, image_position_ids for Gemma 4) are concatenated over the images of the batch.
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


END = '<|im_end|>'   # default (Qwen3.5); the config can name another one, see load_processor


def prompt_text(processor, msgs, enable_thinking=False):
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def encode(processor, bundle, rec, max_length):
    msgs = messages(rec)
    end = getattr(processor, 'lcam_end_of_turn', END)
    prefix_text = prompt_text(processor, msgs[:-1])
    if end not in processor.tokenizer.get_vocab():
        raise EncodeError(f'{end} is not a single token of this tokenizer')
    full_text = prefix_text + rec['target'] + end
    images = load_images(bundle, rec) or None
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
    # other per-token fields the processor returns (e.g. mm_token_type_ids, needed for multimodal RoPE / bidirectional
    # image attention) are kept; so is every per-image tensor (first dimension = number of images)
    extra = {k: v[0] for k, v in full.items()
             if k not in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw') and hasattr(v, 'shape') and v.dim() == 2 and v.shape[1] == ids.shape[0]}
    image_extra = {k: v for k, v in full.items()
                   if k not in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw') and k not in extra and hasattr(v, 'shape')}
    for k, v in image_extra.items():
        if images is None or v.shape[0] != len(images):
            raise EncodeError(f'{rec["id"]}: processor output {k} {tuple(v.shape)} is neither per token nor per image')
    grid = full.get('image_grid_thw')
    if grid is not None:
        merge = getattr(processor.image_processor, 'merge_size', 2)
        n_visual = int((grid.prod(dim=1) // (merge * merge)).sum())
    else:
        img_id = getattr(processor, 'image_token_id', None)
        n_visual = int((ids == img_id).sum()) if img_id is not None else 0
    pv = full.get('pixel_values')
    return {'input_ids': ids, 'labels': labels, 'pixel_values': pv, 'image_grid_thw': grid, 'extra': extra, 'image_extra': image_extra,
            'stats': {'id': rec['id'], 'tokens': int(ids.shape[0]), 'prompt_tokens': p, 'visual_tokens': n_visual,
                      'images': int(pv.shape[0]) if grid is None and pv is not None else (int(grid.shape[0]) if grid is not None else 0),
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
    out = {'input_ids': ids, 'labels': labels, 'attention_mask': att}
    with_img = [b for b in batch if b['pixel_values'] is not None]
    if with_img:
        out['pixel_values'] = torch.cat([b['pixel_values'] for b in with_img])
        if with_img[0]['image_grid_thw'] is not None:
            out['image_grid_thw'] = torch.cat([b['image_grid_thw'] for b in with_img])
        for key in with_img[0].get('image_extra', {}):
            out[key] = torch.cat([b['image_extra'][key] for b in with_img])
    for key in batch[0].get('extra', {}):
        t = torch.zeros((len(batch), n), dtype=batch[0]['extra'][key].dtype)
        for i, b in enumerate(batch):
            t[i, :b['extra'][key].shape[0]] = b['extra'][key]
        out[key] = t
    return out
