"""Character spans of JSON leaf values, so that Geometry-SFT fields listed in `loss_mask_paths` can be excluded
from the loss token by token.

`dumps_with_spans(obj)` returns exactly `json.dumps(obj, ensure_ascii=False)` (default separators) together with
{path: (start, end)} for every leaf value, where a path looks like `room.ceiling` or `objects[0].centre[1]`.
A whole sub-tree can be masked too (`objects[1]` masks every leaf below it).
"""
import json


def dumps_with_spans(obj):
    parts, spans, pos = [], {}, [0]

    def emit(s):
        parts.append(s)
        pos[0] += len(s)

    def walk(x, path):
        if isinstance(x, dict):
            emit('{')
            for i, (k, v) in enumerate(x.items()):
                if i:
                    emit(', ')
                emit(json.dumps(k, ensure_ascii=False) + ': ')
                walk(v, f'{path}.{k}' if path else k)
            emit('}')
        elif isinstance(x, list):
            emit('[')
            for i, v in enumerate(x):
                if i:
                    emit(', ')
                walk(v, f'{path}[{i}]')
            emit(']')
        else:
            start = pos[0]
            emit(json.dumps(x, ensure_ascii=False))
            spans[path] = (start, pos[0])

    walk(obj, '')
    return ''.join(parts), spans


def masked_char_spans(target_text, mask_paths):
    """Spans (start, end) in `target_text` to exclude. Raises if the text is not the canonical serialisation."""
    obj = json.loads(target_text)
    text, spans = dumps_with_spans(obj)
    if text != target_text:
        raise ValueError('target is not in canonical json.dumps form; cannot locate masked fields')
    out = []
    for p in mask_paths:
        hits = [s for k, s in spans.items() if k == p or k.startswith(p + '.') or k.startswith(p + '[')]
        if not hits:
            raise KeyError(f'mask path {p} not found in target')
        out.extend(hits)
    return sorted(out)


def token_mask(offsets, char_spans):
    """For each token (start, end) offset, True when it overlaps any masked character span."""
    res = []
    for a, b in offsets:
        res.append(any(a < e and s < b for s, e in char_spans))
    return res
