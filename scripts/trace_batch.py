"""Batch trace for the training audit: exactly what the model sees and learns for the first samples of a v2 run.

  python scripts/trace_batch.py --config configs/v2/AG.yaml --seed 0 --bundle /data/locus-bundle --out $WORK/audit/batch_trace [--n 3]

Processor only (no weights, no GPU). For the first --n schedule positions of every kind it writes
<out>/<run_family>-s<seed>-<pos>.json with: messages, the templated text, input_ids, labels, the learned text
decoded, image paths + sha256 + image_grid_thw + image placeholder tokens in the sequence, and these checks:
  two_images        two images per record and as many image placeholder tokens as the grid says
  instruction       geometry prompts carry the geometry instruction, action prompts do not
  learned_tail      the learned tokens are the target text followed by <|im_end|>, and nothing of the prompt
  no_hidden_truth   geometry_sft_v2: the target holds the frozen defaults at every hidden path (training_exports
                    meta_v2.json priors), so no true hidden value is in the teacher-forced prefix
Writes <out>/summary.json; exit code 1 when a check fails.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam.data import load_exports, messages, schedule_v2  # noqa: E402
from lcam.encode import END, IGNORE, encode, prompt_text  # noqa: E402
from lcam.model import load_processor  # noqa: E402

GEOM_MARK = '请不要给出动作。请根据图像估计场景几何'


def value_at(obj, path):
    cur = obj
    for part in path.replace('[', '.[').split('.'):
        if part.startswith('['):
            cur = cur[int(part[1:-1])]
        elif part:
            cur = cur[part]
    return cur


def expected_default(target, path, priors):
    i = int(path.split('[')[1].split(']')[0]) if path.startswith('objects') else None
    if path == 'room.ceiling':
        return round(target['room']['floor'] + priors['room_height_m'], 3)
    o = target['objects'][i]
    h = priors['person_height_m'] if o['kind'] == 'person' else priors['box_height_m']
    return round(target['room']['floor'] + h / 2, 3) if path.endswith('centre[1]') else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=3)
    a = ap.parse_args()
    cfg = C.load(a.config)
    ex = C.exposures(cfg, a.config)
    recs = load_exports(a.bundle, 'train', sorted(ex))
    seq, _ = schedule_v2(recs, ex, a.seed)
    processor = load_processor(cfg)
    tok = processor.tokenizer
    img_tok = getattr(processor, 'image_token', '<|image_pad|>')
    img_id = tok.convert_tokens_to_ids(img_tok)
    merge = getattr(processor.image_processor, 'merge_size', 2)
    priors = None
    meta = os.path.join(a.bundle, 'training_exports', 'train', 'meta_v2.json')
    if os.path.exists(meta):
        priors = json.load(open(meta))['priors']
    os.makedirs(a.out, exist_ok=True)
    picked, per_kind, failures = [], {}, []
    for pos, (k, j) in enumerate(seq):
        if per_kind.get(k, 0) < a.n:
            per_kind[k] = per_kind.get(k, 0) + 1
            picked.append((pos, k, j))
        if all(per_kind.get(x, 0) >= a.n for x in ex):
            break
    for pos, k, j in picked:
        rec = recs[k][j]
        e = encode(processor, a.bundle, rec, cfg['data']['max_length'])
        ids, labels = e['input_ids'].tolist(), e['labels'].tolist()
        learned = tok.decode([t for t, l in zip(ids, labels) if l != IGNORE])
        grid = e['image_grid_thw']
        n_expected = int((grid.prod(dim=1) // (merge * merge)).sum()) if grid is not None else 0
        checks = {'two_images': len(rec['images']) == 2 and grid is not None and grid.shape[0] == 2 and ids.count(img_id) == n_expected,
                  'instruction': (GEOM_MARK in rec['prompt']) == k.startswith('geometry'),
                  'learned_tail': learned == rec['target'] + END}
        if k == 'geometry_sft_v2':
            t = json.loads(rec['target'])
            checks['no_hidden_truth'] = priors is not None and not rec.get('loss_mask_paths') and all(
                abs(value_at(t, p) - expected_default(t, p, priors)) < 1e-6 for p in rec.get('hidden_paths', []))
        bad = [c for c, ok in checks.items() if not ok]
        if bad:
            failures.append({'pos': pos, 'id': rec['id'], 'failed': bad})
        msgs = messages(rec)
        json.dump({'schedule_position': pos, 'kind': k, 'id': rec['id'], 'messages': msgs, 'templated_prompt': prompt_text(processor, msgs[:-1]),
                   'images': [{'path': p, 'sha256': hashlib.sha256(open(os.path.join(a.bundle, p), 'rb').read()).hexdigest()} for p in rec['images']],
                   'image_grid_thw': grid.tolist() if grid is not None else None, 'image_placeholder_tokens': ids.count(img_id),
                   'input_ids': ids, 'labels': labels, 'learned_text': learned, 'stats': e['stats'], 'hidden_paths': rec.get('hidden_paths'),
                   'checks': checks}, open(os.path.join(a.out, f'{cfg["run_family"]}-s{a.seed}-{pos}.json'), 'w'), ensure_ascii=False, indent=1)
    summary = {'config': a.config, 'seed': a.seed, 'traced': [(p, k, recs[k][j]['id']) for p, k, j in picked], 'failures': failures,
               'pass': not failures}
    json.dump(summary, open(os.path.join(a.out, f'summary-{cfg["run_family"]}-s{a.seed}.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    sys.exit(0 if summary['pass'] else 1)


if __name__ == '__main__':
    main()
