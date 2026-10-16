"""Data check with the real processor (tokenizer + image processor files only, no weights, no GPU needed).

  python scripts/check_data.py --config configs/common.yaml --bundle /data/locus-bundle [--split train] [--limit 0]

Encodes every Action-SFT and Geometry-SFT record exactly as training does and reports: total / prompt / visual /
target / masked-target tokens, records over max_length, encoding errors (prefix mismatch, context-dependent
tokenisation of the assistant turn, missing mask paths), and one decoded example per kind showing which target
tokens are learned. Output: <bundle>/checks/check_data-<split>.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam.data import check_isolation, load_exports  # noqa: E402
from lcam.encode import IGNORE, EncodeError, encode  # noqa: E402
from lcam.model import load_processor  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    cfg = C.load(a.config)
    processor = load_processor(cfg)
    recs = load_exports(a.bundle, a.split, ['action_sft', 'geometry_sft'])
    check_isolation(a.bundle, a.split, recs)
    out = {'split': a.split, 'max_length': cfg['data']['max_length'], 'processor': cfg['processor'], 'kinds': {}}
    for kind, rs in recs.items():
        stats, errors, example = [], [], None
        for r in rs[:a.limit or None]:
            try:
                e = encode(processor, a.bundle, r, cfg['data']['max_length'])
            except EncodeError as ex:
                errors.append(str(ex))
                continue
            stats.append(e['stats'])
            if example is None:
                lab = e['labels']
                learned = processor.tokenizer.decode([t for t, l in zip(e['input_ids'].tolist(), lab.tolist()) if l != IGNORE])
                example = {'id': r['id'], 'learned_text': learned, 'masked_paths': r.get('loss_mask_paths', [])}
        col = lambda k: [s[k] for s in stats]  # noqa: E731
        out['kinds'][kind] = {'n': len(rs[:a.limit or None]), 'encoded': len(stats), 'errors': errors[:20], 'n_errors': len(errors),
                              **{k: {'median': pct(col(k), .5), 'p95': pct(col(k), .95), 'max': max(col(k)) if stats else None}
                                 for k in ('tokens', 'prompt_tokens', 'visual_tokens', 'target_tokens', 'masked_target_tokens')},
                              'example': example}
    out['pass'] = all(v['n_errors'] == 0 for v in out['kinds'].values())
    os.makedirs(os.path.join(a.bundle, 'checks'), exist_ok=True)
    json.dump(out, open(os.path.join(a.bundle, 'checks', f'check_data-{a.split}.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
