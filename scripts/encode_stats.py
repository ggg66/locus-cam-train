"""Per-record token counts with the real processor (tokenizer + image processor only; no weights, no GPU).

  python scripts/encode_stats.py --config configs/v2/common.yaml --bundle /data/locus-bundle \
      --kinds action_sft geometry_sft geometry_sft_v2 --out $WORK/audit/encode_stats-train.json [--workers 16]

Every record is encoded exactly as training does (lcam.encode). Output per kind: {record index: [tokens,
learned_tokens, visual_tokens]} plus summaries. scripts/calibrate_compute.py and scripts/audit_v1_weighting.py read
this file, so the compute match and the v1 loss audit use the true lengths without touching a GPU.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam import config as C  # noqa: E402
from lcam.data import load_exports  # noqa: E402
from lcam.encode import IGNORE, encode  # noqa: E402

_P = {}


def _init(cfg, bundle, max_length):
    from lcam.model import load_processor
    _P.update(processor=load_processor(cfg), bundle=bundle, max_length=max_length)


def _one(rec):
    e = encode(_P['processor'], _P['bundle'], rec, _P['max_length'])
    learned = int((e['labels'] != IGNORE).sum())
    return [e['stats']['tokens'], learned, e['stats']['visual_tokens']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--kinds', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    cfg = C.load(a.config)
    recs = load_exports(a.bundle, a.split, a.kinds)
    out = {'split': a.split, 'fields': ['tokens', 'learned_tokens', 'visual_tokens'], 'kinds': {}, 'summary': {}}
    with mp.Pool(a.workers, initializer=_init, initargs=(cfg, a.bundle, cfg['data']['max_length'])) as pool:
        for k in a.kinds:
            rows = pool.map(_one, recs[k], chunksize=16)
            out['kinds'][k] = rows
            col = lambda i: sorted(r[i] for r in rows)  # noqa: E731
            out['summary'][k] = {f: {'mean': round(sum(col(i)) / len(rows), 1), 'median': col(i)[len(rows) // 2], 'max': col(i)[-1]}
                                 for i, f in enumerate(out['fields'])}
            out['summary'][k]['n'] = len(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'))
    print(json.dumps(out['summary'], indent=1))


if __name__ == '__main__':
    main()
