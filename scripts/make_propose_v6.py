"""The v6 ablation source propose_v6, derived from imagact_v5 in the bundle (stdlib only, deterministic).

  python scripts/make_propose_v6.py --bundle $BUNDLE

Same records, same four candidate moves in the same order and the same chosen move as imagact_v5; only the imagined
outcome of each candidate (per target height, margin, blocked, hits and the five constraint fields) is left out. A_propose
trained on it separates "imagine the outcomes" from "list candidates, then choose" and from the longer answer format.
Writes training_exports/{train,validation}/propose_v6.jsonl and stops if a result differs from the one the benchmark side
built from the same add-on (EXPECTED).
"""
import argparse
import hashlib
import json
import os

EXPECTED = {   # sha256 of the outputs, built from locus-bundle-v1.7.0-addon-v5 (imagact_v5 export-v5.1.0)
    'train': '0489b90faa9e1c5b579cbf414866b3b79f30fef3800e00d995a9d9cbad613d2f',
    'validation': 'ce8860a51f40682b2de989bd7712386c4fe6760f8e8933198c56aab49de3942b',
}


def propose(rec):
    plan, move = rec['target'].split('\n', 1)
    cands = json.loads(plan).get('candidates')
    if not isinstance(cands, list) or len(cands) != 4 or json.loads(move).keys() != {'move_m'}:
        raise ValueError(f'{rec["id"]}: not an imagact_v5 target')
    out = dict(rec, id=rec['id'].replace(':imagact', ':propose'), export_version='propose-v6 (from imagact ' + rec['export_version'] + ')')
    out['target'] = json.dumps({'candidates': [{'move': c['move']} for c in cands]}, ensure_ascii=False) + '\n' + move
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--no-check', action='store_true', help='skip the comparison with EXPECTED (tests)')
    a = ap.parse_args()
    for split in ('train', 'validation'):
        d = os.path.join(a.bundle, 'training_exports', split)
        recs = [json.loads(l) for l in open(os.path.join(d, 'imagact_v5.jsonl'), encoding='utf-8') if l.strip()]
        text = ''.join(json.dumps(propose(r), ensure_ascii=False) + '\n' for r in recs)
        sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
        print(split, len(recs), 'records, sha256', sha)
        if not a.no_check and sha != EXPECTED[split]:
            raise SystemExit(f'{split}: propose_v6 differs from the benchmark side ({EXPECTED[split]}); is imagact_v5 the export-v5.1.0 one?')
        with open(os.path.join(d, 'propose_v6.jsonl'), 'w', encoding='utf-8') as f:
            f.write(text)


if __name__ == '__main__':
    main()
