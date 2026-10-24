"""Does inference depend on which prompts share a batch? Compare two infer.py runs of the same prompts and adapter.

  python scripts/compare_outputs.py $WORK/v2/audit/batch1.jsonl $WORK/v2/audit/batch8.jsonl [--min-agree 0.99]

Reports the share of prompts with identical raw text and with the same parsed action (lcam.parse, scorer rules).
Exit code 1 when the parsed-action agreement is below --min-agree; evaluation then runs with batch 1.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam.parse import parse  # noqa: E402


def load(p):
    return {(o['task_id'], o['sample'], o['condition']): o['raw'] for o in map(json.loads, open(p, encoding='utf-8'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--min-agree', type=float, default=0.99)
    x = ap.parse_args()
    A, B = load(x.a), load(x.b)
    keys = sorted(set(A) & set(B))
    if len(keys) != len(A) or len(keys) != len(B):
        raise SystemExit(f'the two runs cover different prompts: {len(A)} vs {len(B)}, {len(keys)} shared')
    same_raw = sum(A[k] == B[k] for k in keys)
    same_act = sum(parse(A[k]) == parse(B[k]) for k in keys)
    res = {'n': len(keys), 'identical_raw': round(same_raw / len(keys), 4), 'same_parsed_action': round(same_act / len(keys), 4),
           'differing_examples': [k[0] for k in keys if parse(A[k]) != parse(B[k])][:10]}
    res['pass'] = res['same_parsed_action'] >= x.min_agree
    print(json.dumps(res, indent=1))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
