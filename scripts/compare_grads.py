"""Compare the first-step gradients of two audit runs (train_v2.py --dump-first-grad), e.g. 1 GPU vs 8 GPUs.

  python scripts/compare_grads.py $WORK/audit/grad-1gpu.pt $WORK/audit/grad-8gpu.pt [--max-rel 0.02 --min-cos 0.999]

Both runs must cover the same samples in step 1 (checked) and run with --disable-dropout. Passes when the objective
and every tensor's gradient norm agree within --max-rel and the sampled lora_B gradients have cosine >= --min-cos
(bf16 reductions in a different order are not bit-identical). Exit code 1 on failure.
"""
import argparse
import json
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--max-rel', type=float, default=0.02)
    ap.add_argument('--min-cos', type=float, default=0.999)
    x = ap.parse_args()
    A, B = torch.load(x.a), torch.load(x.b)
    if A['members'] != B['members']:
        raise SystemExit('the two runs did not see the same samples in step 1')
    rel = {n: abs(A['norms'][n] - B['norms'][n]) / max(A['norms'][n], B['norms'][n], 1e-12) for n in A['norms']}
    cos = {n: float(torch.nn.functional.cosine_similarity(A['tensors'][n].flatten(), B['tensors'][n].flatten(), dim=0)) for n in A['tensors']}
    obj_rel = abs(A['objective'] - B['objective']) / max(abs(A['objective']), 1e-12)
    worst = max(rel, key=rel.get)
    res = {'world_sizes': [A['world_size'], B['world_size']], 'objective': [A['objective'], B['objective']], 'objective_rel_diff': obj_rel,
           'max_norm_rel_diff': rel[worst], 'worst_tensor': worst, 'min_cosine': min(cos.values()),
           'pass': obj_rel <= x.max_rel and rel[worst] <= x.max_rel and min(cos.values()) >= x.min_cos}
    print(json.dumps(res, indent=1))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
