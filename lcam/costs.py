"""Training cost of a schedule from per-record token counts (scripts/encode_stats.py), without a GPU. Stdlib only.

padded tokens = for every micro-batch, its size x its longest sequence: what the forward and backward pass
actually process (padding included), the compute measure used to match A-compute to AG. Real tokens (no padding)
are reported next to it so the choice of measure is visible.
"""
from .data import schedule_v2, step_layout


def schedule_cost(seq, stats, eff, micro, ws):
    """{'padded': .., 'real': .., 'learned': {kind: ..}} of one schedule under the v2 layout."""
    padded = real = 0
    learned = {}
    for st in step_layout(len(seq), eff, micro, ws):
        for mbs in st['ranks']:
            for mb in mbs:
                lens = [stats[seq[p][0]][seq[p][1]][0] for p in mb]
                padded += len(mb) * max(lens)
                real += sum(lens)
                for p in mb:
                    k, i = seq[p]
                    learned[k] = learned.get(k, 0) + stats[k][i][1]
    return {'padded': padded, 'real': real, 'learned': learned}


def match_actions(recs_n, stats, target, seed, eff, micro, ws, metric='padded', lo=None, hi=None):
    """Smallest number of action exposures whose action-only schedule (seed) costs at least `target` (metric)."""
    def cost(n):
        seq, _ = schedule_v2({'action_sft': range(recs_n)}, {'action_sft': n}, seed)
        return schedule_cost(seq, stats, eff, micro, ws)[metric]
    lo = lo or eff
    hi = hi or 4 * recs_n * 10
    while cost(hi) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if cost(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo, cost(lo)
