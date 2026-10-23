"""Training objectives computed from the logits (the model is called without labels).

per_sample_ce: for every sample, the sum of next-token cross-entropy over its learned tokens (labels != -100,
shifted exactly like the HF causal-LM loss) and the number of those tokens. Logits are gathered at the learned
positions only and upcast to float32 there, so no full-vocabulary float32 copy of the sequence is made.

Reductions (configs: loss.reduction)
  token_mean_per_microbatch  v1, what scripts/train.py did through `out.loss / accum` with ranks averaged:
                             one micro-batch = sum of its token losses / its learned tokens. A long Geometry-SFT
                             answer therefore outweighs a short action answer that shares its micro-batch, and
                             a sample's weight depends on who it was batched with.
  sample_mean                v2: loss_i = mean over sample i's learned tokens; the objective of an optimizer step
                             is sum_i w_i * loss_i / sum_i w_i over all samples of that step on all ranks. Each
                             micro-batch contributes its part of the numerator divided by the step's total weight
                             (known in advance from the schedule), and rank gradients are summed, not averaged,
                             so the result does not depend on the micro-batch size, accumulation or world size.
"""
import torch
import torch.nn.functional as F

IGNORE = -100


def per_sample_ce(logits, labels):
    lg = logits[:, :-1]
    lb = labels[:, 1:]
    mask = lb != IGNORE
    rows, cols = mask.nonzero(as_tuple=True)
    ce = F.cross_entropy(lg[rows, cols].float(), lb[rows, cols], reduction='none')
    sums = torch.zeros(labels.shape[0], dtype=torch.float32, device=logits.device).index_add(0, rows, ce)
    return sums, mask.sum(dim=1)


def microbatch_objective(sums, counts, reduction, accum=None, weights=None, step_weight=None):
    """The micro-batch's share of the optimizer-step objective (call .backward() on it)."""
    if reduction == 'token_mean_per_microbatch':
        return sums.sum() / counts.sum().clamp(min=1) / accum
    if reduction == 'sample_mean':
        return (weights * sums / counts.clamp(min=1)).sum() / step_weight
    raise ValueError(f'unknown loss reduction {reduction}')


def rank_grad_op(reduction):
    """How the ranks' gradients combine so that the step objective above is what is optimised."""
    return {'token_mean_per_microbatch': 'mean', 'sample_mean': 'sum'}[reduction]
