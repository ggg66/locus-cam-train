"""v2 objective (lcam.loss) on a tiny causal LM: the gradient of one optimizer step is the same for every split of
the step into ranks and micro-batches; the v1 token-mean objective is not. Skipped without torch.
  python3 -m unittest tests.test_loss_torch
"""
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAVE_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAVE_TORCH, 'torch not installed')
class LossV2(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(0)
        self.torch = torch
        V, D = 50, 16
        self.model = torch.nn.Sequential(torch.nn.Embedding(V, D), torch.nn.Linear(D, V))
        self.samples = []
        for n_prompt, n_target in [(5, 2), (6, 12), (4, 3), (7, 15), (5, 2), (3, 2), (6, 14), (4, 3)]:
            ids = torch.randint(0, V, (n_prompt + n_target,))
            labels = ids.clone()
            labels[:n_prompt] = -100
            self.samples.append((ids, labels))

    def batch(self, idx):
        torch = self.torch
        n = max(len(self.samples[i][0]) for i in idx)
        ids = torch.zeros((len(idx), n), dtype=torch.long)
        lab = torch.full((len(idx), n), -100, dtype=torch.long)
        for r, i in enumerate(idx):
            a, b = self.samples[i]
            ids[r, :len(a)], lab[r, :len(b)] = a, b
        return ids, lab

    def grad(self, layout, reduction):
        """layout: list of ranks, each a list of micro-batches (sample indices); one optimizer step."""
        from lcam.loss import microbatch_objective, per_sample_ce, rank_grad_op
        torch = self.torch
        members = [i for r in layout for mb in r for i in mb]
        accum = max(len(r) for r in layout)
        rank_grads = []
        for r in layout:
            self.model.zero_grad()
            for mb in r:
                ids, lab = self.batch(mb)
                sums, cnt = per_sample_ce(self.model(ids), lab)
                obj = microbatch_objective(sums, cnt, reduction, accum=accum, weights=torch.ones(len(mb)), step_weight=float(len(members)))
                obj.backward()
            rank_grads.append([p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()])
        comb = [sum(g[j] for g in rank_grads) for j in range(len(rank_grads[0]))]
        if rank_grad_op(reduction) == 'mean':
            comb = [c / len(layout) for c in comb]
        return torch.cat([c.flatten() for c in comb])

    def test_per_sample_ce_matches_cross_entropy(self):
        from lcam.loss import per_sample_ce
        torch = self.torch
        ids, lab = self.batch([0, 1])
        logits = self.model(ids)
        sums, cnt = per_sample_ce(logits, lab)
        for r in range(2):
            ref = torch.nn.functional.cross_entropy(logits[r, :-1], lab[r, 1:], ignore_index=-100, reduction='sum')
            self.assertAlmostEqual(float(sums[r]), float(ref), places=4)
        self.assertEqual(cnt.tolist(), [2, 12])

    def test_sample_mean_independent_of_split(self):
        torch = self.torch
        one = self.grad([[[0, 1], [2, 3], [4, 5], [6, 7]]], 'sample_mean')
        two = self.grad([[[0, 2], [4, 6]], [[1, 3], [5, 7]]], 'sample_mean')
        eight = self.grad([[[i]] for i in range(8)], 'sample_mean')
        ref = self.grad([[[0, 1, 2, 3, 4, 5, 6, 7]]], 'sample_mean')
        for g in (one, two, eight):
            self.assertTrue(torch.allclose(g, ref, atol=1e-6))

    def test_token_mean_depends_on_split(self):
        torch = self.torch
        a = self.grad([[[0, 1], [2, 3], [4, 5], [6, 7]]], 'token_mean_per_microbatch')
        b = self.grad([[[0, 2], [1, 3], [4, 6], [5, 7]]], 'token_mean_per_microbatch')
        self.assertFalse(torch.allclose(a, b, atol=1e-6))

    def test_equal_weights_equal_task_means_at_3_to_1(self):
        from lcam.loss import microbatch_objective
        torch = self.torch
        per = torch.tensor([1.0, 2.0, 3.0, 10.0])      # three action samples, one geometry sample
        cnt = torch.ones(4)
        v2 = microbatch_objective(per, cnt, 'sample_mean', weights=torch.ones(4), step_weight=4.0)
        self.assertAlmostEqual(float(v2), 0.75 * 2.0 + 0.25 * 10.0, places=6)


if __name__ == '__main__':
    unittest.main()
