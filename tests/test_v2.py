"""v2 schedule, layout, cost and v1-weighting logic (stdlib only; config tests need PyYAML).
  python3 -m unittest discover -s tests
"""
import importlib.util
import json
import os
import random
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from lcam.costs import match_actions, schedule_cost  # noqa: E402
from lcam.data import kind_pool, schedule_v2, step_layout  # noqa: E402

HAVE_YAML = importlib.util.find_spec('yaml') is not None
N_ACT, N_GEO = 5053, 5053
RECS = {'action_sft': range(N_ACT), 'geometry_sft_v2': range(N_GEO)}


def actions(seq):
    return [i for k, i in seq if k == 'action_sft']


class ScheduleV2(unittest.TestCase):
    def test_same_action_sequence_across_families(self):
        for seed in (0, 1, 2):
            a_base, c1 = schedule_v2(RECS, {'action_sft': 15000}, seed)
            ag, c2 = schedule_v2(RECS, {'action_sft': 15000, 'geometry_sft_v2': 5000}, seed)
            a_comp, _ = schedule_v2(RECS, {'action_sft': 21632}, seed)
            self.assertEqual(c1, {'action_sft': 15000})
            self.assertEqual(c2, {'action_sft': 15000, 'geometry_sft_v2': 5000})
            self.assertEqual(len(ag), 20000)
            self.assertEqual(actions(ag), actions(a_base))
            self.assertEqual(actions(a_comp)[:15000], actions(a_base))

    def test_seeds_differ_and_passes_have_no_repeats(self):
        p0, p1 = kind_pool(N_ACT, 15000, 0, 'action_sft'), kind_pool(N_ACT, 15000, 1, 'action_sft')
        self.assertNotEqual(p0, p1)
        for k in range(2):
            self.assertEqual(sorted(p0[k * N_ACT:(k + 1) * N_ACT]), list(range(N_ACT)))

    def test_geometry_spread_over_the_run(self):
        ag, _ = schedule_v2(RECS, {'action_sft': 15000, 'geometry_sft_v2': 5000}, 0)
        for q in range(4):
            part = ag[q * 5000:(q + 1) * 5000]
            share = sum(k == 'geometry_sft_v2' for k, _ in part) / len(part)
            self.assertAlmostEqual(share, 0.25, delta=0.03)


class Layout(unittest.TestCase):
    def test_members_do_not_depend_on_world_size(self):
        n = 15000
        l1, l8 = step_layout(n, 64, 2, 1), step_layout(n, 64, 2, 8)
        self.assertEqual(len(l1), 235)
        self.assertEqual([s['members'] for s in l1], [s['members'] for s in l8])
        for st in l8:
            got = sorted(p for r in st['ranks'] for mb in r for p in mb)
            self.assertEqual(got, st['members'])
        self.assertEqual([len(r) for r in l8[0]['ranks']], [4] * 8)
        self.assertEqual(len(l8[-1]['members']), 15000 - 234 * 64)

    def test_short_last_step_may_leave_ranks_empty(self):
        l8 = step_layout(64 + 3, 64, 2, 8)
        self.assertEqual(sum(len(r) for r in l8[-1]['ranks']), 3)
        self.assertEqual(sum(1 for r in l8[-1]['ranks'] if not r), 5)

    def test_bad_batch_raises(self):
        with self.assertRaises(ValueError):
            step_layout(100, 60, 4, 8)


class Costs(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        self.stats = {'action_sft': [[1400 + rng.randint(-60, 60), 23, 440] for _ in range(N_ACT)],
                      'geometry_sft_v2': [[1540 + rng.randint(-80, 80), 146, 440] for _ in range(N_GEO)]}

    def test_match_is_at_least_and_minimal(self):
        ag, _ = schedule_v2(self.stats, {'action_sft': 15000, 'geometry_sft_v2': 5000}, 0)
        target = schedule_cost(ag, self.stats, 64, 2, 8)['padded']
        n, c = match_actions(N_ACT, self.stats, target, 0, 64, 2, 8, 'padded')
        self.assertGreaterEqual(c, target)
        seq, _ = schedule_v2({'action_sft': self.stats['action_sft']}, {'action_sft': n - 1}, 0)
        self.assertLess(schedule_cost(seq, self.stats, 64, 2, 8)['padded'], target)
        self.assertGreater(n, 20000)

    def test_learned_tokens_per_kind(self):
        ag, _ = schedule_v2(self.stats, {'action_sft': 15000, 'geometry_sft_v2': 5000}, 1)
        c = schedule_cost(ag, self.stats, 64, 2, 8)
        self.assertEqual(c['learned'], {'action_sft': 15000 * 23, 'geometry_sft_v2': 5000 * 146})


class V1Weighting(unittest.TestCase):
    @unittest.skipUnless(HAVE_YAML, 'PyYAML not installed')
    def test_token_mean_weights(self):
        import audit_v1_weighting as AV
        stats = {'action_sft': [[0, 20, 0]] * 4, 'geometry_sft': [[0, 140, 0]] * 4}
        seq = [('action_sft', 0), ('geometry_sft', 0), ('action_sft', 1), ('action_sft', 2)]
        tot, single = AV.v1_weights(seq, stats, eff=4, micro=2, ws=1)
        # micro-batch 1: action 20/160, geometry 140/160; micro-batch 2: two actions 1/2 each; two micro-batches per step
        self.assertAlmostEqual(tot['action_sft'], (20 / 160 + 1.0) / 2)
        self.assertAlmostEqual(tot['geometry_sft'], (140 / 160) / 2)


@unittest.skipUnless(HAVE_YAML, 'PyYAML not installed')
class ConfigV2(unittest.TestCase):
    def test_exposures_and_calibration(self):
        from lcam import config as C
        cdir = os.path.join(ROOT, 'configs', 'v2')
        self.assertEqual(C.exposures(C.load(os.path.join(cdir, 'AG.yaml')), os.path.join(cdir, 'AG.yaml')),
                         {'action_sft': 15000, 'geometry_sft_v2': 5000})
        with tempfile.TemporaryDirectory() as d:
            for f in ('common.yaml', 'A_compute.yaml'):
                open(os.path.join(d, f), 'w').write(open(os.path.join(cdir, f)).read())
            cfg = C.load(os.path.join(d, 'A_compute.yaml'))
            with self.assertRaises(SystemExit):
                C.exposures(cfg, os.path.join(d, 'A_compute.yaml'))
            json.dump({'frozen': False, 'exposures': {'A_compute': {'action_sft': 21632}}}, open(os.path.join(d, 'calibration.json'), 'w'))
            with self.assertRaises(SystemExit):
                C.exposures(cfg, os.path.join(d, 'A_compute.yaml'))
            json.dump({'frozen': True, 'exposures': {'A_compute': {'action_sft': 21632}}}, open(os.path.join(d, 'calibration.json'), 'w'))
            self.assertEqual(C.exposures(cfg, os.path.join(d, 'A_compute.yaml')), {'action_sft': 21632})

    def test_env_expansion(self):
        from lcam import config as C
        os.environ['LCAM_MODEL'] = '/models/q'
        self.assertEqual(C.load(os.path.join(ROOT, 'configs', 'v2', 'A_base.yaml'))['model']['id'], '/models/q')


if __name__ == '__main__':
    unittest.main()
