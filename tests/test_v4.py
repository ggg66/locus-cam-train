"""v4 supervision study: config discovery, sample order and calibration wiring (no torch)."""
import importlib.util
import os
import subprocess
import sys
import unittest

from lcam import config as C
from lcam.costs import match_actions, match_kind
from lcam.data import ALLOWED, schedule_v2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, 'configs')


def load_script(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, 'scripts', f'{name}.py'))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class Families(unittest.TestCase):
    def test_rounds(self):
        self.assertEqual(C.families(os.path.join(CFG, 'v2')), ['A_base', 'AG', 'A_compute'])
        self.assertEqual(C.families(os.path.join(CFG, 'v3_gemma')), ['A_base', 'AG', 'A_compute'])
        self.assertEqual(C.families(os.path.join(CFG, 'v4')), ['A_cons', 'A_geomact', 'A_nextview'])   # AG is reference only

    def test_round_info(self):
        info = load_script('round_info').info(os.path.join(CFG, 'v4'))
        self.assertEqual(info['calibrated'], ['A_cons', 'A_geomact', 'A_nextview'])
        self.assertTrue({'consequence_v4', 'geomact_v4', 'nextview_v4'} <= set(info['kinds']) <= ALLOWED)
        self.assertEqual(info['lora_on'], ['attention', 'shared_expert'])     # the v2 base, unchanged

    def test_v4_keeps_the_v2_protocol(self):
        v2, v4 = C.load(os.path.join(CFG, 'v2', 'common.yaml')), C.load(os.path.join(CFG, 'v4', 'common.yaml'))
        for k in ('trainable', 'optim', 'loss', 'checkpoints', 'evaluation', 'data', 'chat'):
            self.assertEqual(v2[k], v4[k], k)
        self.assertEqual(v2['model']['revision'], v4['model']['revision'])


class EvalCap(unittest.TestCase):
    def test_only_geomact_gets_a_higher_v0_cap(self):
        env = dict(os.environ, LCAM_ROUND='v4')
        for what in ('test', 'val'):
            out = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v2_tasks.py'), what], env=env, capture_output=True, text=True, check=True).stdout
            lines = [l.split('\t') for l in out.splitlines() if '_V0.jsonl' in l.split('\t')[1] and 'V0' in l.split('\t')[1]]
            self.assertTrue(lines)
            for t in lines:
                self.assertEqual(t[5], '1024' if 'A_geomact' in t[0] else '512', t[0])


class Order(unittest.TestCase):
    def test_geomact_follows_a_base_roots(self):
        recs = {'action_sft': range(50), 'geomact_v4': range(50)}
        a_base, _ = schedule_v2(recs, {'action_sft': 120}, seed=1)
        geo, _ = schedule_v2(recs, {'geomact_v4': 180}, seed=1, anchor='geomact_v4', streams={'geomact_v4': 'action_sft'})
        self.assertEqual([i for _, i in a_base], [i for _, i in geo][:120])

    def test_aux_does_not_move_actions(self):
        recs = {'action_sft': range(50), 'consequence_v4': range(300)}
        a_base, _ = schedule_v2(recs, {'action_sft': 120}, seed=2)
        cons, _ = schedule_v2(recs, {'action_sft': 120, 'consequence_v4': 40}, seed=2)
        self.assertEqual([i for _, i in a_base], [i for k, i in cons if k == 'action_sft'])


class Calibration(unittest.TestCase):
    def test_match_kind_equals_match_actions_for_actions_only(self):
        stats = {'action_sft': [[100 + (i * 7) % 40, 20] for i in range(60)]}
        for target in (5000, 12345):
            self.assertEqual(match_kind(stats, {}, 'action_sft', target, 0, 4, 1, 1)[0], match_actions(60, stats, target, 0, 4, 1, 1)[0])

    def test_calibrated_families(self):
        cal = load_script('calibrate_compute').calibrated_families(os.path.join(CFG, 'v4'))
        self.assertEqual({f: (v, fx) for f, (c, v, fx) in cal.items()},
                         {'A_cons': ('consequence_v4', {'action_sft': 15000}), 'A_geomact': ('geomact_v4', {}),
                          'A_nextview': ('nextview_v4', {'action_sft': 15000})})
        v2 = load_script('calibrate_compute').calibrated_families(os.path.join(CFG, 'v2'))
        self.assertEqual({f: v for f, (c, v, fx) in v2.items()}, {'A_compute': 'action_sft'})


if __name__ == '__main__':
    unittest.main()
