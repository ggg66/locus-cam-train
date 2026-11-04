"""v8: two training controls on the main split with the v2 protocol unchanged (no torch)."""
import json
import os
import random
import unittest

from lcam import config as C
from lcam.data import choose_target

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, 'configs')


class Configs(unittest.TestCase):
    def test_same_protocol_as_the_controls(self):
        a, b = C.load(os.path.join(CFG, 'v2', 'common.yaml')), C.load(os.path.join(CFG, 'v8', 'common.yaml'))
        for k in ('trainable', 'optim', 'loss', 'checkpoints', 'evaluation', 'chat', 'processor'):
            self.assertEqual(a[k], b[k], k)
        for k, v in a['model'].items():
            self.assertEqual(b['model'][k], v, f'model.{k}')
        self.assertNotIn('exclude_families', b['data'])
        self.assertEqual(sorted(C.families(os.path.join(CFG, 'v8'))), ['A_cons_full', 'A_multi'])

    def test_multi_differs_from_A_base_only_in_the_labels(self):
        base, multi = C.load(os.path.join(CFG, 'v2', 'A_base.yaml')), C.load(os.path.join(CFG, 'v8', 'A_multi.yaml'))
        self.assertEqual(base['exposures'], multi['exposures'])
        self.assertEqual(multi['data']['target_choice'], 'robust_set')

    def test_cons_full_covers_every_consequence_record_once(self):
        full, v4 = C.load(os.path.join(CFG, 'v8', 'A_cons_full.yaml')), C.load(os.path.join(CFG, 'v4', 'A_cons.yaml'))
        self.assertEqual(full['exposures'], {'action_sft': 15000, 'consequence_v4': 29400})
        self.assertEqual(v4['exposures']['action_sft'], 15000)

    def test_multi_labels_stay_in_the_robust_set(self):
        rec = {'target': '{}', 'target_action': [0, 0, 0], 'robust_set': ['+1,+0,+2', '-1,+0,+2', '+0,+1,-3']}
        drawn = {tuple(choose_target(rec, 'robust_set', 0, pos)['target_action']) for pos in range(200)}
        self.assertEqual(drawn, {(1, 0, 2), (-1, 0, 2), (0, 1, -3)})


if __name__ == '__main__':
    unittest.main()
