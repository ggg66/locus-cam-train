"""C2 prompts of hosted models from their own V0 answers (no network)."""
import importlib.util
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'scripts'))
_s = importlib.util.spec_from_file_location('run_api_diagnostics', os.path.join(HERE, 'scripts', 'run_api_diagnostics.py'))
RD = importlib.util.module_from_spec(_s)
_s.loader.exec_module(RD)
C1 = '任务文字\n\n下面是 12 个候选挪法（含义同上）：\nA ...'


def c1(t):
    return {'task_id': t, 'root_id': 'r', 'split': 'test_id', 'sample': 0, 'images': ['a', 'b'], 'system': 's', 'prompt': C1, 'condition': 'x'}


class BuildC2(unittest.TestCase):
    def test_actions_and_no_action(self):
        v0 = [{'task_id': 'a', 'raw': 'text {"move_m": {"right": 1, "up": 0, "back": -2}}'}, {'task_id': 'b', 'raw': ''}, {'task_id': 'c', 'raw': json.dumps({'right': 9, 'up': 0, 'back': 0})}]
        recs, none = RD.build_c2([c1('a'), c1('b'), c1('c')], v0)
        self.assertEqual([r['task_id'] for r in recs], ['a'])
        self.assertEqual(none, ['b', 'c'])
        self.assertEqual((recs[0]['eval_action'], recs[0]['condition'], recs[0]['split']), ('+1,+0,-2', 'confirm_c2_judge', 'test_id'))
        self.assertIn('待评估动作：right 1，up 0，back -2', recs[0]['prompt'])

    def test_missing_v0_stops(self):
        with self.assertRaises(SystemExit):
            RD.build_c2([c1('a')], [])


if __name__ == '__main__':
    unittest.main()
