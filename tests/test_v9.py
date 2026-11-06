"""v9: the whole feasible set and dense consequence supervision on the main split, and the base-prompt tasks (no torch)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from lcam import config as C
from lcam.data import ALLOWED, choose_target
from lcam.parse import parse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, 'configs')


class Configs(unittest.TestCase):
    def test_same_protocol_as_the_controls(self):
        a, b = C.load(os.path.join(CFG, 'v2', 'common.yaml')), C.load(os.path.join(CFG, 'v9', 'common.yaml'))
        for k in ('trainable', 'optim', 'loss', 'checkpoints', 'evaluation', 'chat', 'processor'):
            self.assertEqual(a[k], b[k], k)
        for k, v in a['model'].items():
            self.assertEqual(b['model'][k], v, f'model.{k}')
        self.assertNotIn('exclude_families', b['data'])
        self.assertEqual(sorted(C.families(os.path.join(CFG, 'v9'))), ['A_cons_dense', 'A_set'])

    def test_set_differs_from_A_base_only_in_the_answer(self):
        base, s = C.load(os.path.join(CFG, 'v2', 'A_base.yaml')), C.load(os.path.join(CFG, 'v9', 'A_set.yaml'))
        self.assertEqual(base['exposures'], s['exposures'])
        self.assertEqual(s['data']['target_choice'], 'robust_set_prefix')
        self.assertGreaterEqual(int(s['eval_min_new_tokens']), 1024)

    def test_dense_covers_every_consequence_v9_record_once(self):
        d = C.load(os.path.join(CFG, 'v9', 'A_cons_dense.yaml'))
        self.assertEqual(d['exposures'], {'action_sft': 15000, 'consequence_v9': 111634})
        self.assertIn('consequence_v9', ALLOWED)
        self.assertEqual(d['probe_kinds'], ['action_sft', 'consequence_v4'])


class SetTarget(unittest.TestCase):
    rec = {'target': json.dumps({'move_m': {'right': 1, 'up': 0, 'back': -3}}), 'target_action': [1, 0, -3],
           'robust_set': ['+2,+0,-3', '+1,+0,-3', '-2,+0,-3']}

    def test_lists_the_set_then_keeps_the_action(self):
        t = choose_target(self.rec, 'robust_set_prefix', 0, 5)['target']
        first, second = t.split('\n')
        self.assertEqual(json.loads(first), {'feasible_moves': [[-2, 0, -3], [1, 0, -3], [2, 0, -3]]})
        self.assertEqual(second, self.rec['target'])
        self.assertEqual(parse(t), ((1, 0, -3), None))          # the scorer reads the action from the last block

    def test_does_not_depend_on_seed_or_position(self):
        self.assertEqual(choose_target(self.rec, 'robust_set_prefix', 0, 1), choose_target(self.rec, 'robust_set_prefix', 2, 99))


class BaseTasks(unittest.TestCase):
    def rows(self, base):
        p = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v9_tasks.py'), '--base', base], capture_output=True, text=True, check=True)
        return [l.split('\t') for l in p.stdout.splitlines() if l]

    def test_three_conditions_no_adapter(self):
        for base in ('qwen', 'gemma'):
            rows = self.rows(base)
            self.assertEqual(len(rows), 62)
            self.assertTrue(all(r[2] == '-' and r[4] == '1' for r in rows))
            self.assertTrue(all(r[0].startswith(f'work/outputs/v9-base/{base}/') for r in rows))
            think = [r for r in rows if r[3] == '1']
            self.assertEqual(len(think), 60)
            self.assertTrue(all(r[1].startswith('eval_prompts/v9_think/test_V0_part-') and r[5] == '8192' for r in think))
            self.assertEqual({r[1] for r in rows if r[3] == '0'}, {'eval_prompts/test_V0_example_random.jsonl', 'eval_prompts/test_V0_schema.jsonl'})


class Probes(unittest.TestCase):
    def test_dense_gets_the_consequence_probes(self):
        with tempfile.TemporaryDirectory() as root:
            for f in ('A_cons_dense', 'A_set'):
                for s in (0, 1, 2):
                    d = os.path.join(root, 'work', 'runs', 'v9', f'{f}-s{s}', 'checkpoints', 'ckpt-4')
                    os.makedirs(d)
                    open(os.path.join(d, 'adapter_model.safetensors'), 'w').close()
            p = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v7_tasks.py'), '--root', root],
                               env=dict(os.environ, LCAM_ROUND='v9'), capture_output=True, text=True, check=True)
            rows = [l.split('\t') for l in p.stdout.splitlines() if l and not l.startswith('#')]
            self.assertEqual(p.stderr, '')
            cons = [r for r in rows if r[1] in ('eval_prompts/test_cons_panel.jsonl', 'eval_prompts/validation_cons.jsonl')]
            self.assertEqual(len(cons), 6)
            self.assertTrue(all('/A_cons_dense-' in r[0] for r in cons))
            s_set = [r for r in rows if '/A_set-' in r[0] and r[1] == 'eval_prompts/test_diagnostic_S.jsonl']
            self.assertEqual(len(s_set), 3)
            self.assertTrue(all(int(r[5]) >= 1024 for r in s_set))


if __name__ == '__main__':
    unittest.main()
