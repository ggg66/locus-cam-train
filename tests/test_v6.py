"""v6 final test: the chosen method and its ablation on every training family, on both bases, with the protocol and the
test suite of the finished controls (no torch)."""
import importlib.util
import json
import os
import subprocess
import sys
import unittest

from lcam import config as C
from lcam.data import ALLOWED

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, 'configs')
spec = importlib.util.spec_from_file_location('make_propose_v6', os.path.join(HERE, 'scripts', 'make_propose_v6.py'))
MP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MP)


class Configs(unittest.TestCase):
    def test_same_protocol_as_the_controls(self):
        for ctrl, rnd in (('v2', 'v6'), ('v3_gemma', 'v6_gemma')):
            a, b = C.load(os.path.join(CFG, ctrl, 'common.yaml')), C.load(os.path.join(CFG, rnd, 'common.yaml'))
            for k in ('trainable', 'optim', 'loss', 'checkpoints', 'evaluation', 'data', 'chat', 'processor'):
                self.assertEqual(a[k], b[k], f'{rnd} {k}')
            for k, v in a['model'].items():            # the same weights; v6 also names hf_id / expect_model_type
                self.assertEqual(b['model'][k], v, f'{rnd} model.{k}')
            self.assertNotIn('exclude_families', b['data'])
        self.assertEqual(C.families(os.path.join(CFG, 'v6')), ['A_imagine', 'A_propose'])
        self.assertEqual(C.families(os.path.join(CFG, 'v6_gemma')), ['A_imagine'])

    def test_the_method_is_the_development_one(self):
        dev = C.load(os.path.join(CFG, 'v5', 'm2_imagact.yaml'))
        for rnd in ('v6', 'v6_gemma'):
            fin = C.load(os.path.join(CFG, rnd, 'A_imagine.yaml'))
            for k in ('anchor', 'pool_streams', 'exposures', 'probe_kinds'):
                self.assertEqual(dev[k], fin[k], f'{rnd} {k}')
            self.assertEqual(fin['eval_min_new_tokens'], 1024)     # the v5 evaluation cap of method 2

    def test_the_ablation_differs_only_in_the_source(self):
        a, p = C.load(os.path.join(CFG, 'v6', 'A_imagine.yaml')), C.load(os.path.join(CFG, 'v6', 'A_propose.yaml'))
        self.assertIn('propose_v6', ALLOWED)
        self.assertEqual((p['anchor'], p['pool_streams'], p['exposures']), ('propose_v6', {'propose_v6': 'action_sft'}, {'propose_v6': 15000}))
        self.assertEqual(a['eval_min_new_tokens'], p['eval_min_new_tokens'])


class Propose(unittest.TestCase):
    def test_outcomes_removed_moves_and_choice_kept(self):
        cands = [{'move': [1, 0, -2], 'targets': [{'name': '蓝色人物', 'height': 0.4, 'margin': 0.23, 'blocked': 0, 'hits': 9}],
                  'legal': True, 'in_frame': True, 'size_ok': True, 'unoccluded': True, 'apart': None}] + \
                [{'move': [k, 0, 0], 'targets': [], 'legal': False, 'in_frame': None, 'size_ok': None, 'unoccluded': None, 'apart': None} for k in (-1, 0, 2)]
        move = '{"move_m": {"right": 1, "up": 0, "back": -2}}'
        rec = {'id': 'train-00001:imagact', 'export_version': 'export-v5.1.0', 'root_id': 'train-1', 'prompt': 'P',
               'target': json.dumps({'candidates': cands}, ensure_ascii=False) + '\n' + move}
        out = MP.propose(rec)
        plan, last = out['target'].split('\n')
        self.assertEqual(json.loads(plan), {'candidates': [{'move': [1, 0, -2]}, {'move': [-1, 0, 0]}, {'move': [0, 0, 0]}, {'move': [2, 0, 0]}]})
        self.assertEqual(last, move)
        self.assertEqual((out['id'], out['root_id'], out['prompt']), ('train-00001:propose', 'train-1', 'P'))
        with self.assertRaises(ValueError):
            MP.propose(dict(rec, target=move + '\n' + move))


class Tasks(unittest.TestCase):
    def tasks(self, rnd, what='test'):
        env = dict(os.environ, LCAM_ROUND=rnd, LCAM_INFER_BATCH='1')
        out = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v2_tasks.py'), what], env=env,
                             check=True, capture_output=True, text=True).stdout.split('\n')
        return [l.split('\t') for l in out if l]

    def test_suite(self):
        rows = self.tasks('v6')
        self.assertEqual({r[0].split('/')[3] for r in rows}, {f'{f}-s{s}' for f in ('A_imagine', 'A_propose') for s in range(3)} | {'B0-nothink'})
        cap = {(r[0].split('/')[3], os.path.basename(r[0])): (r[2], r[4], r[5]) for r in rows}
        self.assertEqual(cap[('A_imagine-s1', 'test_V0.jsonl')], ('work/runs/v6/A_imagine-s1/checkpoints/ckpt-4', '1', '1024'))
        self.assertEqual(cap[('A_propose-s2', 'test_V0.jsonl')][2], '1024')
        self.assertEqual(cap[('A_imagine-s1', 'test_diagnostic_consequence.jsonl')][2], '2048')
        self.assertEqual(cap[('B0-nothink', 'test_V0.jsonl')], ('-', '1', '512'))
        g = self.tasks('v6_gemma')
        self.assertEqual({r[0].split('/')[3] for r in g}, {'A_imagine-s0', 'A_imagine-s1', 'A_imagine-s2', 'B0-nothink'})
        self.assertTrue(all(r[0].startswith('work/outputs/v6_gemma/') for r in g))
        self.assertEqual(len(self.tasks('v6', 'val')), 2 * 3 * 4)          # every checkpoint of every run

    def test_runner_parses(self):
        subprocess.run(['bash', '-n', os.path.join(HERE, 'scripts', 'run_v6.sh')], check=True)


if __name__ == '__main__':
    unittest.main()
