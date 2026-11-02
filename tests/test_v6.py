"""v6 final test: the chosen method on every training family, the v2 protocol and the v2 test suite (no torch)."""
import os
import subprocess
import sys
import unittest

from lcam import config as C

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, 'configs')


class Configs(unittest.TestCase):
    def test_v6_is_the_v2_protocol_on_every_family(self):
        v2, v6 = C.load(os.path.join(CFG, 'v2', 'common.yaml')), C.load(os.path.join(CFG, 'v6', 'common.yaml'))
        for k in ('trainable', 'optim', 'loss', 'checkpoints', 'evaluation', 'data', 'chat', 'processor'):
            self.assertEqual(v2[k], v6[k], k)
        self.assertNotIn('exclude_families', v6['data'])
        self.assertEqual(v2['model']['revision'], v6['model']['revision'])
        self.assertEqual(C.families(os.path.join(CFG, 'v6')), ['A_imagine'])

    def test_the_method_is_the_development_one(self):
        dev, fin = C.load(os.path.join(CFG, 'v5', 'm2_imagact.yaml')), C.load(os.path.join(CFG, 'v6', 'A_imagine.yaml'))
        for k in ('anchor', 'pool_streams', 'exposures', 'probe_kinds'):
            self.assertEqual(dev[k], fin[k], k)
        self.assertEqual(fin['eval_min_new_tokens'], 1024)     # the v5 evaluation cap of method 2


class Tasks(unittest.TestCase):
    def test_suite(self):
        env = dict(os.environ, LCAM_ROUND='v6', LCAM_INFER_BATCH='1')
        out = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v2_tasks.py'), 'test'], env=env,
                             check=True, capture_output=True, text=True).stdout.split('\n')
        rows = [l.split('\t') for l in out if l]
        runs = {r[0].split('/')[3] for r in rows}
        self.assertEqual(runs, {'A_imagine-s0', 'A_imagine-s1', 'A_imagine-s2', 'B0-nothink'})
        cap = {(r[0].split('/')[3], os.path.basename(r[0])): (r[2], r[4], r[5]) for r in rows}
        self.assertEqual(cap[('A_imagine-s1', 'test_V0.jsonl')], ('work/runs/v6/A_imagine-s1/checkpoints/ckpt-4', '1', '1024'))
        self.assertEqual(cap[('A_imagine-s1', 'test_diagnostic_consequence.jsonl')][2], '2048')
        self.assertEqual(cap[('B0-nothink', 'test_V0.jsonl')], ('-', '1', '512'))

    def test_runner_parses(self):
        subprocess.run(['bash', '-n', os.path.join(HERE, 'scripts', 'run_v6.sh')], check=True)


if __name__ == '__main__':
    unittest.main()
