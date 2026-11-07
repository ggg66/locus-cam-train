"""v10: geometry-only runs on random structures and on the training families, and their test tasks (no torch)."""
import os
import subprocess
import sys
import tempfile
import unittest

from lcam import config as C
from lcam.data import ALLOWED

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, 'configs')


class Configs(unittest.TestCase):
    def test_same_protocol_as_v2(self):
        a, b = C.load(os.path.join(CFG, 'v2', 'common.yaml')), C.load(os.path.join(CFG, 'v10', 'common.yaml'))
        for k in ('trainable', 'optim', 'checkpoints', 'evaluation', 'chat', 'processor', 'data'):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(a['loss']['reduction'], b['loss']['reduction'])
        for k, v in a['loss']['sample_weight'].items():
            self.assertEqual(b['loss']['sample_weight'][k], v)
        self.assertEqual(b['loss']['sample_weight']['geometry_rand_v10'], 1.0)
        for k, v in a['model'].items():
            self.assertEqual(b['model'][k], v, f'model.{k}')
        self.assertEqual(sorted(C.families(os.path.join(CFG, 'v10'))), ['G_fam', 'G_rand'])

    def test_arms_differ_only_in_the_source(self):
        r, f = C.load(os.path.join(CFG, 'v10', 'G_rand.yaml')), C.load(os.path.join(CFG, 'v10', 'G_fam.yaml'))
        self.assertEqual(r['exposures'], {'geometry_rand_v10': 15000})
        self.assertEqual(f['exposures'], {'geometry_sft_v2': 15000})
        self.assertEqual(r['probe_kinds'], f['probe_kinds'])
        self.assertIn('geometry_rand_v10', ALLOWED)
        for k in ('trainable', 'optim', 'loss', 'checkpoints', 'evaluation', 'data', 'model'):
            self.assertEqual(r[k], f[k], k)


class Tasks(unittest.TestCase):
    def test_geometry_prompts_on_every_endpoint(self):
        with tempfile.TemporaryDirectory() as root:
            for f in ('G_rand', 'G_fam'):
                for s in (0, 1, 2):
                    d = os.path.join(root, 'work', 'runs', 'v10', f'{f}-s{s}', 'checkpoints', 'ckpt-4')
                    os.makedirs(d)
                    open(os.path.join(d, 'adapter_model.safetensors'), 'w').close()
            p = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v10_tasks.py'), '--root', root],
                               capture_output=True, text=True, check=True)
            rows = [line.split('\t') for line in p.stdout.splitlines() if line]
            self.assertEqual(p.stderr, '')
            self.assertEqual(len(rows), 12)
            self.assertEqual({r[1] for r in rows}, {'eval_prompts/test_geom_predict.jsonl', 'eval_prompts/test_diagnostic_geom_predict.jsonl'})
            self.assertTrue(all(r[2].endswith('/checkpoints/ckpt-4') and r[3] == '0' and r[4] == '1' and r[5] == '1024' for r in rows))
            self.assertTrue(all(r[0].startswith('work/outputs/v10/') for r in rows))

    def test_missing_endpoint_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            p = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v10_tasks.py'), '--root', root],
                               capture_output=True, text=True, check=True)
            self.assertEqual(p.stdout, '')
            self.assertEqual(len(p.stderr.split()), 6)


if __name__ == '__main__':
    unittest.main()
