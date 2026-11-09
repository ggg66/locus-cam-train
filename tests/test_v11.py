"""v11: decisions, geometry-then-decision and geometry on random scenes; geometry from 10 random structures; blind decisions (no torch)."""
import os
import subprocess
import sys
import tempfile
import unittest

from lcam import config as C
from lcam.data import ALLOWED, schedule_v2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, 'configs')
FAMS = ['A_rand', 'D_fam', 'D_rand', 'GA_rand', 'G_k10', 'G_rand_r']


def cfg(f):
    return C.load(os.path.join(CFG, 'v11', f'{f}.yaml'))


class Configs(unittest.TestCase):
    def test_same_protocol_as_v2(self):
        a, b = C.load(os.path.join(CFG, 'v2', 'common.yaml')), C.load(os.path.join(CFG, 'v11', 'common.yaml'))
        for k in ('trainable', 'optim', 'checkpoints', 'evaluation', 'chat', 'processor', 'data'):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(a['loss']['reduction'], b['loss']['reduction'])
        for k, v in a['model'].items():
            self.assertEqual(b['model'][k], v, f'model.{k}')
        self.assertEqual(sorted(C.families(os.path.join(CFG, 'v11'))), sorted(FAMS))

    def test_one_source_per_run_and_matching_exposures(self):
        exp = {'A_rand': {'action_rand_v11': 15000}, 'GA_rand': {'geomact_rand_v11': 19072}, 'G_rand_r': {'geometry_randr_v11': 15000},
               'G_k10': {'geometry_randk10_v11': 15000}, 'D_fam': {'blind_fam_v11': 15000}, 'D_rand': {'blind_rand_v11': 15000}}
        a_base, a_geomact = C.load(os.path.join(CFG, 'v2', 'A_base.yaml')), C.load(os.path.join(CFG, 'v10', 'G_fam.yaml'))
        self.assertEqual(a_base['exposures']['action_sft'], 15000)
        self.assertEqual(a_geomact['exposures']['geometry_sft_v2'], 15000)
        for f, e in exp.items():
            c = cfg(f)
            self.assertEqual(c['exposures'], e, f)
            (kind,) = e
            self.assertEqual(c['anchor'], kind, f)                        # the sole source anchors the schedule (v10 fix)
            self.assertIn(kind, ALLOWED)
            self.assertEqual(c['loss']['sample_weight'][kind], 1.0)
        self.assertEqual(cfg('GA_rand')['pool_streams'], {'geomact_rand_v11': 'action_rand_v11'})
        self.assertEqual(int(cfg('GA_rand')['eval_min_new_tokens']), 1024)

    def test_schedule_runs_with_the_anchor(self):
        for f in FAMS:
            c = cfg(f)
            (kind, n), = c['exposures'].items()
            seq, counts = schedule_v2({kind: list(range(5053))}, c['exposures'], 0, anchor=c['anchor'], streams=c.get('pool_streams'))
            self.assertEqual(len(seq), n)
            self.assertEqual(counts, {kind: n})

    def test_geomact_roots_follow_action_order(self):
        a, g = cfg('A_rand'), cfg('GA_rand')
        sa, _ = schedule_v2({'action_rand_v11': list(range(5053))}, a['exposures'], 1, anchor=a['anchor'])
        sg, _ = schedule_v2({'geomact_rand_v11': list(range(5053))}, g['exposures'], 1, anchor=g['anchor'], streams=g['pool_streams'])
        self.assertEqual([i for _, i in sa], [i for _, i in sg][:15000])


class Tasks(unittest.TestCase):
    def test_prompts_on_every_endpoint(self):
        with tempfile.TemporaryDirectory() as root:
            for f in FAMS:
                for s in (0, 1, 2):
                    d = os.path.join(root, 'work', 'runs', 'v11', f'{f}-s{s}', 'checkpoints', 'ckpt-4')
                    os.makedirs(d)
                    open(os.path.join(d, 'adapter_model.safetensors'), 'w').close()
            p = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v11_tasks.py'), '--root', root],
                               capture_output=True, text=True, check=True)
            rows = [line.split('\t') for line in p.stdout.splitlines() if line]
            self.assertEqual(p.stderr, '')
            by = {}
            for r in rows:
                by.setdefault(r[1], []).append(r)
            want = {'eval_prompts/test_V0.jsonl': 6, 'eval_prompts/test_geomact_true.jsonl': 3,
                    'eval_prompts/test_geomact_mirror.jsonl': 3, 'eval_prompts/test_geom_predict.jsonl': 6, 'eval_prompts/test_blind_true.jsonl': 6}
            for k in (0, 1, 2):
                want[f'eval_prompts/test_blind_grand_s{k}.jsonl'] = 2
                want[f'eval_prompts/test_blind_gfam_s{k}.jsonl'] = 2
            self.assertEqual({k: len(v) for k, v in by.items()}, want)
            for r in rows:                                               # a blind run reads the predictions of its own seed
                if '/D_' in r[0] and ('grand' in r[1] or 'gfam' in r[1]):
                    self.assertEqual(r[0].split('-s')[1][0], r[1].split('_s')[-1][0])
            self.assertTrue(all(r[2].endswith('/checkpoints/ckpt-4') and r[3] == '0' and r[4] == '1' for r in rows))
            self.assertEqual({r[5] for r in rows if '/A_rand-' in r[0]}, {'512'})
            self.assertEqual({r[5] for r in by['eval_prompts/test_V0.jsonl'] if '/GA_rand-' in r[0]}, {'1024'})

    def test_missing_endpoint_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            p = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v11_tasks.py'), '--root', root],
                               capture_output=True, text=True, check=True)
            self.assertEqual(p.stdout, '')
            self.assertEqual(len(p.stderr.split()), 18)


if __name__ == '__main__':
    unittest.main()
