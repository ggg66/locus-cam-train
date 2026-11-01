"""v5 method study: rewards, development split, multi-solution targets, configs and the development scorer (no torch)."""
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

from lcam import config as C
from lcam import rewards as R
from lcam.data import ALLOWED, choose_target, load_exports

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, 'configs', 'v5')
DEV = ['F09_cabinet_gap', 'F14_depth_line']     # export_v5.py DEV_FAMILIES


def table(two=False):
    """One root: (+1,+0,-3) satisfied, (+0,+0,+0) unmet with 3 of 6 checks, (-2,-1,-3) illegal."""
    rows = {
        '+1,+0,-3': {'s': 'satisfied', 'p': 6, 'n': 6, 'f': [True, True, True, True, None], 't': [[0.472, 0.163, 0, 9]]},
        '+0,+0,+0': {'s': 'spatial_unmet', 'p': 3, 'n': 6, 'f': [True, True, False, False, None], 't': [[0.301, 0.324, 9, 9]]},
        '-2,-1,-3': {'s': 'illegal_endpoint', 'p': 0, 'n': 0, 'f': [False, None, None, None, None], 't': None},
    }
    return {'root_id': 'train-1', 'names': ['蓝色人物'], 'two': two, 'rows': rows}


def move(r, u, b):
    return json.dumps({'move_m': {'right': r, 'up': u, 'back': b}})


IMAG_OK = json.dumps({'targets': [{'name': '蓝色人物', 'height': 0.47, 'margin': 0.16, 'blocked': 0, 'hits': 9}],
                      'legal': True, 'in_frame': True, 'size_ok': True, 'unoccluded': True, 'apart': None}, ensure_ascii=False)


class ActionReward(unittest.TestCase):
    def test_outcomes(self):
        t = table()
        self.assertEqual(R.action_reward(move(1, 0, -3), t, {})[0], 1.0)
        self.assertAlmostEqual(R.action_reward(move(0, 0, 0), t, {})[0], 0.15)
        self.assertEqual(R.action_reward(move(-2, -1, -3), t, {})[0], 0.0)
        self.assertEqual(R.action_reward('no json', t, {})[0], -0.5)
        self.assertEqual(R.action_reward(move(3, 0, 0), t, {})[1]['outcome'], 'out_of_domain')

    def test_last_json_is_the_action(self):   # the frozen parser: an imagination block before the move does not hide it
        self.assertEqual(R.action_reward(IMAG_OK + '\n' + move(1, 0, -3), table(), {})[0], 1.0)


class ImaginationReward(unittest.TestCase):
    cfg = {'type': 'imagact', 'imagination_weight': 0.5}

    def test_exact_imagination(self):
        r, info = R.reward(IMAG_OK + '\n' + move(1, 0, -3), table(), self.cfg)
        self.assertEqual(info['imagination'], 1.0)
        self.assertAlmostEqual(r, 1.5)

    def test_wrong_fields_and_tolerance(self):
        bad = json.loads(IMAG_OK)
        bad['targets'][0]['height'] = 0.55      # off by 0.08 > 0.03
        bad['size_ok'] = False                  # wrong
        r, info = R.reward(json.dumps(bad, ensure_ascii=False) + '\n' + move(1, 0, -3), table(), self.cfg)
        self.assertAlmostEqual(info['imagination'], 6 / 8)    # legal, in_frame, size_ok, unoccluded, 4 quantities

    def test_illegal_endpoint(self):
        pred = json.dumps({'targets': [], 'legal': False, 'in_frame': None, 'size_ok': None, 'unoccluded': None, 'apart': None})
        self.assertEqual(R.reward(pred + '\n' + move(-2, -1, -3), table(), self.cfg)[1]['imagination'], 1.0)
        self.assertEqual(R.reward(IMAG_OK + '\n' + move(-2, -1, -3), table(), self.cfg)[1]['imagination'], 0.0)

    def test_no_credit_without_a_valid_move(self):
        r, info = R.reward(IMAG_OK, table(), self.cfg)
        self.assertEqual(r, -0.5)
        self.assertIsNone(info['imagination'])

    def test_missing_imagination(self):
        self.assertEqual(R.reward(move(1, 0, -3), table(), self.cfg), (1.0, {'outcome': 'satisfied', 'key': '+1,+0,-3', 'imagination': 0.0}))


class Advantages(unittest.TestCase):
    def test_group(self):
        self.assertEqual(R.group_advantages([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])
        a = R.group_advantages([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(a), 0.0, places=6)
        self.assertGreater(a[0], 0)
        self.assertEqual(R.group_advantages([1.0, 0.0], normalize_std=False), [0.5, -0.5])


class DevSplitAndTargets(unittest.TestCase):
    def test_exclude_families(self):
        with tempfile.TemporaryDirectory() as b:
            os.makedirs(os.path.join(b, 'training_exports', 'train'))
            with open(os.path.join(b, 'training_exports', 'train', 'action_sft.jsonl'), 'w') as f:
                for i, fam in enumerate(['F01_x', 'F09_cabinet_gap', 'F14_depth_line', 'F11_y']):
                    f.write(json.dumps({'id': str(i), 'source_split': 'train', 'family_id': fam}) + '\n')
            self.assertEqual(len(load_exports(b, 'train', ['action_sft'])['action_sft']), 4)
            self.assertEqual([r['family_id'] for r in load_exports(b, 'train', ['action_sft'], DEV)['action_sft']], ['F01_x', 'F11_y'])

    def test_choose_target(self):
        rec = {'target': move(1, 0, -3), 'robust_set': ['+1,+0,-3', '+2,+0,-3', '-2,+0,-3']}
        picks = [choose_target(rec, 'robust_set', 0, p) for p in range(60)]
        self.assertEqual({json.loads(p['target'])['move_m']['right'] for p in picks}, {1, 2, -2})
        self.assertEqual(picks[7], choose_target(rec, 'robust_set', 0, 7))            # fixed by (seed, position)
        self.assertEqual(picks[7]['target'], move(*picks[7]['target_action']))       # the action_sft target format
        self.assertIs(choose_target(rec, None, 0, 0), rec)


class Configs(unittest.TestCase):
    def test_v5_configs(self):
        common = C.load(os.path.join(CFG, 'common.yaml'))
        self.assertEqual(common['data']['exclude_families'], DEV)
        v2 = C.load(os.path.join(HERE, 'configs', 'v2', 'common.yaml'))
        for k in ('trainable', 'optim', 'loss', 'checkpoints', 'evaluation', 'chat'):
            self.assertEqual(v2[k], common[k], k)
        for f in ('base', 'm4_multi', 'm3_inverse', 'm2_imagact'):
            c = C.load(os.path.join(CFG, f'{f}.yaml'))
            self.assertTrue(set(c['exposures']) <= ALLOWED)
        self.assertEqual(C.load(os.path.join(CFG, 'm4_multi.yaml'))['data']['target_choice'], 'robust_set')
        for f, init, kind in (('m1_rl', 'base', 'action'), ('m2_rl', 'm2_imagact', 'imagact')):
            c = C.load(os.path.join(CFG, f'{f}.yaml'))
            self.assertEqual((c['init_from'], c['rl']['reward']['type']), (init, kind))
            self.assertFalse(c['rl'].get('kl'))


class ScoreDev(unittest.TestCase):
    def test_scorer(self):
        with tempfile.TemporaryDirectory() as b:
            for d in ('eval_prompts', 'reward_tables', 'validation_tables', 'out/base-s0', 'out/m1_rl-s0'):
                os.makedirs(os.path.join(b, d))
            held = [{'task_id': f't{i}', 'root_id': f'train-{i}', 'family_id': 'F09_cabinet_gap'} for i in range(4)]
            seen = [{'task_id': 'v0', 'root_id': 'validation-0', 'family_id': 'F01_x'}]
            for name, rows in (('dev_heldfam_V0', held), ('dev_seen_V0', seen)):
                open(os.path.join(b, 'eval_prompts', f'{name}.jsonl'), 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
            with gzip.open(os.path.join(b, 'reward_tables', 'train.jsonl.gz'), 'wt') as f:
                for h in held:
                    f.write(json.dumps(dict(table(), root_id=h['root_id'])) + '\n')
            json.dump({'v0': 'validation-0'}, open(os.path.join(b, 'validation_tables', 'index.json'), 'w'))
            json.dump({'+1,+0,-3': {'status': 'satisfied'}, '+0,+0,+0': {'status': 'spatial_unmet'}},
                      open(os.path.join(b, 'validation_tables', 'validation-0.json'), 'w'))
            for run, oks in (('base-s0', [1, 0, 0, 0]), ('m1_rl-s0', [1, 1, 1, 0])):
                with open(os.path.join(b, 'out', run, 'dev_heldfam.part0.jsonl'), 'w') as f:
                    for h, ok in zip(held, oks):
                        f.write(json.dumps({'task_id': h['task_id'], 'raw': move(1, 0, -3) if ok else move(0, 0, 0)}) + '\n')
                open(os.path.join(b, 'out', run, 'dev_seen.part0.jsonl'), 'w').write(json.dumps({'task_id': 'v0', 'raw': move(1, 0, -3)}) + '\n')
            subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'score_dev.py'), '--bundle', b, '--outputs', os.path.join(b, 'out'),
                            '--out', os.path.join(b, 's.json')], check=True, capture_output=True)
            s = json.load(open(os.path.join(b, 's.json')))
            self.assertEqual(s['methods']['base']['held_success'], 0.25)
            self.assertEqual(s['methods']['m1_rl']['held_minus_base'], 0.5)
            self.assertEqual(s['methods']['m1_rl']['seen_success'], 1.0)
            self.assertEqual(s['runs']['base-s0']['outcomes_held'], {'satisfied': 1, 'spatial_unmet': 3})


if __name__ == '__main__':
    unittest.main()
