"""v7: inference-only tasks on finished checkpoints (no torch)."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tasks(rnd, root):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', 'make_v7_tasks.py'), '--root', root],
                       env=dict(os.environ, LCAM_ROUND=rnd), capture_output=True, text=True, check=True)
    return [l.split('\t') for l in p.stdout.splitlines() if l and not l.startswith('#')], p.stderr


def fake_ckpt(root, rnd, tag):
    d = os.path.join(root, 'work', 'runs', rnd, tag, 'checkpoints', 'ckpt-4')
    os.makedirs(d)
    open(os.path.join(d, 'adapter_model.safetensors'), 'w').close()


class Tasks(unittest.TestCase):
    def test_every_round_and_family(self):
        with tempfile.TemporaryDirectory() as root:
            want = {'v2': ['A_base', 'AG', 'A_compute'], 'v3_gemma': ['A_base', 'AG', 'A_compute'],
                    'v4': ['A_cons', 'A_geomact', 'A_nextview'], 'v6': ['A_imagine', 'A_propose'], 'v6_gemma': ['A_imagine']}
            for rnd, fams in want.items():
                for f in fams:
                    for s in (0, 1, 2):
                        fake_ckpt(root, rnd, f'{f}-s{s}')
            for rnd, fams in want.items():
                rows, err = tasks(rnd, root)
                self.assertEqual(err, '')
                prompts = [(r[0].split('/')[4], r[1]) for r in rows]
                self.assertIn(('B0-nothink', 'eval_prompts/test_diagnostic_S.jsonl'), prompts)
                for f in fams:
                    for s in (0, 1, 2):
                        self.assertIn((f'{f}-s{s}', 'eval_prompts/test_diagnostic_S.jsonl'), prompts)
                self.assertEqual(sum(p == 'eval_prompts/test_geom_predict.jsonl' for _, p in prompts), 3 if 'AG' in fams else 0)
                self.assertEqual(sum(p == 'eval_prompts/test_geomact_true.jsonl' for _, p in prompts), 3 if 'A_geomact' in fams else 0)
                self.assertEqual(sum(p == 'eval_prompts/test_geomact_mirror.jsonl' for _, p in prompts), 3 if 'A_geomact' in fams else 0)
                for r in rows:
                    self.assertEqual(r[4], '1')                       # batch 1, as in every earlier round
                    if r[0].split('/')[4] != 'B0-nothink':
                        self.assertTrue(r[2].endswith('/checkpoints/ckpt-4'))
                caps = {r[1]: int(r[5]) for r in rows if r[0].split('/')[4].startswith(('A_imagine', 'A_propose', 'A_geomact'))}
                if 'eval_prompts/test_diagnostic_S.jsonl' in caps:
                    self.assertEqual(caps['eval_prompts/test_diagnostic_S.jsonl'], 1024)   # answers that write more than the action

    def test_missing_checkpoint_is_skipped_and_named(self):
        with tempfile.TemporaryDirectory() as root:
            fake_ckpt(root, 'v4', 'A_geomact-s0')
            rows, err = tasks('v4', root)
            self.assertIn('A_cons-s0', err)
            self.assertNotIn('A_geomact-s0', err)
            self.assertEqual({r[0].split('/')[4] for r in rows}, {'B0-nothink', 'A_geomact-s0'})


class Prefix(unittest.TestCase):
    def test_infer_continues_the_prefix(self):
        src = open(os.path.join(HERE, 'scripts', 'infer.py')).read()
        self.assertIn("prompt_text(processor, msgs, enable_thinking=a.thinking) + it.get('assistant_prefix', '')", src)


if __name__ == '__main__':
    unittest.main()
