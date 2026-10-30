"""C2 prompts rebuilt on the server (scripts/make_confirm_tasks.py) must equal the frozen 2026-10-26 prompts byte for
byte. Needs the confirm add-on prompts: set LCAM_CONFIRM_PROMPTS to a directory holding confirm_c1_direct_choice.jsonl
and confirm_c2_v0_*.jsonl (the bundle's eval_prompts/); skipped otherwise."""
import glob
import importlib.util
import json
import os
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_s = importlib.util.spec_from_file_location('mct', os.path.join(HERE, 'scripts', 'make_confirm_tasks.py'))
MCT = importlib.util.module_from_spec(_s)
_s.loader.exec_module(MCT)
D = os.environ.get('LCAM_CONFIRM_PROMPTS')


@unittest.skipUnless(D and os.path.isdir(D), 'LCAM_CONFIRM_PROMPTS not set')
class FrozenC2(unittest.TestCase):
    def test_rebuild_matches_frozen(self):
        c1 = {r['task_id']: r for r in MCT.read(os.path.join(D, 'confirm_c1_direct_choice.jsonl'))}
        files = sorted(glob.glob(os.path.join(D, 'confirm_c2_v0_*.jsonl'))) + [os.path.join(D, 'confirm_c2_reference.jsonl')]
        n = 0
        for f in files:
            for frozen in MCT.read(f):
                act = [int(x) for x in frozen['eval_action'].split(',')]
                rebuilt = MCT.c2_record(c1[frozen['task_id']], act)
                for k in ('prompt', 'images', 'system', 'condition', 'eval_action', 'task_id', 'split'):
                    self.assertEqual(rebuilt[k], frozen[k], f'{os.path.basename(f)} {frozen["task_id"]} {k}')
                n += 1
        self.assertGreater(n, 1000)


if __name__ == '__main__':
    unittest.main()
