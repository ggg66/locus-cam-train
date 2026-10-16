"""Stdlib-only tests (no torch): JSON spans, mask paths on real exports, schedule counts, parse parity.
  python3 -m unittest discover -s tests            # LCAM_EXPORTS=<bundle>/training_exports also checks every real record
"""
import collections
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcam.parse import parse  # noqa: E402
from lcam.spans import dumps_with_spans, masked_char_spans, token_mask  # noqa: E402
from lcam.targets import classify  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('lcam_data', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lcam', 'data.py'))
data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data)


class Spans(unittest.TestCase):
    def test_roundtrip_and_paths(self):
        obj = {'room': {'left': -3.0, 'front': -18, 'ceiling': 2.05}, 'objects': [{'name': '蓝色人物', 'centre': [-0.25, -0.75, -8.2], 'height': 1.7}]}
        text, spans = dumps_with_spans(obj)
        self.assertEqual(text, json.dumps(obj, ensure_ascii=False))
        a, b = spans['objects[0].centre[1]']
        self.assertEqual(text[a:b], '-0.75')
        self.assertEqual(text[slice(*spans['room.ceiling'])], '2.05')
        sub = masked_char_spans(text, ['objects[0]'])
        self.assertEqual(len(sub), 5)  # name, 3 centre values, height

    def test_missing_path_raises(self):
        with self.assertRaises(KeyError):
            masked_char_spans(json.dumps({'a': 1}), ['b'])

    def test_token_mask(self):
        self.assertEqual(token_mask([(0, 2), (2, 5), (5, 6)], [(3, 4)]), [False, True, False])

    def test_real_exports(self):
        root = os.environ.get('LCAM_EXPORTS')
        if not root:
            self.skipTest('set LCAM_EXPORTS to check real records')
        n = 0
        for split in ('train', 'validation'):
            with open(os.path.join(root, split, 'geometry_sft.jsonl'), encoding='utf-8') as fh:
                lines = fh.readlines()
            for line in lines:
                r = json.loads(line)
                spans = masked_char_spans(r['target'], r['loss_mask_paths'])
                self.assertEqual(len(spans) >= len(r['loss_mask_paths']), True)
                n += 1
        self.assertGreater(n, 0)


class Schedule(unittest.TestCase):
    def test_counts_and_passes(self):
        recs = {'action_sft': list(range(10)), 'geometry_sft': list(range(7))}
        seq, counts = data.schedule(recs, {'action_sft': .667, 'geometry_sft': .333}, 30, seed=0)
        self.assertEqual(len(seq), 30)
        self.assertEqual(counts, {'action_sft': 20, 'geometry_sft': 10})
        c = collections.Counter(seq)
        self.assertTrue(all(v <= 3 for v in c.values()))  # 20 over 10 records = exactly 2 passes
        self.assertEqual(sum(1 for k, _ in seq if k == 'action_sft'), 20)
        seq2, _ = data.schedule(recs, {'action_sft': .667, 'geometry_sft': .333}, 30, seed=0)
        self.assertEqual(seq, seq2)

    def test_disallowed_sources(self):
        with self.assertRaises(ValueError):
            data.load_exports('/nonexistent', 'train', ['outcome_records'])
        with self.assertRaises(ValueError):
            data.load_exports('/nonexistent', 'train', ['geometry_sft_privileged'])


class Targets(unittest.TestCase):
    def test_categories(self):
        L = 'model.language_model.layers.3'
        self.assertEqual(classify(f'{L}.self_attn.q_proj'), 'attention')
        self.assertEqual(classify(f'{L}.linear_attn.in_proj_qkvz'), 'attention')
        self.assertEqual(classify(f'{L}.mlp.shared_expert.up_proj'), 'shared_expert')
        self.assertEqual(classify(f'{L}.mlp.shared_expert_gate'), 'shared_expert_gate')
        self.assertEqual(classify(f'{L}.mlp.experts.17.down_proj'), 'routed_expert')
        self.assertEqual(classify(f'{L}.mlp.gate'), 'router')
        self.assertEqual(classify('model.visual.blocks.0.attn.qkv'), 'vision')
        self.assertEqual(classify('lm_head'), 'embedding')


class Parse(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(parse('x {"move_m": {"right": 1, "up": 0, "back": -2}}')[0], (1, 0, -2))
        self.assertEqual(parse('{"move_m": {"right": 0.5, "up": 0, "back": 0}}')[1], 'out_of_domain')
        self.assertEqual(parse('{"move_m": {"right": "1", "up": 0, "back": 0}}')[1], 'format_error')
        self.assertEqual(parse('')[1], 'format_error')


if __name__ == '__main__':
    unittest.main()
