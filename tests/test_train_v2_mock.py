"""scripts/train_v2.py end to end with a fake LoRA model, fake records and a fake encoder (no transformers, no
weights, CPU). Checks the training loop itself: exposures, checkpoints at the planned fractions, the warm-up-aware
gradient / update checks, probe losses, the calibrated A_compute budget, and that splitting one optimizer step into
different micro-batches gives the same gradient. Skipped without torch, PyYAML or Pillow.
  python3 -m unittest tests.test_train_v2_mock
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
HAVE = all(importlib.util.find_spec(m) is not None for m in ('torch', 'yaml', 'PIL'))


def fake_hf_modules():
    """Minimal stand-ins when transformers / peft are not installed (train_v2 only reads versions and the scheduler)."""
    import torch
    if 'transformers' not in sys.modules and importlib.util.find_spec('transformers') is None:
        tf = types.ModuleType('transformers')
        tf.__version__ = 'fake'

        def get_cosine_schedule_with_warmup(opt, warm, total):
            def f(step):
                if step < warm:
                    return step / max(1, warm)
                return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            return torch.optim.lr_scheduler.LambdaLR(opt, f)
        tf.get_cosine_schedule_with_warmup = get_cosine_schedule_with_warmup
        sys.modules['transformers'] = tf
    if 'peft' not in sys.modules and importlib.util.find_spec('peft') is None:
        pf = types.ModuleType('peft')
        pf.__version__ = 'fake'
        sys.modules['peft'] = pf


@unittest.skipUnless(HAVE, 'torch, PyYAML and Pillow needed')
class TrainV2Mock(unittest.TestCase):
    def setUp(self):
        import torch
        import yaml
        fake_hf_modules()
        import train_v2 as T
        self.T, self.torch = T, torch
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        common = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'v2', 'common.yaml')))
        common['optim'].update(effective_batch_size=4, lr=0.01, warmup_ratio=0.5)   # 3 steps for AG, the first at lr 0
        common['optim'].pop('micro_batch')            # the fake model has no batch interaction; several micro-batch sizes are tested
        self.cdir = os.path.join(self.dir, 'cfg')
        os.makedirs(self.cdir)
        yaml.safe_dump(common, open(os.path.join(self.cdir, 'common.yaml'), 'w'))
        for fam, ex in (('A_base', {'action_sft': 8}), ('AG', {'action_sft': 8, 'geometry_sft_v2': 4}), ('A_compute', {'action_sft': 'from_calibration'})):
            yaml.safe_dump({'base': 'common.yaml', 'run_family': fam, 'exposures': ex}, open(os.path.join(self.cdir, f'{fam}.yaml'), 'w'))

        class LoraLinear(torch.nn.Module):
            def __init__(self, d, r=2):
                super().__init__()
                self.lora_A = torch.nn.Parameter(torch.randn(r, d) * 0.3)
                self.lora_B = torch.nn.Parameter(torch.zeros(d, r))

            def forward(self, x):
                return x @ self.lora_A.T @ self.lora_B.T

        class FakeModel(torch.nn.Module):
            def __init__(self, V=40, D=8):
                super().__init__()
                self.emb = torch.nn.Embedding(V, D)
                self.head = torch.nn.Linear(D, V)
                for p in list(self.emb.parameters()) + list(self.head.parameters()):
                    p.requires_grad_(False)
                self.layers = torch.nn.ModuleDict({'self_attn': LoraLinear(D), 'shared_expert': LoraLinear(D)})
                self.drop = torch.nn.Dropout(0.05)

            def forward(self, input_ids, attention_mask=None, **kw):
                h = self.emb(input_ids)
                for m in self.layers.values():
                    h = h + m(self.drop(h))
                return types.SimpleNamespace(logits=self.head(h))

            def save_pretrained(self, d):
                os.makedirs(d, exist_ok=True)
                torch.save(self.state_dict(), os.path.join(d, 'adapter.pt'))

        def load_exports(bundle, split, kinds):
            return {k: [{'id': f'{split}-{k}-{i}', 'images': ['m.png', 'o.png'], 'root_id': f'{split}-{i}', 'task_id': f't{i}',
                         'pair_group_id': f'{split}-{i}', 'source_split': split} for i in range(10)] for k in kinds}

        def encode(processor, bundle, rec, max_length):
            g = torch.Generator().manual_seed(sum(map(ord, rec['id'])))
            n_p = 5 + int(torch.randint(0, 4, (1,), generator=g))
            n_t = 12 if 'geometry' in rec['id'] else 3
            ids = torch.randint(1, 40, (n_p + n_t,), generator=g)
            labels = ids.clone()
            labels[:n_p] = -100
            return {'input_ids': ids, 'labels': labels, 'pixel_values': None, 'image_grid_thw': None, 'extra': {},
                    'stats': {'id': rec['id'], 'tokens': n_p + n_t, 'prompt_tokens': n_p, 'visual_tokens': 0, 'target_tokens': n_t, 'masked_target_tokens': 0}}

        def load_model(cfg, adapter=None, for_training=False):
            return FakeModel(), {'class': 'FakeModel'}

        T.load_exports, T.check_isolation, T.encode = load_exports, (lambda *a: None), encode
        T.load_model = load_model
        T.load_processor = lambda cfg: types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, fam, out, *extra, micro=2):
        argv = sys.argv
        sys.argv = ['train_v2.py', '--config', os.path.join(self.cdir, f'{fam}.yaml'), '--seed', '0', '--bundle', self.dir,
                    '--out', out, '--micro-batch', str(micro), '--workers', '0', '--allow-optimizer-step', *extra]
        try:
            self.T.main()
        finally:
            sys.argv = argv
        return json.load(open(os.path.join(out, 'run.json')))

    def test_full_ag_run(self):
        out = os.path.join(self.dir, 'AG')
        run = self.run_main('AG', out)
        self.assertEqual(run['actual_exposures'], {'action_sft': 8, 'geometry_sft_v2': 4})
        self.assertEqual(run['planned_optimizer_steps'], 3)
        self.assertEqual(run['zero_grad_params_step1']['non_lora_A'], [])
        self.assertGreater(run['zero_grad_params_step1']['n'], 0)                  # lora_A, since lora_B starts at zero
        self.assertEqual(run['zero_grad_params_after_first_update'], {'step': 3, 'n': 0, 'examples': []})   # step 1 had lr 0
        ms = [json.loads(l) for l in open(os.path.join(out, 'metrics.jsonl'))]
        self.assertEqual([m['step'] for m in ms], [1, 2, 3])
        self.assertEqual(ms[0]['lr'], 0.0)
        self.assertEqual(sum(m['per_kind']['geometry_sft_v2']['n'] for m in ms), 4)
        cks = sorted(os.listdir(os.path.join(out, 'checkpoints')))
        self.assertEqual(cks, ['ckpt-1', 'ckpt-2', 'ckpt-3'])
        last = json.load(open(os.path.join(out, 'checkpoints', 'ckpt-3', 'exposures.json')))
        self.assertEqual(last['fraction'], 1.0)
        self.assertEqual(len(last['probe_ce']), 8)

    def test_action_sequence_shared(self):
        a = self.run_main('A_base', os.path.join(self.dir, 'A'))
        g = self.run_main('AG', os.path.join(self.dir, 'G'), '--max-steps', '1')
        self.assertEqual(a['action_sequence']['sha256'], g['action_sequence']['sha256'])

    def test_calibrated_budget(self):
        json.dump({'frozen': True, 'exposures': {'A_compute': {'action_sft': 12}}}, open(os.path.join(self.cdir, 'calibration.json'), 'w'))
        run = self.run_main('A_compute', os.path.join(self.dir, 'C'))
        self.assertEqual(run['actual_exposures'], {'action_sft': 12})

    def test_step_gradient_independent_of_micro_batch(self):
        torch = self.torch
        dumps = []
        for micro in (1, 2, 4):
            p = os.path.join(self.dir, f'g{micro}.pt')
            self.run_main('AG', os.path.join(self.dir, f'eq{micro}'), '--max-steps', '1', '--disable-dropout', '--dump-first-grad', p, micro=micro)
            dumps.append(torch.load(p))
        for d in dumps[1:]:
            self.assertEqual(d['members'], dumps[0]['members'])
            self.assertAlmostEqual(d['objective'], dumps[0]['objective'], places=5)
            for n, t in d['tensors'].items():
                self.assertTrue(torch.allclose(t, dumps[0]['tensors'][n], atol=1e-6))


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(HAVE, 'torch, PyYAML and Pillow needed')
class MicroBatchPinned(unittest.TestCase):
    def test_config_micro_batch_is_enforced(self):
        import yaml
        fake_hf_modules()
        import train_v2 as T
        with tempfile.TemporaryDirectory() as d:
            c = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'v2', 'common.yaml')))
            self.assertEqual(c['optim']['micro_batch'], 1)
            yaml.safe_dump(c, open(os.path.join(d, 'common.yaml'), 'w'))
            yaml.safe_dump({'base': 'common.yaml', 'run_family': 'A_base', 'exposures': {'action_sft': 8}}, open(os.path.join(d, 'A_base.yaml'), 'w'))
            T.load_exports = lambda b, s, ks: {k: [{'id': f'{k}{i}', 'images': ['a', 'b'], 'root_id': str(i), 'task_id': str(i),
                                                   'pair_group_id': str(i), 'source_split': s} for i in range(10)] for k in ks}
            T.check_isolation = lambda *x: None
            argv = sys.argv
            sys.argv = ['train_v2.py', '--config', os.path.join(d, 'A_base.yaml'), '--seed', '0', '--bundle', d, '--out', os.path.join(d, 'o'), '--micro-batch', '2']
            try:
                with self.assertRaises(SystemExit):
                    T.main()
            finally:
                sys.argv = argv
