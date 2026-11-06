"""Inference tasks of the untrained bases under other prompts (v9, docs/v9/README.md). No training, no adapter.

  python scripts/make_v9_tasks.py --base qwen  > work/v9/tasks-v9-base-qwen.txt    # pool with LCAM_ROUND=v9
  python scripts/make_v9_tasks.py --base gemma > work/v9/tasks-v9-base-gemma.txt   # pool with LCAM_ROUND=v3_gemma

For each base, on the 600 test tasks:
  test_V0_example_random  the output example of the prompt is one nonzero move drawn per task (addon v9), thinking off
  test_V0_schema          the example is replaced by a description of the output keys (addon v9), thinking off
  test_V0 + thinking      the unchanged prompt with the base model's thinking mode on, 8192 new tokens, in the 60 parts of
                          10 tasks of eval_prompts/v9_think/ (addon v9) so that the long generations spread over the GPUs
Thinking off uses the B0 settings of every round (512 new tokens). Batch 1 throughout.
"""
import argparse

OUTS = 'work/outputs/v9-base'
N_PARTS = 60
TASKS = [  # (output name, prompt file, thinking, max new tokens)
    ('test_V0_example_random', 'test_V0_example_random', 0, 512),
    ('test_V0_schema', 'test_V0_schema', 0, 512),
] + [(f'test_V0_thinking/part-{k:02d}', f'v9_think/test_V0_part-{k:02d}', 1, 8192) for k in range(N_PARTS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, choices=['qwen', 'gemma'])
    a = ap.parse_args()
    for name, prompts, thinking, maxnew in TASKS:
        print('\t'.join([f'{OUTS}/{a.base}/{name}.jsonl', f'eval_prompts/{prompts}.jsonl', '-', str(thinking), '1', str(maxnew)]))


if __name__ == '__main__':
    main()
