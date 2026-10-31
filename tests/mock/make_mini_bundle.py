"""A few records from a real data bundle, for smoke tests.

  python tests/mock/make_mini_bundle.py <real bundle> <out> [--train 12] [--val 6] [--test 4]
"""
import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('out')
ap.add_argument('--train', type=int, default=12)
ap.add_argument('--val', type=int, default=6)
ap.add_argument('--test', type=int, default=4)
a = ap.parse_args()
shutil.rmtree(a.out, ignore_errors=True)
images = set()


def take(rel, n, pick=None):
    lines = [json.loads(l) for l in open(os.path.join(a.src, rel), encoding='utf-8')]
    if pick:
        lines = sorted(lines, key=pick)
    lines = lines[:n]
    os.makedirs(os.path.dirname(os.path.join(a.out, rel)), exist_ok=True)
    with open(os.path.join(a.out, rel), 'w', encoding='utf-8') as f:
        for r in lines:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            images.update(r['images'])
    return lines


for split, n in (('train', a.train), ('validation', a.val)):
    take(f'training_exports/{split}/action_sft.jsonl', n)
    take(f'training_exports/{split}/geometry_sft.jsonl', n, pick=lambda r: -len(r['loss_mask_paths']))  # prefer records with masks
    if os.path.exists(os.path.join(a.src, f'training_exports/{split}/geometry_sft_v2.jsonl')):   # v2 add-on
        take(f'training_exports/{split}/geometry_sft_v2.jsonl', n, pick=lambda r: -len(r['hidden_paths']))
        shutil.copyfile(os.path.join(a.src, f'training_exports/{split}/meta_v2.json'), os.path.join(a.out, f'training_exports/{split}/meta_v2.json'))
    for k in ('consequence_v4', 'geomact_v4', 'nextview_v4'):     # v4 add-on (geomact_v4 is aligned with action_sft: same first n roots)
        if os.path.exists(os.path.join(a.src, f'training_exports/{split}/{k}.jsonl')):
            take(f'training_exports/{split}/{k}.jsonl', n)
val = take('eval_prompts/validation_V0.jsonl', a.val)
for name in ('test_V0', 'test_variants_V0', 'test_diagnostic_S', 'test_diagnostic_no_image', 'test_diagnostic_geom_predict',
             'test_diagnostic_consequence', 'test_target_swap_V0', 'dev_V0', 'dev_V0_placeholder'):
    if os.path.exists(os.path.join(a.src, f'eval_prompts/{name}.jsonl')):
        take(f'eval_prompts/{name}.jsonl', a.test)
os.makedirs(os.path.join(a.out, 'validation_tables'))
index = json.load(open(os.path.join(a.src, 'validation_tables', 'index.json')))
keep = {r['task_id']: index[r['task_id']] for r in val}
for tid, root in keep.items():
    shutil.copyfile(os.path.join(a.src, 'validation_tables', root + '.json'), os.path.join(a.out, 'validation_tables', root + '.json'))
json.dump(keep, open(os.path.join(a.out, 'validation_tables', 'index.json'), 'w'))
shutil.copyfile(os.path.join(a.src, 'splits_index.json'), os.path.join(a.out, 'splits_index.json'))
for p in images:
    os.makedirs(os.path.dirname(os.path.join(a.out, p)), exist_ok=True)
    shutil.copyfile(os.path.join(a.src, p), os.path.join(a.out, p))
print('mini bundle', a.out, len(images), 'images')
