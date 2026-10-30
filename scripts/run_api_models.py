"""Zero-shot evaluation of hosted frontier models on the benchmark prompts (no training, no GPU).

  pip install openai anthropic google-genai
  OPENAI_API_KEY=.. GEMINI_API_KEY=.. ANTHROPIC_API_KEY=.. python scripts/run_api_models.py \
      --bundle $BUNDLE --out $LOCUS_ROOT/work/api [--models gpt-6-astra,gemini-4-argon,claude-fable-5-1] \
      [--prompts test_V0] [--concurrency 8] [--plan]

Prompts are the bundle's eval_prompts/<name>.jsonl, byte for byte the text, system prompt and image order that the
four API models of the main table received (checked on test_V0 and test_diagnostic_S). One independent request per
prompt; provider-default sampling; no prefill, no tools, no few-shot. Settings per model (MODELS below):
  - output budget: each API's large output limit, so that answers are not cut by the budget;
  - reasoning effort: the provider default unless --effort is given (recorded in every answer);
  - Anthropic: no refusal fallback. A fallback would re-run a declined request on another model, so the answer would
    not be the named model's; a refusal is kept as the answer (stop_reason "refusal") and scored as a failure.
Output: <out>/<model>/<prompt>.jsonl, one line per answer in the format of scripts/infer.py (task_id, sample, condition,
split, raw, infra_error, finish, ...) plus usage and seconds; reruns skip finished prompts. Transient errors are retried
(SDK retries, then up to 4 rounds here, rate limits with 2-16 minute waits). Only model answers are written: a billing
problem or a rate limit that outlasts the retries stops the run (exit 3), other persistent errors go to errors.jsonl and
are retried by the next run (exit 2 while any remain).
--plan prints the request count and a cost estimate (assumed tokens per request) and sends nothing.
Requests go straight to each provider; set HTTPS_PROXY if the machine needs a proxy. Keys come only from the environment.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

# model id -> provider, max output tokens, (input $, output $) per million tokens as listed by the provider on 2026-10-30
MODELS = {
    'gpt-6-astra': ('openai', 128000, (10.0, 50.0)),
    'gpt-6.1-sol': ('openai', 128000, (2.0, 10.0)),
    'gemini-4-argon': ('google', 65536, (2.0, 10.0)),      # price through 2026-11-30 per third-party listings; check the rate card
    'gemini-3.8-flash': ('google', 65536, (0.75, 3.75)),
    'claude-fable-5-1': ('anthropic', 128000, (10.0, 50.0)),
    'claude-opus-5-5': ('anthropic', 128000, (4.0, 20.0)),
}
DEFAULT_MODELS = 'gpt-6-astra,gemini-4-argon,claude-fable-5-1'
EST_IN, EST_OUT = 2000, 12000      # tokens per request for --plan (the four reasoning models of the main table: 1.1-1.6k in, 6.5-17k out)
BILLING_WORDS = ('insufficient_quota', 'credit balance', 'billing', 'payment', 'exceeded your current quota')
RATE_WORDS = ('rate limit', 'rate_limit', 'ratelimit', '429', 'resource_exhausted', 'too many requests', 'overloaded', '529')
STOP = threading.Event()


def b64(path):
    return base64.standard_b64encode(open(path, 'rb').read()).decode()


def call_openai(model, rec, images, max_out, effort, timeout):
    from openai import OpenAI
    client = OpenAI(timeout=timeout)
    content = [{'type': 'input_image', 'image_url': f'data:image/png;base64,{b64(p)}'} for p in images]
    content.append({'type': 'input_text', 'text': rec['prompt']})
    kw = {'reasoning': {'effort': effort}} if effort else {}
    with client.responses.stream(model=model, instructions=rec['system'], input=[{'role': 'user', 'content': content}],
                                 max_output_tokens=max_out, **kw) as stream:
        r = stream.get_final_response()
    finish = 'stop' if r.status == 'completed' else ('length' if getattr(r.incomplete_details, 'reason', '') == 'max_output_tokens'
                                                      else str(r.status))
    u = r.usage
    return {'raw': r.output_text, 'finish': finish, 'usage': {'input_tokens': u.input_tokens, 'output_tokens': u.output_tokens,
            'reasoning_tokens': getattr(getattr(u, 'output_tokens_details', None), 'reasoning_tokens', None)}}


def call_google(model, rec, images, max_out, effort, timeout):
    from google import genai
    from google.genai import types
    client = genai.Client(http_options=types.HttpOptions(timeout=timeout * 1000))
    parts = [types.Part.from_bytes(data=open(p, 'rb').read(), mime_type='image/png') for p in images] + [rec['prompt']]
    cfg = {'system_instruction': rec['system'], 'max_output_tokens': max_out}
    if effort:
        cfg['thinking_config'] = types.ThinkingConfig(thinking_level=effort)
    text, last = [], None
    for chunk in client.models.generate_content_stream(model=model, contents=parts, config=types.GenerateContentConfig(**cfg)):
        last = chunk
        for c in chunk.candidates or []:
            for part in (c.content.parts if c.content else None) or []:
                if part.text and not getattr(part, 'thought', False):
                    text.append(part.text)
    reason = str(last.candidates[0].finish_reason) if last and last.candidates else 'none'
    um = last.usage_metadata if last else None
    return {'raw': ''.join(text), 'finish': 'stop' if reason.endswith('STOP') else ('length' if 'MAX_TOKENS' in reason else reason),
            'usage': {'input_tokens': getattr(um, 'prompt_token_count', None), 'output_tokens': getattr(um, 'candidates_token_count', None),
                      'reasoning_tokens': getattr(um, 'thoughts_token_count', None)}}


def call_anthropic(model, rec, images, max_out, effort, timeout):
    import anthropic
    client = anthropic.Anthropic(timeout=timeout)
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64(p)}} for p in images]
    content.append({'type': 'text', 'text': rec['prompt']})
    kw = {'output_config': {'effort': effort}} if effort else {}
    # thinking is always on for these models; streaming because a long answer can exceed a non-streaming request's limits
    with client.messages.stream(model=model, max_tokens=max_out, system=rec['system'],
                                messages=[{'role': 'user', 'content': content}], **kw) as stream:
        m = stream.get_final_message()
    text = ''.join(b.text for b in m.content if b.type == 'text')
    finish = {'end_turn': 'stop', 'max_tokens': 'length'}.get(m.stop_reason, m.stop_reason)
    out = {'raw': text, 'finish': finish, 'usage': {'input_tokens': m.usage.input_tokens, 'output_tokens': m.usage.output_tokens}}
    if m.stop_reason == 'refusal' and getattr(m, 'stop_details', None):
        out['refusal_category'] = getattr(m.stop_details, 'category', None)
    return out


CALL = {'openai': call_openai, 'google': call_google, 'anthropic': call_anthropic}


def answer(model, rec, bundle, effort, timeout):
    """('ok', result) | ('stop', error): billing problem or a rate limit that outlasted the retries | ('error', error)."""
    provider, max_out, _ = MODELS[model]
    images = [os.path.join(bundle, p) for p in rec['images']]
    err, rate_tries = None, 0
    for k in range(4):
        t0 = time.time()
        try:
            out = CALL[provider](model, rec, images, max_out, effort, timeout)
            return 'ok', {**out, 'infra_error': None, 'seconds': round(time.time() - t0, 1), 'attempts': k + 1 + rate_tries}
        except Exception as e:  # SDK error classes differ per provider; classify by status text
            err = f'{type(e).__name__}: {str(e)[:500]}'
            low = err.lower()
            if any(w in low for w in BILLING_WORDS):
                return 'stop', err
            if any(w in low for w in RATE_WORDS):
                rate_tries += 1
                if rate_tries > 4:
                    return 'stop', err
                time.sleep(60 * 2 ** rate_tries)     # 2, 4, 8, 16 minutes
                continue
            time.sleep(30 * (k + 1))
    return 'error', err


def plan(models, prompt_files):
    n = sum(len(v) for v in prompt_files.values())
    print(f'{n} prompts per model ({", ".join(f"{k}: {len(v)}" for k, v in prompt_files.items())}); '
          f'cost estimate at {EST_IN} input + {EST_OUT} output tokens per request:')
    total = 0
    for m in models:
        pin, pout = MODELS[m][2]
        c, lo, hi = (n * (EST_IN * pin + f * EST_OUT * pout) / 1e6 for f in (1, 0.5, 2))
        total += c
        print(f'  {m:18s} ~${c:,.0f}   (half / double the output tokens: ${lo:,.0f} - ${hi:,.0f})')
    print(f'  total ~${total:,.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=os.environ.get('BUNDLE'), required=os.environ.get('BUNDLE') is None)
    ap.add_argument('--out', default=os.path.join(os.environ.get('LOCUS_ROOT', '.'), 'work', 'api'))
    ap.add_argument('--models', default=DEFAULT_MODELS)
    ap.add_argument('--prompts', default='test_V0', help='comma-separated eval_prompts names, e.g. test_V0,test_diagnostic_S')
    ap.add_argument('--effort', default=None, help='reasoning effort for every model (default: each provider default)')
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=3600, help='seconds per request')
    ap.add_argument('--plan', action='store_true')
    a = ap.parse_args()
    models = a.models.split(',')
    for m in models:
        if m not in MODELS:
            raise SystemExit(f'unknown model {m}; known: {", ".join(MODELS)}')
    files = {p: [json.loads(l) for l in open(os.path.join(a.bundle, 'eval_prompts', f'{p}.jsonl'), encoding='utf-8') if l.strip()]
             for p in a.prompts.split(',')}
    if a.plan:
        plan(models, files)
        return
    env = {'openai': 'OPENAI_API_KEY', 'google': 'GEMINI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}
    missing = sorted({env[MODELS[m][0]] for m in models if not os.environ.get(env[MODELS[m][0]])})
    if missing:
        raise SystemExit(f'set {", ".join(missing)} in the environment')
    lock = threading.Lock()
    unresolved = False
    for m in models:
        for pname, recs in files.items():
            path = os.path.join(a.out, m, f'{pname}.jsonl')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            done = {(json.loads(l)['task_id'], json.loads(l).get('sample', 0)) for l in open(path)} if os.path.exists(path) else set()
            todo = [r for r in recs if (r['task_id'], r.get('sample', 0)) not in done]
            print(f'{m} {pname}: {len(todo)} to run, {len(done)} done', flush=True)

            def one(r, m=m, pname=pname, path=path):
                if STOP.is_set():
                    return
                kind, res = answer(m, r, a.bundle, a.effort, a.timeout)
                with lock:
                    if kind == 'ok':
                        line = {'task_id': r['task_id'], 'sample': r.get('sample', 0), 'condition': r['condition'], 'split': r['split'],
                                'model': m, 'provider': MODELS[m][0], 'effort': a.effort or 'provider default',
                                'max_output_tokens': MODELS[m][1], 'prompt_file': pname, **res}
                        with open(path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        print(m, r['task_id'], res['finish'], res['seconds'], flush=True)
                    else:
                        with open(os.path.join(os.path.dirname(path), 'errors.jsonl'), 'a', encoding='utf-8') as f:
                            f.write(json.dumps({'time': time.strftime('%F %T'), 'prompt_file': pname, 'task_id': r['task_id'], 'kind': kind,
                                                'error': res}, ensure_ascii=False) + '\n')
                        print(m, r['task_id'], kind, res[:160], flush=True)
                        if kind == 'stop':
                            STOP.set()

            with cf.ThreadPoolExecutor(a.concurrency) as ex:
                list(ex.map(one, todo))
            if STOP.is_set():
                print('stopped: billing problem or a persistent rate limit (see errors.jsonl); fix it and rerun the same command')
                sys.exit(3)
            left = len(recs) - sum(1 for _ in open(path))
            if left:
                unresolved = True
                print(f'{m} {pname}: {left} prompts failed after retries (errors.jsonl); rerun the same command to retry them')
    sys.exit(2 if unresolved else 0)


if __name__ == '__main__':
    main()
