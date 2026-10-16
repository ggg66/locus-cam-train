"""Submission parsing and validation scoring, identical to the benchmark scorer (score-v1.0.0, rules P1-P4).
Only validation tables travel to the training server; test answers are scored on the benchmark machine."""
import json
import re

RIGHT, UP, BACK = [-2, -1, 0, 1, 2], [-1, 0, 1, 2], [-3, -2, -1, 0, 1, 2]


def extract_json(text):
    for b in reversed(re.findall(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', text or '')):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    return None


def parse(raw):
    obj = extract_json(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict):
        return None, 'format_error'
    move = obj.get('move_m', obj)
    if not isinstance(move, dict) or not all(k in move for k in ('right', 'up', 'back')):
        return None, 'format_error'
    act = (move['right'], move['up'], move['back'])
    if not all(isinstance(a, (int, float)) and not isinstance(a, bool) for a in act):
        return None, 'format_error'
    if act[0] not in RIGHT or act[1] not in UP or act[2] not in BACK:
        return None, 'out_of_domain'
    return tuple(int(a) for a in act), None


def outcome(raw, table):
    """table: the frozen 120-action table of one validation root ({"+r,+u,+b": {"status": ...}})."""
    act, err = parse(raw)
    if err:
        return err
    return table['%+d,%+d,%+d' % act]['status']
