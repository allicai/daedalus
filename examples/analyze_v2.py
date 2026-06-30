#!/usr/bin/env python3
"""Quick analysis of tasks_v2.jsonl — run after batch completion."""
import json, re
from pathlib import Path
from collections import Counter

OUT = Path(__file__).parent / 'output' / 'tasks_v2.jsonl'
CACHE = Path.home() / '.daedalus' / 'github_cache'
FORBIDDEN = ['Competing Hypothesis', 'Alternative Explanation', '## Competing', 'Hypothesis:']

tasks = []
with OUT.open(encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            tasks.append(json.loads(line))

n = len(tasks)
validated = [t for t in tasks if t['quality_status'] == 'validated']
rejected  = [t for t in tasks if t['quality_status'] == 'rejected']

print(f'Total: {n}  Validated: {len(validated)}  Rejected: {len(rejected)}  Yield: {len(validated)/n*100:.1f}%')

# Ownership type distribution
roles = Counter()
for t in validated:
    d = t['transformation_metadata']['intended_distractor']
    for role in ('PRODUCER','CONSUMER','TRANSFORMER','VALIDATOR','COORDINATOR'):
        if role in d:
            roles[role] += 1
            break

print('\nOwnership type distribution (validated only):')
for role, cnt in roles.most_common():
    bar = '█' * cnt
    print(f'  {role:<12} {cnt:>3}  {bar}')

# Forbidden pattern check
violations = []
for t in tasks:
    orig = t.get('original_problem_statement', '')
    mod  = t.get('modified_problem_statement', '')
    tail = mod[len(orig):] if mod.startswith(orig) else mod[-500:]
    for pat in FORBIDDEN:
        if pat.lower() in tail.lower():
            violations.append((t['source_instance_id'], pat))

print(f'\nForbidden pattern violations: {len(violations)}')
for iid, pat in violations[:10]:
    print(f'  {iid}: {repr(pat)}')

# Score distribution
from collections import defaultdict
scores = defaultdict(list)
score_re = re.compile(r'plausibility=(\d+)\s+locality=(\d+)\s+competitiveness=(\d+)\s+semantic_fit=(\d+)')
for t in tasks:
    m = score_re.search(t.get('validation_notes', ''))
    if m:
        scores['plausibility'].append(int(m.group(1)))
        scores['locality'].append(int(m.group(2)))
        scores['competitiveness'].append(int(m.group(3)))
        scores['semantic_fit'].append(int(m.group(4)))

print('\nScore distributions (all LLM-scored tasks):')
for dim in ('plausibility', 'locality', 'competitiveness', 'semantic_fit'):
    vals = scores[dim]
    if vals:
        from statistics import mean, median
        dist = Counter(vals)
        print(f'  {dim:<16} mean={mean(vals):.2f} median={median(vals):.1f}  dist={dict(sorted(dist.items()))}')

# GitHub API cache stats
cache_n = len(list(CACHE.glob('*.json'))) if CACHE.exists() else 0
cache_mb = sum(f.stat().st_size for f in CACHE.glob('*.json')) / 1_048_576 if CACHE.exists() else 0
print(f'\nGitHub API cache: {cache_n} entries, {cache_mb:.1f} MB')

# Rejection breakdown
heuristic_rej = [t for t in rejected if '[heuristic]' in t.get('validation_notes', '')]
llm_rej = [t for t in rejected if '[llm]' in t.get('validation_notes', '')]
print(f'\nRejection breakdown:')
print(f'  Heuristic: {len(heuristic_rej)}')
print(f'  LLM:       {len(llm_rej)}')

# Sample 3 validated tasks
print('\n--- 3 SAMPLE VALIDATED TASKS ---')
import random
random.seed(42)
samples = random.sample(validated, min(3, len(validated)))
for t in samples:
    orig  = t['original_problem_statement']
    mod   = t['modified_problem_statement']
    tail  = mod[len(orig):].lstrip('\n') if mod.startswith(orig) else '(prefix mismatch)'
    notes = t.get('validation_notes', '')
    print(f"\n{t['source_instance_id']}")
    print(f"  Distractor: {t['transformation_metadata']['intended_distractor'][:100]}...")
    print(f"  Scores: {notes[6:80] if notes.startswith('[llm]') else notes[:80]}")
    print(f"  Appended: {tail[:200]}")
