# Daedalus

Daedalus is a research framework for studying whether controlled ownership-boundary ambiguity in bug reports can systematically redirect coding-agent reasoning. It generates variants of [SWE-bench Verified](https://www.swebench.com/) tasks by appending a competing (incorrect) causal hypothesis to the problem statement, then measures whether agents investigating the variant diverge in trajectory and diagnosis from the original.

## Research Motivation

Coding agents investigating a bug must not only find relevant code but form a correct causal hypothesis about which component is responsible. When a bug sits at the boundary between two components — a producer and a consumer, a transformer and a validator — both explanations can appear structurally sound from a surface read. Daedalus tests whether agents can be reliably misdirected by a plausible wrong hypothesis at exactly these boundaries.

[SWE-ABS (Yu et al., ICML 2026)](https://arxiv.org/abs/2603.00520) targets the **patch-evaluation** phase: it augments test suites to catch incorrect patches that pass weak tests. Daedalus targets the earlier phase — before an agent writes any patch, it forms a causal hypothesis by investigating the codebase. The question is not "can tests catch a wrong patch?" but "does misleading context cause the agent to produce the wrong hypothesis — and therefore the wrong patch — in the first place?"

## Artifact Taxonomy

When the problem-statement variant alone is insufficient to mislead an agent, the fabricator embeds small code-level artifacts that make the distractor look responsible:

**Class A — Structural (preferred):** Changes that cannot break any test because they only affect comments, docstrings, or local names.
- Stale comment or docstring: text that once described accurate behavior but no longer matches what the code does.
- Misleading local variable or parameter name: implies the distractor manages state it doesn't own.
- Narrowed test assertion: checks only the happy path, making it appear the distractor has been tested for the missing behavior.

**Class B — Behavioral (extra manual review required):** Changes that add new observable output.
- Log statement: a `logger.debug/info` call implying the distractor processes relevant state. Requires the module to already import and define `logger`.
- Subtly-wrong docstring: describes incorrect intended behavior, suggesting the distractor enforces an invariant it does not.

**Class C — Prohibited (auto-rejected):** Any change that touches runtime logic.
- New control-flow keyword (`if`, `elif`, `return`, `raise`, `while`, `for`, `try`, `except`, `yield`)
- Added logical operators (`and`/`or`) to an existing condition
- Function or class definition renamed (`def foo` → `def bar`)
- Functional code replaced by pure comments or whitespace (`def clone(self):` → `# clones the query`)
- Any increase in AST control-flow node count

Class C violations are caught by programmatic gates before any artifact reaches human review.

## Safety Check Methodology

Every proposed artifact undergoes a two-stage check in a SWE-bench Docker container before reaching manual review:

**Stage 1 — No regression:** Apply the fabricated diff. Run up to 200 PASS_TO_PASS tests. If any previously-passing test now fails → auto-reject. No override.

**Stage 2 — Gold patch integrity:** Apply the gold fix patch on top of the fabrication. Run the FAIL_TO_PASS tests. If the issue no longer resolves → auto-reject. No override.

Only artifacts that clear both stages are presented for manual review. The output is a unified diff, fabricator reasoning, and safety result written to `examples/output/fabrications.jsonl`.

## Current Status

**Proof of concept.** The fabrication pipeline has been validated on 4 tasks (`django__django-13212`, `django__django-13810`, `django__django-15127`, `django__django-12965`).

From the most recent fabrication run (12 proposed):
- 3 auto-rejected by gates (1 `class_c`, 1 `text_not_found`, 1 `too_short`)
- 2 retroactively rejected post-review (logger not in scope; commented-out log ≠ Class B)
- **7 clean accepted artifacts** — all pending Docker safety check and manual review

No agent-level evaluation against fabricated repositories has been run yet. The `evaluate-swe` command is implemented and tested on small batches, but end-to-end results for the fabrication condition are not yet available.

The problem-statement variant pipeline has generated ~116 `root_cause_attribution` variants from `django/django`, of which 12 passed cross-model re-validation (DeepSeek V4 Pro scoring Llama-generated variants). The full dataset needs regeneration with both fixes active (framing diversity + cross-model validation).

## Installation

```bash
git clone https://github.com/Allicai/daedalus
cd daedalus
pip install -e .
```

Set `TOGETHER_KEY` (or `OPENROUTER_API_KEY`) in a `.env` file at the project root. `LLM_PROVIDER=together` selects Together AI and is the default when `TOGETHER_KEY` is present.

## Usage

### Problem-statement variants

```bash
# Generate variants
daedalus generate --dataset hf --output tasks.jsonl --limit 50 --repo django/django

# Resume an interrupted run (skips already-exported tasks)
daedalus generate --dataset hf --output tasks.jsonl --limit 50 --repo django/django

# Read-only agent evaluation (file-path heuristics, no Docker required)
# Any Together AI or OpenRouter model string works via --model
daedalus evaluate --tasks tasks.jsonl --output eval.jsonl --repo-dir /path/to/repo --model deepseek-ai/DeepSeek-V4-Pro

# Full SWE-agent evaluation (requires Docker Desktop + mini-swe-agent)
daedalus evaluate-swe --tasks tasks.jsonl --output swe_eval/ --model deepseek-ai/DeepSeek-V4-Pro
```

### Fabrication (Phase 2)

```bash
# Generate fabrication candidates for one or more validated tasks
python examples/fabricate_review.py generate django__django-13212
python examples/fabricate_review.py generate django__django-13212 django__django-13810 --skip-safety

# List all candidates with status
python examples/fabricate_review.py list

# Review decisions
python examples/fabricate_review.py approve fab_django__django-13212_abc12345
python examples/fabricate_review.py reject  fab_django__django-13212_abc12345 --notes "too obvious"
```

### SWE-agent evaluation prerequisites

- **Docker Desktop** must be running. Evaluation degrades gracefully to `resolved=None` when Docker is unavailable.
- mini-SWE-agent v2.3.0: `pip install mini-swe-agent`
- SWE-bench Docker images are pulled automatically on first use (~1–2 GB each).

```bash
# Export task instances for inspection
daedalus export-instances --tasks tasks_v2.jsonl --output instances/ --limit 20

# Run mini-SWE-agent on both conditions, evaluate patches, write results
daedalus evaluate-swe \
    --tasks tasks_v2.jsonl \
    --output swe_eval/ \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --workers 1 \
    --limit 20
```

Outputs written to `swe_eval/`:
- `original/preds.json`, `variant/preds.json` — raw SWE-agent patches (resume-safe)
- `swe_runs.jsonl` — one `EvaluationRun` per condition per task
- `swe_pairs.jsonl` — one `EvaluationPair` per task with bucket classification

## Limitations

- **Django only:** the current dataset covers `django/django`; multi-repo generation is supported by the pipeline but requires pre-cloning additional repositories.
- **Single variant type:** only `root_cause_attribution` is implemented.
- **Template uniformity:** early dataset runs produced variants with low framing diversity. Fixed in the current codebase via randomly selected framing starters; the 116-task dataset predates this fix.
- **Self-validation bias:** the 116-task dataset was validated by the same model used for generation (113/116 scored exactly 4/4/4/4). Fixed: the validator now uses DeepSeek V4 Pro. Retroactive re-validation yielded 12/116 (10.3%).
- **No fabrication evaluation results yet:** the `evaluate-swe` pipeline is implemented but no aggregate resolved-rate figures under the fabrication condition are available.

## Design Decisions

**Safety gates have no override path.** Any regression in passing tests or failure to apply the gold patch on top of the fabrication results in automatic rejection. An artifact that breaks tests has already changed observable behavior, which disqualifies it regardless of how it was classified statically.

**Class C enforcement is programmatic, not prompt-only.** The system prompt prohibits control-flow additions, renames, and deletion-as-comment; the programmatic gates enforce it regardless. Models still violate these constraints ~25% of the time even when explicitly instructed not to.

**The gold patch is not forwarded to the LLM.** It is used only to locate which subsystem the fix touches. Distractors must be derived from the issue description and call-graph context, not reverse-engineered from the fix.

**Distractor selection is grounded in import connectivity.** The repo analyzer scores candidate files by import relationships and keyword relevance to the problem statement, excluding the patched file. Directory-proximity selection produces distractors with no call or state relationship to the true bug site, which can be dismissed with a single grep.
