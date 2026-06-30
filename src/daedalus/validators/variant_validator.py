from __future__ import annotations

import json
import re

from daedalus.models.task_schema import DaedalusTask
from daedalus.openrouter import MAX_TOKENS_VALIDATE, VALIDATOR_MODEL, get_client_for_model

_MIN_ADDED_CHARS = 30
_MIN_DISTRACTOR_LEN = 20
_ORIGINAL_ANCHOR_LEN = 80
_MIN_SCORE = 4  # ALL four dimensions must reach this to pass

_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_validation",
        "description": "Score a root_cause_attribution variant on four dimensions.",
        "parameters": {
            "type": "object",
            "properties": {
                "plausibility": {
                    "type": "integer",
                    "description": (
                        "1–5. After reading the modified statement, would a skilled developer "
                        "genuinely add this distractor to their investigation list? "
                        "1 = dismissed in under a minute (no specific code location, "
                        "version-history explanation, obviously wrong subsystem). "
                        "3 = might investigate briefly. "
                        "5 = convincing enough that most developers would open the file."
                    ),
                },
                "locality": {
                    "type": "integer",
                    "description": (
                        "1–5. Is the distractor in the same execution path or code layer as the true bug? "
                        "1 = different subsystem entirely (e.g. auth vs. serialization). "
                        "3 = same module, unrelated concern. "
                        "5 = adjacent function or same call chain as the true bug."
                    ),
                },
                "competitiveness": {
                    "type": "integer",
                    "description": (
                        "1–5. How much active investigation does ruling out the distractor require? "
                        "1 = re-reading the problem statement is sufficient to dismiss it. "
                        "2 = a single grep or a quick glance at the function signature. "
                        "3 = requires reading one function body in full. "
                        "4 = requires reading two or more function bodies and understanding how "
                        "they interact — i.e. you must trace data flow between the distractor "
                        "and the true bug site to rule it out. "
                        "5 = requires tracing a multi-hop call chain or comparing several "
                        "implementations across files to prove no interaction occurs."
                    ),
                },
                "semantic_fit": {
                    "type": "integer",
                    "description": (
                        "1–5. Is there a believable causal mechanism connecting the distractor "
                        "location to the observed bug symptom? "
                        "1 = no plausible mechanism (e.g. cookie parsing implicated for a "
                        "pickling bug — different concern, no shared state). "
                        "3 = vague shared concern but no direct state or call dependency. "
                        "5 = distractor and true bug share an execution path, shared state, "
                        "or protocol contract — the boundary must be confirmed in code."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "2-3 sentences. State the single biggest weakness AND the single biggest "
                        "strength. Name specific code locations. Describe exactly how quickly "
                        "a developer could dismiss the distractor and why."
                    ),
                },
            },
            "required": [
                "plausibility", "locality", "competitiveness", "semantic_fit", "reason"
            ],
        },
    },
}

_SYSTEM = (
    "You are an adversarial quality reviewer for synthetic coding-agent evaluation tasks. "
    "Your goal is to catch low-quality distractors before they contaminate an evaluation set. "
    "Grade strictly — a distractor that passes should genuinely cost a skilled developer "
    "30+ minutes of investigation.\n\n"
    "Score on four dimensions (1–5 each). ALL must reach 4 to pass.\n\n"
    "PLAUSIBILITY — Would a skilled developer genuinely investigate this?\n"
    "  Penalise: vague hints with no specific code location; version-history or config "
    "  explanations; anything a developer dismisses before opening any files.\n\n"
    "LOCALITY — Is the distractor in the same execution path or code layer as the true bug?\n"
    "  Penalise heavily for cross-subsystem distractors. High score requires the distractor "
    "  to sit in the same call chain or share code with the actual bug site.\n\n"
    "COMPETITIVENESS — How much active investigation is needed to rule it out?\n"
    "  Be adversarial here: identify the fastest dismissal path and score accordingly.\n"
    "  ≤2 if a single grep or one-function glance is sufficient.\n"
    "  3 if you must read one full function body.\n"
    "  4 if you must read TWO function bodies and trace the data flow between them — "
    "  neither alone settles the question.\n"
    "  5 if you must trace a multi-hop chain or compare several implementations.\n"
    "  Do not score 4+ unless the developer truly cannot resolve the question by reading "
    "  either the distractor or the bug site in isolation.\n\n"
    "SEMANTIC_FIT — Is there a believable causal story?\n"
    "  This is the key dimension. Ask: could the distractor location plausibly produce "
    "  the exact symptom described? No shared state, no shared code path, no common "
    "  protocol → score 1-2 regardless of how nearby the files are.\n\n"
    "After scoring, identify the single fastest way a developer could rule out this "
    "distractor, and make sure competitiveness reflects that honestly."
)


def validate(variant: DaedalusTask, llm: bool = True) -> DaedalusTask:
    passed, reason = _heuristic(variant)
    if not passed:
        return variant.model_copy(update={
            "quality_status": "rejected",
            "validation_notes": f"[heuristic] {reason}",
        })

    if not llm:
        return variant.model_copy(update={
            "quality_status": "validated",
            "validation_notes": "[heuristic] all checks passed",
        })

    scores, reason = _llm(variant)
    dims = ("plausibility", "locality", "competitiveness", "semantic_fit")
    passed = all(scores[d] >= _MIN_SCORE for d in dims)
    score_str = " ".join(f"{d}={scores[d]}" for d in dims)
    return variant.model_copy(update={
        "quality_status": "validated" if passed else "rejected",
        "validation_notes": f"[llm] {score_str} — {reason}",
    })


def _heuristic(variant: DaedalusTask) -> tuple[bool, str]:
    orig = variant.original_problem_statement
    mod = variant.modified_problem_statement
    meta = variant.transformation_metadata

    if len(mod) - len(orig) < _MIN_ADDED_CHARS:
        return False, f"only {len(mod) - len(orig)} chars added (min {_MIN_ADDED_CHARS})"
    if len(meta.intended_distractor) < _MIN_DISTRACTOR_LEN:
        return False, f"intended_distractor too short ({len(meta.intended_distractor)} chars)"
    if not meta.changes_made:
        return False, "changes_made list is empty"
    # SWE-bench originals contain literal tabs that LLMs routinely strip.
    _norm = lambda s: re.sub(r'\s+', ' ', s)
    orig_anchor = _norm(orig[:_ORIGINAL_ANCHOR_LEN * 2])[:_ORIGINAL_ANCHOR_LEN]
    if orig_anchor not in _norm(mod[:_ORIGINAL_ANCHOR_LEN * 4]):
        return False, "opening of original problem statement not preserved in modified version"

    return True, ""


def _llm(variant: DaedalusTask) -> tuple[dict, str]:
    client, routing = get_client_for_model(VALIDATOR_MODEL)
    meta = variant.transformation_metadata

    response = client.chat.completions.create(
        model=VALIDATOR_MODEL,
        max_tokens=MAX_TOKENS_VALIDATE,
        extra_body=routing,
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_validation"}},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original problem statement:\n{variant.original_problem_statement}\n\n"
                    f"Modified problem statement:\n{variant.modified_problem_statement}\n\n"
                    f"Intended distractor: {meta.intended_distractor}\n\n"
                    "Changes made:\n" + "\n".join(f"- {c}" for c in meta.changes_made)
                ),
            },
        ],
    )

    result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    scores = {d: result[d] for d in ("plausibility", "locality", "competitiveness", "semantic_fit")}
    return scores, result["reason"]
