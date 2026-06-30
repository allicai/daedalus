from __future__ import annotations

import json
import random
import re

from daedalus.loaders.swebench_loader import SWEBenchTask
from daedalus.models.task_schema import DaedalusTask, Provenance, TransformationMetadata
from daedalus.openrouter import MAX_TOKENS_GENERATE, MODEL, PROVIDER_ROUTING, get_client
from daedalus.repo_analyzer import analyze, format_for_prompt

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_variant",
            "description": "Output a root_cause_attribution variant of the given SWE-Bench task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modified_problem_statement": {
                        "type": "string",
                        "description": (
                            "Copy the ENTIRE original problem statement character-for-character "
                            "as the first part of this field — every line, every whitespace, "
                            "every metadata header, every code block, every URL, unchanged. "
                            "Do not clean, reformat, deduplicate, or summarise the original.\n\n"
                            "Then append 1–3 sentences written as NATURAL BUG REPORT CONTEXT — "
                            "an observation from triage, a note from a colleague, or a finding "
                            "that surfaces the distractor component as a plausible contributor. "
                            "Write in the voice of an engineer adding a follow-up note to the "
                            "original report. The reader must not recognise this as a structured "
                            "distractor — it must feel like organic context.\n\n"
                            "CRITICAL RULES:\n"
                            "  ✓  Name the specific symbol and file path\n"
                            "  ✓  Describe a plausible but brief causal observation\n"
                            "  ✓  Vary the framing — never use the same opening phrase twice\n"
                            "  ✗  NEVER use the word 'hypothesis' anywhere in the appended text\n"
                            "  ✗  NEVER label it 'Competing Hypothesis:', 'Alternative "
                            "Explanation:', 'Hypothesis:', or any variant\n"
                            "  ✗  NEVER use markdown headers (##) or bold labels (**...**)\n\n"
                            "EXAMPLE FRAMINGS (rotate among these — do not copy verbatim):\n"
                            "  'Initial investigation also flagged <Symbol> in <file>, which "
                            "handles <responsibility> — it may be worth checking whether "
                            "<specific behavior>.'\n"
                            "  'During triage, a colleague noted that <Symbol> in <file> could "
                            "be involved, since it is responsible for <responsibility>.'\n"
                            "  'An earlier review flagged <file> as potentially relevant — "
                            "<Symbol> owns <responsibility> and its behavior here may warrant "
                            "a closer look.'\n"
                            "  'One thread in the discussion points to <Symbol> in <file>: it "
                            "<responsibility>, and if <condition>, that would explain the "
                            "observed behavior.'\n"
                            "  'Worth noting: the issue was also observed downstream of "
                            "<Symbol> in <file>, which <responsibility>.'"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Three sentences. "
                            "(1) FAILING STATE: the specific value or invariant that is violated, "
                            "in domain terms (not code terms). "
                            "(2) OWNERSHIP CONFLICT: which stakeholder role (PRODUCER/CONSUMER/"
                            "TRANSFORMER/VALIDATOR/COORDINATOR) makes this distractor a plausible "
                            "alternative owner, and what responsibility they hold over the state. "
                            "(3) INVESTIGATION COST: what a developer must trace between the "
                            "distractor and the true bug site — state explicitly why reading "
                            "either one in isolation does not settle the question."
                        ),
                    },
                    "changes_made": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Discrete list of changes made to the original problem statement.",
                    },
                    "intended_distractor": {
                        "type": "string",
                        "description": (
                            "Format: '<Symbol> in <file> — <ROLE>: <ownership sentence>'\n"
                            "ROLE must be one of:\n"
                            "  PRODUCER     — distractor creates or computes the failing state\n"
                            "  CONSUMER     — distractor receives the state and should handle it\n"
                            "  TRANSFORMER  — distractor modifies the state in transit\n"
                            "  VALIDATOR    — distractor should enforce the invariant but doesn't\n"
                            "  COORDINATOR  — distractor orchestrates multiple owners of this state\n"
                            "The ownership sentence must state: (a) what state this stakeholder "
                            "owns or should own, and (b) why ruling out this owner requires "
                            "reading both the distractor AND the true bug site."
                        ),
                    },
                },
                "required": [
                    "modified_problem_statement",
                    "description",
                    "changes_made",
                    "intended_distractor",
                ],
            },
        },
    }
]

_GOLD_EXAMPLE = """\
━━━ GOLD STANDARD EXAMPLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Task: annotations are propagated into count() subqueries, causing incorrect SQL.

  FAILING STATE
    Unused annotations are included in the count subquery — the SQL that Django
    generates for .count() carries annotation columns it doesn't need.

  RESPONSIBILITY MAP
    PRODUCER:  count()          — creates the count subquery from the current queryset
    CONSUMER:  AggregateQuery   — receives the query state and builds the aggregate
    TRANSFORMER: Query.clone()  — copies query state, including annotations, into the subquery

  COMPETING HYPOTHESES
    H1 [PRODUCER]  count() should strip non-aggregated annotations before building the subquery.
    H2 [CONSUMER]  AggregateQuery.__init__() should strip annotations it receives on construction.
    H3 [TRANSFORMER] Query.clone() should drop annotations that are not needed in count context.

  SELECTED: H2
    Why it wins: ruling out H2 requires reading count() to see it passes annotations
    through without filtering, AND reading AggregateQuery.__init__() to see it does
    not filter them on receipt. Neither function alone settles whether the responsibility
    is "strip before handing off" or "strip on receipt" — the ownership boundary between
    the two functions must be traced.

  DISCARD H1: reading count() alone shows it passes the query to AggregateQuery without
    filtering. That makes H1 sound like the true bug site, not the distractor.
  DISCARD H3: reading Query.clone() alone confirms it is a general-purpose copy operation
    with no count-specific logic — dismissed in one read.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

_SYSTEM = f"""You are a research assistant creating synthetic task variants to evaluate coding agents.

GOAL: produce a root_cause_attribution variant that introduces a competing, INCORRECT causal
explanation — one that proposes an alternative OWNER of the failing state.

Do not optimize for: same file, same module, nearby symbols.
Optimize for: competing ownership of a specific failing state.

{_GOLD_EXAMPLE}

━━━ PHASE 1: IDENTIFY THE FAILING STATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

State the specific value, property, or invariant that is wrong.
Use DOMAIN TERMS — not code terms.

  ✓  "unused annotations are propagated into the count subquery"
  ✓  "backward migration leaves old index name in the schema state"
  ✓  "URL reverse() resolves to wrong namespace for related admin objects"
  ✓  "rotation matrix has incorrect sign in the (2,0) element"
  ✗  "the bug is in count()" — names code, not state

━━━ PHASE 2: MAP RESPONSIBILITY BOUNDARIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For the failing state, identify its stakeholders:

  PRODUCER     — computes or creates this state
  CONSUMER     — receives and acts on this state
  TRANSFORMER  — normalizes, converts, or modifies this state in transit
  VALIDATOR    — enforces or checks the invariant
  COORDINATOR  — orchestrates multiple owners (e.g., migration executor, signal dispatcher)

Name at least two stakeholders. The ownership boundary between them is where the
distractor lives.

━━━ PHASE 3: EVALUATE CANDIDATES AS STAKEHOLDERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL: Your distractor MUST be a symbol from the candidate list provided below.
Do not use any symbol not in the list — it may not exist in this version of the codebase.

For each candidate symbol in the list, ask:
  "What ownership role does this symbol play for the failing state I identified?"
  "Is it a PRODUCER (creates the state), CONSUMER (receives it), TRANSFORMER
   (modifies it in transit), VALIDATOR (should enforce it), or COORDINATOR?"

Then ask: "If this candidate were the alternative owner, would dismissing that
hypothesis require reading BOTH this candidate AND the true bug site?"

Prefer [imports this] candidates — they are directly called by the patched code.

━━━ PHASE 4: SELECT BY INVESTIGATION COST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Select the candidate where dismissal requires:
  ✓  Reading the distractor to understand what it does with the state
  ✓  Reading the true bug site to understand what it produces or expects
  ✓  Tracing the ownership boundary between them — neither alone is enough

DISQUALIFY any hypothesis where a developer can:
  ✗  Run a REPL call that immediately confirms or denies it
  ✗  Read one function body and settle the question
  ✗  Identify an architectural mismatch without opening any files
  ✗  Use a grep to prove no data connection exists
  ✗  Spot an obvious role mismatch (e.g., URL encoder blamed for namespace resolution)

━━━ HARD DISQUALIFIERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✗  Utility / primitive functions with no domain ownership stake
  ✗  Hypotheses that rest on version history or deprecated behavior
  ✗  Distractors that are downstream effects of the bug, not alternative causes
  ✗  Foundational operations (string formatting, basic arithmetic) — REPL-dismissible

━━━ PHASE 5: WRITE AS NATURAL BUG REPORT CONTEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The appended text in modified_problem_statement must read as organic engineering
context — not as a labeled alternative.

FORBIDDEN in the appended text:
  ✗  "Competing Hypothesis:", "Alternative Explanation:", "Hypothesis:"
  ✗  Any phrasing that labels this as a structured alternative
  ✗  The word "hypothesis" anywhere in the appended text
  ✗  Markdown headers (## ...) or bold section labels (**...**)

WRITE AS: an engineer's follow-up observation added to the original report.
Vary the framing across tasks."""

_SYSTEM_UNGROUNDED = f"""You are a research assistant creating synthetic task variants to evaluate coding agents.

GOAL: produce a root_cause_attribution variant that introduces a competing, INCORRECT causal
explanation — one that proposes an alternative OWNER of the failing state.

Do not optimize for: same file, same module, nearby symbols.
Optimize for: competing ownership of a specific failing state.

{_GOLD_EXAMPLE}

━━━ PHASE 1: IDENTIFY THE FAILING STATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

State the specific value, property, or invariant that is wrong.
Use DOMAIN TERMS — not code terms.

  ✓  "unused annotations are propagated into the count subquery"
  ✓  "backward migration leaves old index name in the schema state"
  ✗  "the bug is in count()" — names code, not state

━━━ PHASE 2: MAP RESPONSIBILITY BOUNDARIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For the failing state, identify its stakeholders:

  PRODUCER     — computes or creates this state
  CONSUMER     — receives and acts on this state
  TRANSFORMER  — normalizes, converts, or modifies this state in transit
  VALIDATOR    — enforces or checks the invariant
  COORDINATOR  — orchestrates multiple owners

Name at least two stakeholders. The ownership boundary between them is where the
distractor lives.

━━━ PHASE 3: GENERATE COMPETING HYPOTHESES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate 3–5 competing explanations for who should own the fix.
Pick a real symbol from the same subsystem. It must have a traceable data or
call connection to the failing state — not just directory proximity.

━━━ PHASE 4: SELECT BY INVESTIGATION COST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Select the hypothesis where dismissal requires reading BOTH the distractor AND
the true bug site and tracing ownership between them.

DISQUALIFY:
  ✗  Single function read settles it
  ✗  REPL call immediately confirms or denies
  ✗  Architectural mismatch obvious without opening files
  ✗  Utility/primitive with no ownership stake in the failing state
  ✗  Foundational operation (arithmetic, string format) — REPL-dismissible

━━━ HARD DISQUALIFIERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✗  No traceable call, data, or state connection to the failing state
  ✗  Hypothesis rests on version history or deprecated behavior
  ✗  Distractor is a downstream effect of the bug, not an alternative cause

━━━ PHASE 5: WRITE AS NATURAL BUG REPORT CONTEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The appended text in modified_problem_statement must read as organic engineering
context — not as a labeled alternative.

FORBIDDEN in the appended text:
  ✗  "Competing Hypothesis:", "Alternative Explanation:", "Hypothesis:"
  ✗  Any phrasing that labels this as a structured alternative
  ✗  The word "hypothesis" anywhere in the appended text
  ✗  Markdown headers (## ...) or bold section labels (**...**)

WRITE AS: an engineer's follow-up observation added to the original report.
Vary the framing across tasks."""

def _enforce_original_prefix(orig: str, mod: str) -> str:
    """Ensure modified_problem_statement starts with the verbatim original.
    LLMs strip tabs/zero-width chars from SWE-bench statements even when instructed not to."""
    if mod.startswith(orig):
        return mod

    # Normalise whitespace AND invisible unicode (zero-width spaces, BOM, etc.)
    _WS = re.compile(r'[\s​﻿ ]+')
    norm = lambda s: _WS.sub(' ', s)

    # Case 1: only normalisation differences — normalised strings share a prefix
    orig_n = norm(orig).rstrip()
    mod_n  = norm(mod)
    if mod_n.startswith(orig_n):
        tail = mod_n[len(orig_n):].strip()
        return (orig + "\n\n" + tail) if tail else orig

    # Case 2: heavier rewriting — anchor on the last N chars of orig,
    # trying progressively shorter anchors to skip over stripped characters
    for anchor_len in (80, 40, 20):
        anchor = norm(orig[-anchor_len:]).strip()
        pos = mod_n.rfind(anchor)
        if pos != -1:
            tail = mod_n[pos + len(anchor):].strip()
            return (orig + "\n\n" + tail) if tail else orig

    # Case 3: unrecoverable — return as-is (heuristic will reject it)
    return mod


_FRAMING_STARTERS = [
    "Initial investigation also flagged",
    "During triage, a colleague noted that",
    "An earlier review flagged",
    "One thread in the discussion points to",
    "Worth noting: the issue was also observed downstream of",
]


_DEFAULT_PROVENANCE = Provenance(
    dataset="SWE-bench Verified",
    license="MIT",
    source_url="https://github.com/princeton-nlp/SWE-bench",
)


def transform(
    task: SWEBenchTask,
    index: int = 1,
    provenance: Provenance | None = None,
) -> DaedalusTask:
    client = get_client()

    grounded_context = ""
    system_prompt = _SYSTEM_UNGROUNDED
    if task.patch:
        try:
            candidates = analyze(
                repo=task.repo,
                commit=task.base_commit,
                patch=task.patch,
                problem_statement=task.problem_statement,
            )
            if candidates:
                grounded_context = "\n\n" + format_for_prompt(candidates, task.base_commit)
                system_prompt = _SYSTEM
        except Exception:
            pass  # fall back to ungrounded generation

    candidate_constraint = (
        "\n\nCONSTRAINT: Your distractor must be one of the candidate symbols listed above. "
        "Do not use symbols not in this list — they may not exist at this commit."
        if grounded_context else ""
    )

    framing_starter = random.choice(_FRAMING_STARTERS)

    user_content = (
        f"Repository: {task.repo}\n\n"
        f"Original problem statement:\n{task.problem_statement}"
        f"{grounded_context}"
        f"{candidate_constraint}\n\n"
        f"FRAMING REQUIREMENT: The appended observation in modified_problem_statement "
        f"MUST open with exactly these words: \"{framing_starter} ...\"\n\n"
        "Work through all four phases: identify the failing state in domain terms, map the "
        "stakeholders, evaluate each candidate symbol for what ownership role it could play, "
        "then select the one whose hypothesis requires reading both it and the true bug site "
        "to resolve. Produce the variant."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    last_exc: Exception | None = None
    for attempt in range(3):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS_GENERATE,
            extra_body=PROVIDER_ROUTING,
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "create_variant"}},
            messages=messages,
        )
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            break
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            last_exc = exc
    else:
        raise RuntimeError(f"Failed to parse tool call after 3 attempts: {last_exc}")

    modified_ps = _enforce_original_prefix(
        task.problem_statement,
        result["modified_problem_statement"],
    )

    return DaedalusTask(
        source_instance_id=task.instance_id,
        variant_id=DaedalusTask.make_variant_id(task.instance_id, "root_cause_attribution", index),
        variant_type="root_cause_attribution",
        repo=task.repo,
        base_commit=task.base_commit,
        original_problem_statement=task.problem_statement,
        modified_problem_statement=modified_ps,
        transformation_scope=["problem_statement"],
        transformation_metadata=TransformationMetadata(
            description=result["description"],
            changes_made=result["changes_made"],
            intended_distractor=result["intended_distractor"],
        ),
        target_failure_mode="wrong_root_cause",
        quality_status="unverified",
        provenance=provenance or _DEFAULT_PROVENANCE,
    )
