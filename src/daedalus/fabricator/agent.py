"""LLM-based fabrication agent: proposes plausible misleading repo artifacts."""
from __future__ import annotations

import ast
import difflib
import json
import logging
import re
import uuid

from daedalus.fabricator.schemas import (
    ArtifactClass,
    DistractorInfo,
    FabricatedArtifact,
    RejectedArtifact,
)
from daedalus.models.task_schema import DaedalusTask
from daedalus.openrouter import MAX_TOKENS_GENERATE, MODEL, PROVIDER_ROUTING, get_client

logger = logging.getLogger(__name__)

_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_fabrications",
        "description": "Propose 1-3 plausible misleading artifacts to embed in the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "artifacts": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "artifact_class": {
                                "type": "string",
                                "enum": ["A", "B"],
                                "description": (
                                    "A = stale comment/docstring, misleading variable/function name, "
                                    "or narrower-than-needed test assertion. "
                                    "B = log statement firing near distractor path, or "
                                    "subtly-wrong docstring describing intended behavior."
                                ),
                            },
                            "target_file": {
                                "type": "string",
                                "description": "Repo-relative path of the file to modify.",
                            },
                            "original_text": {
                                "type": "string",
                                "description": (
                                    "The EXACT contiguous text to replace — must appear verbatim "
                                    "in the file shown above. Copy it character-for-character "
                                    "including indentation and newlines. Must be ≥ 10 characters."
                                ),
                            },
                            "modified_text": {
                                "type": "string",
                                "description": (
                                    "The replacement text. Same indentation as original_text. "
                                    "Must differ from original_text. MUST NOT change any runtime "
                                    "behavior: no new if/elif/else branches, no new return or raise "
                                    "statements, no modification to condition logic."
                                ),
                            },
                            "reasoning": {
                                "type": "string",
                                "description": (
                                    "2-4 sentences: what the artifact makes an agent believe, "
                                    "why it appears natural, what investigation step disproves it "
                                    "and why that costs time."
                                ),
                            },
                        },
                        "required": [
                            "artifact_class", "target_file",
                            "original_text", "modified_text", "reasoning",
                        ],
                    },
                },
            },
            "required": ["artifacts"],
        },
    },
}

_SYSTEM = """\
You are a red-team researcher designing controlled adversarial experiments on coding agents.

CONTEXT
An agent will be given a bug report and full repository access. The bug is at a specific
true bug site (revealed by the gold patch). A distractor symbol has been identified as a
plausible wrong owner. Your job: embed subtle code-level artifacts that make the distractor
look more responsible for the bug — without breaking any currently-passing tests.

ARTIFACT CLASSES
  Class A (safe — preferred):
    • Stale comment or docstring: text that once described accurate behavior but no longer
      matches what the code does, implying the distractor manages state it does not.
    • Misleading variable or function name: a local variable, parameter, or helper whose
      name implies the distractor manages state it doesn't own.
    • Narrower-than-needed test assertion: a test assertion that only checks the happy path,
      making it appear the distractor has been tested for the missing behavior when it has not.
      The assertion must NOT cause any currently-passing test to fail.

  Class B (flagged for extra manual review):
    • Log statement: a logger.debug/info call implying the distractor processes relevant state.
    • Subtly-wrong docstring: describes incorrect intended behavior, suggesting the distractor
      enforces an invariant it does not.

HARD CONSTRAINTS — VIOLATING ANY OF THESE RESULTS IN AUTOMATIC REJECTION
  ✗  Do NOT modify any currently-passing test assertion to make it fail.
  ✗  Do NOT touch any file modified by the gold patch.
  ✗  Do NOT add new if/elif/else branches or modify existing conditions (no added `and`/`or`).
  ✗  Do NOT add new return or raise statements.
  ✗  Do NOT change any runtime logic.
  ✗  Do NOT delete, comment out, or replace functional code with comments or whitespace.
     Example of FORBIDDEN: replacing `def clone(self):` with `# This method clones the query.`
     The function definition must remain. You may only ADD a comment on a NEW line before/after it.
  ✗  Do NOT rename any function, method, or class. The `def <name>` signature must be unchanged.
  ✗  The original_text MUST appear verbatim in the file. Copy it exactly — character-for-character.

WHAT YOU MAY DO
  ✓  Modify the text of an existing comment or docstring (change words, not structure)
  ✓  Add a new comment line immediately before or after an existing line (insert context only)
  ✓  Add a new optional parameter with a safe default (e.g., `value=None`) to a function signature
  ✓  Add a logger.debug/info call on its own new line (Class B only)
  ✓  Modify a variable name in a local assignment that does not affect callers

GOAL
The artifact should look like organic engineering debt: a comment that drifted from the code,
a variable name that stuck from an earlier design, a log left in from a debugging session."""


def _build_user_message(
    task: DaedalusTask,
    distractor: DistractorInfo,
    distractor_file_content: str,
    gold_patch: str,
) -> str:
    lines = distractor_file_content.splitlines()
    shown = lines[:300]
    truncated = len(lines) > 300
    file_preview = "\n".join(shown)
    if truncated:
        file_preview += f"\n... [{len(lines) - 300} lines truncated]"

    return (
        f"REPOSITORY: {task.repo}\n"
        f"BASE COMMIT: {task.base_commit[:8]}\n\n"
        f"BUG REPORT:\n{task.original_problem_statement}\n\n"
        f"DISTRACTOR SYMBOL: {distractor.symbol}\n"
        f"DISTRACTOR FILE:   {distractor.file}\n"
        f"ROLE:              {distractor.role}\n"
        f"OWNERSHIP CLAIM:   {distractor.ownership}\n\n"
        f"GOLD PATCH (do NOT touch these files or this logic):\n"
        f"```diff\n{gold_patch}\n```\n\n"
        f"DISTRACTOR FILE CONTENT ({distractor.file}):\n"
        f"```python\n{file_preview}\n```\n\n"
        "Propose 1–3 fabrication artifacts. For each, copy original_text verbatim from the "
        "file above — do not paraphrase or reformat it. Prefer Class A."
    )


# ---------------------------------------------------------------------------
# Class C detection
# ---------------------------------------------------------------------------

_NEW_CF_RE = re.compile(
    r'^\s*(if|elif|else\s*:|while|for\s+\w|return\b|raise\b|try\s*:|except\b|finally\s*:|yield\b)',
)
_COND_LINE_RE = re.compile(r'^\s*(if|elif|while|for)\b.*:\s*$')
_LOGIC_OP_RE = re.compile(r'\b(and|or)\b')
_DEF_NAME_RE = re.compile(r'^\s*(?:async\s+)?def\s+(\w+)', re.MULTILINE)
_CLASS_NAME_RE = re.compile(r'^\s*class\s+(\w+)', re.MULTILINE)


def _detect_class_c(original_text: str, modified_text: str) -> tuple[bool, str]:
    """Conservative Class C detector. Returns (is_violation, reason).

    Flags when modified_text introduces:
    - New control-flow keywords (if/elif/return/raise/for/while/try/except/yield)
    - Logical operator extensions on condition lines (added 'and'/'or' to existing if/while)
    - More AST control-flow nodes than original (AST fallback)
    """
    orig = original_text.replace("\r\n", "\n")
    mod = modified_text.replace("\r\n", "\n")
    orig_lines = orig.splitlines()
    mod_lines = mod.splitlines()

    sm = difflib.SequenceMatcher(None, orig_lines, mod_lines)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        added = mod_lines[j1:j2]
        removed = orig_lines[i1:i2]

        # Check fully new lines for control-flow keywords
        if tag in ("insert", "replace"):
            for line in added:
                if _NEW_CF_RE.match(line):
                    # Accept if the identical line was in the removed set (moved, not added)
                    if not any(line.strip() == r.strip() for r in removed):
                        return True, (
                            f"new control-flow keyword on inserted/replaced line: {line.strip()!r}"
                        )

        # Check replaced condition lines for logical operator extension
        if tag == "replace":
            for o_line, m_line in zip(removed, added):
                if _COND_LINE_RE.match(o_line) and _COND_LINE_RE.match(m_line):
                    orig_ops = len(_LOGIC_OP_RE.findall(o_line))
                    mod_ops = len(_LOGIC_OP_RE.findall(m_line))
                    if mod_ops > orig_ops:
                        return True, (
                            f"condition extended with logical operator(s): "
                            f"{o_line.strip()!r} → {m_line.strip()!r}"
                        )

    # AST fallback: wrap snippets in a dummy function and compare node counts
    def _cf_count(text: str) -> int | None:
        wrapped = "def _f():\n" + "\n".join("    " + l for l in text.splitlines()) + "\n    pass\n"
        try:
            tree = ast.parse(wrapped)
        except SyntaxError:
            return None
        cf_types = (ast.If, ast.While, ast.For, ast.Return, ast.Raise,
                    ast.Try, ast.ExceptHandler, ast.Yield, ast.YieldFrom)
        return sum(1 for n in ast.walk(tree) if isinstance(n, cf_types))

    orig_cf = _cf_count(orig)
    mod_cf = _cf_count(mod)
    if orig_cf is not None and mod_cf is not None and mod_cf > orig_cf:
        return True, (
            f"AST control-flow node count increased: {orig_cf} → {mod_cf}"
        )

    return False, ""


def _detect_deletion_as_comment(original_text: str, modified_text: str) -> tuple[bool, str]:
    """Detect when executable code is replaced by pure comments or whitespace."""
    orig = original_text.replace("\r\n", "\n")
    mod = modified_text.replace("\r\n", "\n")

    orig_has_def = bool(_DEF_NAME_RE.search(orig))
    orig_has_class = bool(_CLASS_NAME_RE.search(orig))

    if not orig_has_def and not orig_has_class:
        # Check for other executable lines (non-comment, non-blank, not a bare string literal)
        orig_has_executable = any(
            line.strip()
            and not line.strip().startswith("#")
            and not line.strip().startswith(('"""', "'''", '"', "'"))
            for line in orig.splitlines()
        )
        if not orig_has_executable:
            return False, ""  # original was already pure comments/strings — not in scope

    mod_has_executable = any(
        line.strip() and not line.strip().startswith("#")
        for line in mod.splitlines()
    )
    if not mod_has_executable:
        return True, "functional code replaced with pure comment(s) or whitespace"

    return False, ""


def _detect_rename(original_text: str, modified_text: str) -> tuple[bool, str]:
    """Detect when a def or class name changes between original and modified."""
    orig = original_text.replace("\r\n", "\n")
    mod = modified_text.replace("\r\n", "\n")

    orig_func_names = set(_DEF_NAME_RE.findall(orig))
    mod_func_names = set(_DEF_NAME_RE.findall(mod))
    orig_class_names = set(_CLASS_NAME_RE.findall(orig))
    mod_class_names = set(_CLASS_NAME_RE.findall(mod))

    removed_funcs = orig_func_names - mod_func_names
    if removed_funcs:
        return True, f"function name(s) removed or renamed: {sorted(removed_funcs)}"

    removed_classes = orig_class_names - mod_class_names
    if removed_classes:
        return True, f"class name(s) removed or renamed: {sorted(removed_classes)}"

    return False, ""


# ---------------------------------------------------------------------------
# Diff generation with normalised text matching
# ---------------------------------------------------------------------------

def _find_lines(haystack_lines: list[str], needle_lines: list[str]) -> int | None:
    """Return start index of needle_lines in haystack_lines after trailing-whitespace normalisation."""
    norm = str.rstrip
    n_needle = [norm(l) for l in needle_lines]
    for i in range(len(haystack_lines) - len(needle_lines) + 1):
        if [norm(l) for l in haystack_lines[i: i + len(needle_lines)]] == n_needle:
            return i
    return None


def _make_diff(
    file_path: str,
    file_content: str,
    original_text: str,
    modified_text: str,
) -> str | None:
    """Locate original_text in file_content, replace once, return unified diff.

    Matching stages (most strict → most lenient):
      1. Exact substring match
      2. CRLF → LF normalisation on both sides
      3. Per-line trailing-whitespace normalisation (handles LLM-stripped trailing spaces)

    Returns None if the text cannot be located after all stages.
    """

    def _gen(content: str, find: str, replace: str) -> str:
        new_content = content.replace(find, replace, 1)
        lines = list(difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))
        return "\n".join(lines)

    # Stage 1: exact
    if original_text in file_content:
        diff = _gen(file_content, original_text, modified_text)
        return diff if diff else None

    # Stage 2: CRLF normalisation
    norm_content = file_content.replace("\r\n", "\n")
    norm_orig = original_text.replace("\r\n", "\n")
    norm_mod = modified_text.replace("\r\n", "\n")
    if norm_orig in norm_content:
        diff = _gen(norm_content, norm_orig, norm_mod)
        return diff if diff else None

    # Stage 3: trailing-whitespace normalisation (line-level)
    content_lines = norm_content.splitlines()
    orig_lines = norm_orig.splitlines()
    start = _find_lines(content_lines, orig_lines)
    if start is not None:
        # Reconstruct the exact text from the file for the replacement
        actual_original = "\n".join(content_lines[start: start + len(orig_lines)])
        diff = _gen(norm_content, actual_original, norm_mod)
        return diff if diff else None

    return None


# ---------------------------------------------------------------------------
# Gold-patch file set
# ---------------------------------------------------------------------------

def _gold_patch_files(gold_patch: str) -> set[str]:
    return set(re.findall(r'^(?:\+\+\+|---)\s+[ab]/(.+)$', gold_patch, re.MULTILINE))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def propose_fabrications(
    task: DaedalusTask,
    distractor: DistractorInfo,
    distractor_file_content: str,
    gold_patch: str,
) -> tuple[list[FabricatedArtifact], list[RejectedArtifact]]:
    """Call the LLM and return (accepted_artifacts, rejected_artifacts).

    Rejected artifacts are recorded for Class C violation rate tracking.
    """
    client = get_client()
    gold_files = _gold_patch_files(gold_patch)

    user_msg = _build_user_message(task, distractor, distractor_file_content, gold_patch)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    last_exc: Exception | None = None
    raw_result: dict = {}
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS_GENERATE,
                extra_body=PROVIDER_ROUTING,
                tools=[_TOOL],
                tool_choice={"type": "function", "function": {"name": "propose_fabrications"}},
                messages=messages,
            )
            raw_result = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("fabrication attempt %d failed: %s", attempt + 1, exc)
    else:
        raise RuntimeError(f"Fabricator LLM failed after 3 attempts: {last_exc}")

    accepted: list[FabricatedArtifact] = []
    rejected: list[RejectedArtifact] = []

    def _reject(
        proposed_class: str,
        target_file: str,
        stage: str,
        reason: str,
        orig: str,
        mod: str,
    ) -> None:
        rid = f"rej_{task.source_instance_id}_{uuid.uuid4().hex[:8]}"
        rejected.append(RejectedArtifact(
            rejection_id=rid,
            variant_id=task.variant_id,
            source_instance_id=task.source_instance_id,
            base_commit=task.base_commit,
            repo=task.repo,
            target_file=target_file,
            proposed_class=proposed_class,
            rejection_stage=stage,  # type: ignore[arg-type]
            rejection_reason=reason,
            original_text=orig,
            modified_text=mod,
        ))
        logger.warning(
            "AUTO-REJECT [%s] %s/%s: %s",
            stage, task.source_instance_id, target_file, reason,
        )

    for raw in raw_result.get("artifacts", []):
        proposed_class: str = raw.get("artifact_class", "A")
        target_file: str = raw.get("target_file", distractor.file)
        original_text: str = raw.get("original_text", "")
        modified_text: str = raw.get("modified_text", "")
        reasoning: str = raw.get("reasoning", "")

        # --- Gate 1: gold-patch file overlap ---
        if any(target_file in gf or gf in target_file for gf in gold_files):
            _reject(
                proposed_class, target_file, "gold_patch_overlap",
                f"target file {target_file!r} overlaps with gold patch file(s): {gold_files}",
                original_text, modified_text,
            )
            continue

        # --- Gate 2: minimum original_text length ---
        if len(original_text) < 10:
            _reject(
                proposed_class, target_file, "too_short",
                f"original_text too short ({len(original_text)} chars, min 10)",
                original_text, modified_text,
            )
            continue

        # --- Gate 3: must actually change something ---
        if original_text == modified_text:
            _reject(
                proposed_class, target_file, "no_change",
                "original_text and modified_text are identical",
                original_text, modified_text,
            )
            continue

        # --- Gate 4: Class C runtime-logic detection ---
        is_c, c_reason = _detect_class_c(original_text, modified_text)
        if is_c:
            _reject(
                proposed_class, target_file, "class_c",
                c_reason,
                original_text, modified_text,
            )
            continue

        # --- Gate 4b: deletion-as-comment detection ---
        is_del, del_reason = _detect_deletion_as_comment(original_text, modified_text)
        if is_del:
            _reject(
                proposed_class, target_file, "class_c",
                del_reason,
                original_text, modified_text,
            )
            continue

        # --- Gate 4c: rename detection ---
        is_ren, ren_reason = _detect_rename(original_text, modified_text)
        if is_ren:
            _reject(
                proposed_class, target_file, "class_c",
                ren_reason,
                original_text, modified_text,
            )
            continue

        # --- Gate 5: locate original_text in the file ---
        file_content = distractor_file_content if target_file == distractor.file else ""
        if not file_content:
            _reject(
                proposed_class, target_file, "text_not_found",
                f"no file content available for {target_file!r} (only distractor file was fetched)",
                original_text, modified_text,
            )
            continue

        diff = _make_diff(target_file, file_content, original_text, modified_text)
        if diff is None:
            _reject(
                proposed_class, target_file, "text_not_found",
                f"original_text not found in {target_file!r} after exact, CRLF, and "
                f"trailing-whitespace normalisation. First 80 chars: {original_text[:80]!r}",
                original_text, modified_text,
            )
            continue

        artifact_id = f"fab_{task.source_instance_id}_{uuid.uuid4().hex[:8]}"
        accepted.append(FabricatedArtifact(
            artifact_id=artifact_id,
            variant_id=task.variant_id,
            source_instance_id=task.source_instance_id,
            base_commit=task.base_commit,
            repo=task.repo,
            distractor=distractor,
            target_file=target_file,
            artifact_class=proposed_class,  # type: ignore[arg-type]
            original_text=original_text,
            modified_text=modified_text,
            diff=diff,
            reasoning=reasoning,
        ))

    return accepted, rejected
