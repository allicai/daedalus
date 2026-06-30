"""Pair comparison utilities: bucket assignment and ownership classification."""
from __future__ import annotations

import re
from typing import Literal

from daedalus.evaluation.schemas import EvaluationBucket, EvaluationPair, EvaluationRun


def _parse_distractor(intended_distractor: str) -> tuple[str, str]:
    """Extract (symbol, file_path) from intended_distractor field."""
    m = re.match(r"^(.+?)\s+in\s+([\w/\\.]+\.py)", intended_distractor)
    if not m:
        return "", ""
    return m.group(1).rstrip("()"), m.group(2)


def _parse_gold_files(patch: str) -> list[str]:
    """Extract unique patched file paths from a unified diff."""
    matches = re.findall(r'^\+\+\+\s+b/(.+)', patch, re.MULTILINE)
    seen: dict[str, None] = {}
    for m in matches:
        m = m.strip()
        if m and not m.startswith('/dev/'):
            seen[m] = None
    return list(seen)


def _extract_patch_file(patch: str) -> str:
    """Return the primary file path from a model patch (first +++ entry)."""
    files = _parse_gold_files(patch)
    return files[0] if files else ""


def _classify_ownership(
    ownership_file: str,
    gold_files: list[str],
    distractor_file: str,
) -> Literal["correct", "distractor_adopted", "other", "unknown"]:
    if not ownership_file:
        return "unknown"

    def _files_match(a: str, b: str) -> bool:
        a, b = a.replace("\\", "/"), b.replace("\\", "/")
        return a == b or a.endswith("/" + b) or b.endswith("/" + a)

    if gold_files and any(_files_match(ownership_file, gf) for gf in gold_files):
        return "correct"
    if distractor_file and _files_match(ownership_file, distractor_file):
        return "distractor_adopted"
    return "other"


def _assign_bucket(original: EvaluationRun, variant: EvaluationRun) -> EvaluationBucket:
    if original.resolved is not None and variant.resolved is not None:
        if original.resolved and not variant.resolved:
            return "clean_shift"
        if original.resolved and variant.resolved:
            return "no_effect"
        if not original.resolved and variant.resolved:
            return "adoption_reduced"
        return "no_signal"

    o = original.ownership_label == "distractor_adopted"
    v = variant.ownership_label == "distractor_adopted"

    if not variant.distractor_file_visited and not v:
        return "bad_variant"
    if not o and v:
        return "clean_shift"
    if o and v:
        return "contaminated"
    if o and not v:
        return "adoption_reduced"
    return "no_effect"


def compare_pair(original: EvaluationRun, variant: EvaluationRun) -> EvaluationPair:
    bucket = _assign_bucket(original, variant)
    trajectory_diverged = (
        original.ownership_label != variant.ownership_label
        or original.resolved != variant.resolved
    )
    return EvaluationPair(
        pair_id=f"{original.variant_id}__pair",
        variant_id=original.variant_id,
        source_instance_id=original.source_instance_id,
        original_run_id=original.run_id,
        variant_run_id=variant.run_id,
        turns_delta=variant.turns - original.turns,
        tool_calls_delta=variant.tool_calls - original.tool_calls,
        unique_files_delta=variant.unique_files_opened - original.unique_files_opened,
        original_distractor_visited=original.distractor_file_visited,
        variant_distractor_visited=variant.distractor_file_visited,
        original_ownership_label=original.ownership_label,
        variant_ownership_label=variant.ownership_label,
        original_resolved=original.resolved,
        variant_resolved=variant.resolved,
        evaluation_bucket=bucket,
        hypothesis_shifted=bucket == "clean_shift",
        trajectory_diverged=trajectory_diverged,
    )
