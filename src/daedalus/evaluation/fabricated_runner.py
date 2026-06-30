"""Paired agent evaluation with fabricated repository artifacts applied to the repo."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from daedalus.evaluation.runner import compare_pair, run_condition
from daedalus.evaluation.schemas import EvaluationPair, EvaluationRun
from daedalus.fabricator.schemas import FabricatedArtifact

logger = logging.getLogger(__name__)


def _apply_diff(repo_path: Path, diff: str) -> bool:
    r = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=diff.encode(),
        cwd=str(repo_path),
        capture_output=True,
    )
    if r.returncode == 0:
        return True
    r = subprocess.run(
        ["patch", "-p1", "-f", "--ignore-whitespace"],
        input=diff.encode(),
        cwd=str(repo_path),
        capture_output=True,
    )
    return r.returncode == 0


def _revert_diff(repo_path: Path, diff: str) -> bool:
    r = subprocess.run(
        ["git", "apply", "-R", "--whitespace=nowarn"],
        input=diff.encode(),
        cwd=str(repo_path),
        capture_output=True,
    )
    return r.returncode == 0


def run_fabricated_pair(
    task: dict,
    artifact: FabricatedArtifact,
    repo_path: Path,
    gold_patch: str = "",
    model: str | None = None,
    max_turns: int | None = None,
) -> tuple[EvaluationRun, EvaluationRun, EvaluationPair]:
    """Run original condition (clean repo) then variant condition (artifact applied).

    The diff is reverted after the variant run regardless of outcome. If apply
    fails, the variant still runs against the unmodified repo and a warning is logged.
    """
    original_run = run_condition(
        task, "original", repo_path,
        max_turns=max_turns, gold_patch=gold_patch, model=model,
    )

    applied = _apply_diff(repo_path, artifact.diff)
    if not applied:
        logger.warning(
            "Could not apply diff for %s — variant runs on unmodified repo",
            artifact.artifact_id,
        )

    try:
        variant_run = run_condition(
            task, "variant", repo_path,
            max_turns=max_turns, gold_patch=gold_patch, model=model,
        )
    finally:
        if applied and not _revert_diff(repo_path, artifact.diff):
            logger.error(
                "Failed to revert diff for %s — repo at %s may be dirty; run: git checkout -- .",
                artifact.artifact_id,
                repo_path,
            )

    return original_run, variant_run, compare_pair(original_run, variant_run)
