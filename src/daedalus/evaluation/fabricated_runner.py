"""SWE-agent evaluation with fabricated repository artifacts applied via Docker image modification."""
from __future__ import annotations

import logging
from pathlib import Path

from daedalus.evaluation.schemas import EvaluationPair, EvaluationRun
from daedalus.evaluation.swe_runner import (
    build_swe_pairs,
    load_swebench_metadata,
    prepare_fabricated_image,
    restore_image,
    run_swe_condition,
)
from daedalus.fabricator.schemas import FabricatedArtifact

logger = logging.getLogger(__name__)


def run_swe_fabricated_pair(
    task: dict,
    artifact: FabricatedArtifact,
    output_dir: Path,
    model: str,
) -> tuple[list[EvaluationRun], list[EvaluationPair]]:
    """Run mini-SWE-agent on original (clean) then variant (fabricated diff applied) condition.

    For the variant condition the fabricated diff is pre-applied to the SWE-bench
    Docker image before mini-SWE-agent starts, so the agent investigates the modified
    codebase without knowing about the fabrication. The image tag is restored after
    the run regardless of outcome.
    """
    instance_id = artifact.source_instance_id
    artifact_dir = output_dir / artifact.artifact_id

    metadata = load_swebench_metadata({instance_id})

    orig_preds_path = run_swe_condition([task], "original", artifact_dir, model)

    image_name, original_image_id = prepare_fabricated_image(instance_id, artifact.diff)
    if original_image_id is None:
        logger.warning(
            "Could not apply fabricated diff for %s — variant runs on unmodified image",
            artifact.artifact_id,
        )

    try:
        var_preds_path = run_swe_condition([task], "variant", artifact_dir, model)
    finally:
        if original_image_id is not None:
            restore_image(image_name, original_image_id)

    return build_swe_pairs([task], orig_preds_path, var_preds_path, metadata)
