"""mini-SWE-agent integration: runs full edit+test agent evaluation in Docker."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from daedalus.evaluation.schemas import EvaluationPair, EvaluationRun
from daedalus.evaluation.runner import compare_pair

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress manager no-op implementation
# ---------------------------------------------------------------------------

class _NoOpProgress:
    """Satisfies the RunBatchProgressManager interface without side effects."""

    def on_instance_start(self, instance_id: str) -> None:
        pass

    def update_instance_status(self, instance_id: str, status: str) -> None:
        pass

    def on_instance_end(self, instance_id: str, exit_status: str) -> None:
        pass

    def on_uncaught_exception(self, instance_id: str, exc: Exception) -> None:
        pass


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config(model: str) -> dict:
    """Load default swebench.yaml config and override the model name."""
    from minisweagent.config import get_config_from_spec, builtin_config_dir

    config_path = builtin_config_dir / "benchmarks" / "swebench.yaml"
    config = get_config_from_spec(str(config_path))

    if "model" not in config:
        config["model"] = {}
    config["model"]["model_name"] = model

    return config


# ---------------------------------------------------------------------------
# SWE-agent condition runner
# ---------------------------------------------------------------------------

def run_swe_condition(
    tasks: list[dict],
    condition: str,
    output_dir: Path,
    model: str,
    workers: int = 1,
) -> Path:
    """Run mini-SWE-agent for all tasks under one condition.

    Args:
        tasks: List of DaedalusTask dicts (with source_instance_id, original/modified problem statement).
        condition: "original" or "variant".
        output_dir: Base output directory; writes to output_dir/original/ or output_dir/variant/.
        model: Model name to pass to mini-SWE-agent.
        workers: Number of parallel workers (currently serial).

    Returns:
        Path to the written preds.json file.
    """
    from minisweagent.run.benchmarks.swebench import process_instance

    cond_dir = output_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    preds_path = cond_dir / "preds.json"

    # Load already-completed predictions for resume support
    existing_preds: dict = {}
    if preds_path.exists():
        try:
            existing_preds = json.loads(preds_path.read_text(encoding="utf-8"))
        except Exception:
            existing_preds = {}

    config = _build_config(model)
    progress = _NoOpProgress()

    errors = 0
    for task in tasks:
        instance_id: str = task["source_instance_id"]

        # Resume: skip if already predicted
        if instance_id in existing_preds:
            logger.debug("Skipping already-predicted instance: %s", instance_id)
            continue

        if condition == "original":
            problem_statement = task["original_problem_statement"]
        else:
            problem_statement = task["modified_problem_statement"]

        instance = {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
        }

        try:
            process_instance(instance, cond_dir, config, progress)

            # Merge newly written preds into existing_preds
            if preds_path.exists():
                try:
                    fresh = json.loads(preds_path.read_text(encoding="utf-8"))
                    existing_preds.update(fresh)
                except Exception:
                    pass
        except Exception as exc:
            errors += 1
            logger.error("Error processing instance %s: %s", instance_id, exc)

    if errors:
        logger.warning("run_swe_condition(%s): %d instance(s) failed", condition, errors)

    return preds_path


# ---------------------------------------------------------------------------
# Patch evaluator
# ---------------------------------------------------------------------------

def evaluate_patch(
    instance_id: str,
    patch: str,
    fail_to_pass: list[str],
    test_patch: str = "",
) -> bool | None:
    """Apply a model patch to the SWE-bench Docker image and run the test suite.

    Returns:
        True  — all FAIL_TO_PASS tests pass after applying the patch.
        False — tests fail (or patch application fails).
        None  — Docker is unavailable or the image cannot be pulled.
    """
    try:
        import docker  # type: ignore
    except ImportError:
        logger.warning("docker SDK not installed — skipping patch evaluation")
        return None

    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        logger.warning("Docker unavailable (%s) — skipping patch evaluation", exc)
        return None

    # Derive image name from instance_id (SWE-bench convention)
    image_name = (
        f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"
    )

    # Ensure image is available
    try:
        client.images.get(image_name)
    except Exception:
        try:
            client.images.pull(image_name)
        except Exception as exc:
            logger.warning("Could not pull image %s: %s — skipping", image_name, exc)
            return None

    container = None
    try:
        container = client.containers.run(
            image_name,
            command="bash",
            detach=True,
            tty=True,
            working_dir="/testbed",
        )

        def _exec(cmd: str) -> tuple[int, str]:
            exit_code, output = container.exec_run(
                ["bash", "-c", cmd],
                workdir="/testbed",
            )
            return exit_code, output.decode("utf-8", errors="replace") if output else ""

        # Reset repo to clean state
        _exec("git reset --hard HEAD && git clean -fd")

        def _put_file(name: str, content: str) -> None:
            """Write content into /tmp/<name> inside the container via put_archive."""
            import io
            import tarfile
            data = content.encode("utf-8")
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            buf.seek(0)
            container.put_archive("/tmp", buf)

        # Apply model patch
        if patch:
            _put_file("_model_patch.diff", patch)
            apply_code, apply_out = _exec(
                "git apply --whitespace=nowarn /tmp/_model_patch.diff"
                " || patch -p1 -f --ignore-whitespace < /tmp/_model_patch.diff"
            )
            if apply_code != 0:
                logger.debug("Patch application failed for %s: %s", instance_id, apply_out)
                return False

        # Apply test patch if provided
        if test_patch:
            _put_file("_test_patch.diff", test_patch)
            _exec(
                "git apply --whitespace=nowarn /tmp/_test_patch.diff"
                " || patch -p1 -f --ignore-whitespace < /tmp/_test_patch.diff"
            )

        # Run failing tests
        tests_str = " ".join(fail_to_pass)
        test_cmd = f"python -m pytest {tests_str} -x -q --no-header --tb=no"
        exit_code, _ = _exec(test_cmd)
        return exit_code == 0

    except Exception as exc:
        logger.error("evaluate_patch failed for %s: %s", instance_id, exc)
        return None
    finally:
        if container is not None:
            try:
                container.stop(timeout=10)
                container.remove(force=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# SWE-bench metadata loader
# ---------------------------------------------------------------------------

def load_swebench_metadata(instance_ids: set[str]) -> dict[str, dict]:
    """Load FAIL_TO_PASS and test_patch from the HuggingFace SWE-bench dataset.

    Returns:
        Mapping of instance_id -> {"fail_to_pass": list[str], "test_patch": str}.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    metadata: dict[str, dict] = {}
    for row in ds:
        iid = row["instance_id"]
        if iid not in instance_ids:
            continue
        raw_ftp = row.get("FAIL_TO_PASS", "[]")
        try:
            fail_to_pass: list[str] = json.loads(raw_ftp) if isinstance(raw_ftp, str) else list(raw_ftp)
        except Exception:
            fail_to_pass = []
        metadata[iid] = {
            "fail_to_pass": fail_to_pass,
            "test_patch": row.get("test_patch", ""),
        }

    return metadata


# ---------------------------------------------------------------------------
# Fabricated image preparation
# ---------------------------------------------------------------------------

def _swe_image_name(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def _put_file_in_container(container, name: str, content: str) -> None:
    import io
    import tarfile
    data = content.encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    container.put_archive("/tmp", buf)


def prepare_fabricated_image(instance_id: str, diff: str) -> tuple[str, str | None]:
    """Apply a fabricated diff to the SWE-bench Docker image and retag it locally.

    mini-SWE-agent selects the image by instance_id convention, so retagging the
    local image causes the agent to see the modified codebase without any config
    changes. The original image ID is returned so the tag can be restored afterward.

    Returns (image_name, original_image_id). original_image_id is None on failure,
    in which case the variant runs against the unmodified image.
    """
    try:
        import docker
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        logger.warning("Docker unavailable — cannot prepare fabricated image: %s", exc)
        return _swe_image_name(instance_id), None

    image_name = _swe_image_name(instance_id)
    try:
        original_image = client.images.get(image_name)
    except Exception:
        try:
            original_image = client.images.pull(image_name)
        except Exception as exc:
            logger.warning("Could not get/pull %s: %s", image_name, exc)
            return image_name, None

    original_id: str = original_image.id
    container = None
    try:
        container = client.containers.run(
            image_name, "bash", detach=True, tty=True, working_dir="/testbed"
        )
        diff = diff.replace("\r\n", "\n").replace("\r", "\n")
        _put_file_in_container(container, "_fab.diff", diff)
        exit_code, output = container.exec_run(
            ["bash", "-c",
             "git apply --whitespace=nowarn /tmp/_fab.diff"
             " || patch -p1 -f --ignore-whitespace < /tmp/_fab.diff"],
            workdir="/testbed",
        )
        if exit_code != 0:
            out_str = output.decode("utf-8", errors="replace") if output else ""
            logger.warning("Could not apply fabricated diff in container: %s", out_str)
            return image_name, None

        repo, tag = image_name.rsplit(":", 1)
        container.commit(repository=repo, tag=tag)
        return image_name, original_id

    except Exception as exc:
        logger.error("Failed to prepare fabricated image for %s: %s", instance_id, exc)
        return image_name, None
    finally:
        if container is not None:
            try:
                container.stop(timeout=10)
                container.remove(force=True)
            except Exception:
                pass


def restore_image(image_name: str, original_image_id: str) -> None:
    """Restore the SWE-bench image tag to the pre-fabrication image."""
    try:
        import docker
        client = docker.from_env()
        repo, tag = image_name.rsplit(":", 1)
        client.images.get(original_image_id).tag(repo, tag)
    except Exception as exc:
        logger.warning(
            "Could not restore %s to %s — run: docker pull %s  (%s)",
            image_name, original_image_id[:12], image_name, exc,
        )


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

def build_swe_pairs(
    tasks: list[dict],
    orig_preds_path: Path,
    var_preds_path: Path,
    metadata: dict[str, dict],
) -> tuple[list[EvaluationRun], list[EvaluationPair]]:
    """Build EvaluationRun and EvaluationPair objects from SWE-agent predictions.

    For each task, evaluates original and variant patches via Docker, then
    calls compare_pair() to derive the evaluation bucket.

    Returns:
        (runs, pairs) — flat lists ready for export.
    """
    # Load predictions from both preds.json files
    def _load_preds(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    orig_preds = _load_preds(orig_preds_path)
    var_preds = _load_preds(var_preds_path)

    runs: list[EvaluationRun] = []
    pairs: list[EvaluationPair] = []

    import time

    for task in tasks:
        instance_id: str = task["source_instance_id"]
        variant_id: str = task.get("variant_id", instance_id)

        meta = metadata.get(instance_id, {})
        fail_to_pass: list[str] = meta.get("fail_to_pass", [])
        test_patch: str = meta.get("test_patch", "")

        # Retrieve patches (empty string if agent didn't produce one)
        orig_entry = orig_preds.get(instance_id, {})
        var_entry = var_preds.get(instance_id, {})
        orig_patch: str = orig_entry.get("model_patch", "") if isinstance(orig_entry, dict) else ""
        var_patch: str = var_entry.get("model_patch", "") if isinstance(var_entry, dict) else ""

        # Evaluate patches
        orig_resolved = evaluate_patch(instance_id, orig_patch, fail_to_pass, test_patch)
        var_resolved = evaluate_patch(instance_id, var_patch, fail_to_pass, test_patch)

        t_now = int(time.time())

        original_run = EvaluationRun(
            run_id=f"{variant_id}__original__{t_now}",
            variant_id=variant_id,
            source_instance_id=instance_id,
            condition="original",
            agent_id="mini-swe-agent",
            stopped_reason="natural",
            model_patch=orig_patch,
            resolved=orig_resolved,
        )

        variant_run = EvaluationRun(
            run_id=f"{variant_id}__variant__{t_now}",
            variant_id=variant_id,
            source_instance_id=instance_id,
            condition="variant",
            agent_id="mini-swe-agent",
            stopped_reason="natural",
            model_patch=var_patch,
            resolved=var_resolved,
        )

        pair = compare_pair(original_run, variant_run)

        runs.append(original_run)
        runs.append(variant_run)
        pairs.append(pair)

    return runs, pairs
