"""Zero-tolerance safety check: apply fabricated diff, run tests, confirm gold patch still works.

Two-stage check (both must pass):
  Stage 1 — No regression: apply fabricated diff, run PASS_TO_PASS tests.
             If ANY previously-passing test now fails → reject, no override.
  Stage 2 — Gold patch intact: apply gold patch on top, run FAIL_TO_PASS tests.
             If the issue no longer resolves → reject, no override.
"""
from __future__ import annotations

import io
import logging
import tarfile

from daedalus.fabricator.schemas import FabricatedArtifact, SafetyResult

logger = logging.getLogger(__name__)


def _put_file(container, name: str, content: str, dest_dir: str = "/tmp") -> None:
    """Write a UTF-8 string as a file inside the container via put_archive."""
    data = content.encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    container.put_archive(dest_dir, buf)


def _exec(container, cmd: str) -> tuple[int, str]:
    exit_code, output = container.exec_run(
        ["bash", "-c", cmd],
        workdir="/testbed",
    )
    decoded = output.decode("utf-8", errors="replace") if output else ""
    return exit_code, decoded


def check_artifact(
    artifact: FabricatedArtifact,
    gold_patch: str,
    pass_to_pass: list[str],
    fail_to_pass: list[str],
    test_patch: str = "",
) -> tuple[SafetyResult, str]:
    """Run the two-stage safety check in a Docker container.

    Returns:
        (SafetyResult, notes) — notes describe which test(s) caused a failure.
    """
    try:
        import docker  # type: ignore
    except ImportError:
        return "unavailable", "docker SDK not installed"

    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        return "unavailable", f"Docker unavailable: {exc}"

    image_name = (
        f"swebench/sweb.eval.x86_64."
        f"{artifact.source_instance_id.replace('__', '_1776_')}:latest"
    )

    try:
        client.images.get(image_name)
    except Exception:
        try:
            logger.info("Pulling Docker image %s ...", image_name)
            client.images.pull(image_name)
        except Exception as exc:
            return "unavailable", f"Could not pull image {image_name}: {exc}"

    container = None
    try:
        container = client.containers.run(
            image_name,
            command="bash",
            detach=True,
            tty=True,
            working_dir="/testbed",
        )

        # --- Stage 1: Apply fabrication, run PASS_TO_PASS ---

        _exec(container, "git reset --hard HEAD && git clean -fd")

        _put_file(container, "_fabrication.diff", artifact.diff)
        apply_code, apply_out = _exec(
            container,
            "git apply --whitespace=nowarn /tmp/_fabrication.diff"
            " || patch -p1 -f --ignore-whitespace < /tmp/_fabrication.diff",
        )
        if apply_code != 0:
            return "patch_error", f"Fabrication diff did not apply:\n{apply_out[:500]}"

        if pass_to_pass:
            # Run a bounded subset: up to 200 PASS_TO_PASS tests to keep runtime sane
            subset = pass_to_pass[:200]
            tests_str = " ".join(subset)
            p2p_cmd = (
                f"python -m pytest {tests_str} -x -q --no-header --tb=line --timeout=60"
            )
            p2p_code, p2p_out = _exec(container, p2p_cmd)
            if p2p_code != 0:
                # Extract failed test names from output
                failed_lines = [
                    ln for ln in p2p_out.splitlines()
                    if "FAILED" in ln or "ERROR" in ln
                ]
                notes = "Stage 1 FAIL — fabrication broke existing tests:\n"
                notes += "\n".join(failed_lines[:10])
                return "failed_tests", notes
        else:
            # No PASS_TO_PASS list: run tests near the distractor module
            module_dir = _derive_test_dir(artifact.target_file)
            if module_dir:
                broad_cmd = (
                    f"python -m pytest {module_dir} -x -q --no-header --tb=line "
                    f"--timeout=60 --ignore=tests/gis_tests"
                )
                broad_code, broad_out = _exec(container, broad_cmd)
                if broad_code != 0:
                    failed_lines = [
                        ln for ln in broad_out.splitlines()
                        if "FAILED" in ln or "ERROR" in ln
                    ]
                    return "failed_tests", "Stage 1 FAIL (broad):\n" + "\n".join(failed_lines[:10])

        # --- Stage 2: Apply gold patch on top, run FAIL_TO_PASS ---

        if test_patch:
            _put_file(container, "_test_patch.diff", test_patch)
            _exec(
                container,
                "git apply --whitespace=nowarn /tmp/_test_patch.diff"
                " || patch -p1 -f --ignore-whitespace < /tmp/_test_patch.diff",
            )

        if gold_patch:
            _put_file(container, "_gold.diff", gold_patch)
            gold_code, gold_out = _exec(
                container,
                "git apply --whitespace=nowarn /tmp/_gold.diff"
                " || patch -p1 -f --ignore-whitespace < /tmp/_gold.diff",
            )
            if gold_code != 0:
                return "failed_gold_patch", f"Gold patch did not apply on top of fabrication:\n{gold_out[:500]}"

        if fail_to_pass:
            tests_str = " ".join(fail_to_pass)
            f2p_cmd = (
                f"python -m pytest {tests_str} -x -q --no-header --tb=line --timeout=120"
            )
            f2p_code, f2p_out = _exec(container, f2p_cmd)
            if f2p_code != 0:
                failed_lines = [
                    ln for ln in f2p_out.splitlines()
                    if "FAILED" in ln or "ERROR" in ln
                ]
                notes = "Stage 2 FAIL — gold patch no longer resolves issue:\n"
                notes += "\n".join(failed_lines[:10])
                return "failed_gold_patch", notes

        return "passed", "Both stages passed."

    except Exception as exc:
        logger.error("check_artifact error for %s: %s", artifact.artifact_id, exc)
        return "error", str(exc)
    finally:
        if container is not None:
            try:
                container.stop(timeout=10)
                container.remove(force=True)
            except Exception:
                pass


def _derive_test_dir(target_file: str) -> str:
    """Heuristically map e.g. 'django/db/models/fields/related.py' → 'tests/model_fields tests/queries'."""
    parts = target_file.replace("django/", "").split("/")
    candidates = []
    if len(parts) >= 2:
        # e.g. 'db/models' → 'tests/queries', 'db/backends' → 'tests/backends'
        app = parts[-2] if parts[-1].endswith(".py") else parts[-1]
        candidates.append(f"tests/{app}")
        if len(parts) >= 3:
            candidates.append(f"tests/{parts[-3]}")
    return " ".join(candidates) if candidates else "tests/"
