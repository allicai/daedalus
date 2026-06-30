from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SWEBenchTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    patch: str = ""        # gold fix — preserved internally, never sent to the generator
    test_patch: str = ""   # failing test — preserved internally, never sent to the generator


def load_task(source: dict | str | Path) -> SWEBenchTask:
    if isinstance(source, dict):
        data = source
    else:
        with Path(source).open() as f:
            data = json.load(f)

    return SWEBenchTask(
        instance_id=data["instance_id"],
        repo=data["repo"],
        base_commit=data["base_commit"],
        problem_statement=data["problem_statement"],
        hints_text=data.get("hints_text", ""),
        patch=data.get("patch", ""),
        test_patch=data.get("test_patch", ""),
    )


def load_tasks_from_jsonl(path: str | Path) -> list[SWEBenchTask]:
    tasks = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(load_task(json.loads(line)))
    return tasks


def load_from_hf(
    limit: int | None = None,
    repo: str | None = None,
    seed: int | None = None,
    exclude_instance_ids: set[str] | None = None,
) -> list[SWEBenchTask]:
    # patch and test_patch are loaded but must never be forwarded to the generator —
    # distractors must be derived from issue context, not reverse-engineered from the fix.
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required. Run: pip install datasets"
        )

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    if repo:
        ds = ds.filter(lambda x: x["repo"] == repo)

    if exclude_instance_ids:
        ds = ds.filter(lambda x: x["instance_id"] not in exclude_instance_ids)

    if seed is not None:
        ds = ds.shuffle(seed=seed)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    tasks = []
    for record in ds:
        tasks.append(SWEBenchTask(
            instance_id=record["instance_id"],
            repo=record["repo"],
            base_commit=record["base_commit"],
            problem_statement=record["problem_statement"],
            hints_text=record.get("hints_text") or "",
            patch=record.get("patch") or "",
            test_patch=record.get("test_patch") or "",
        ))

    return tasks


def load_patches_by_id(instance_ids: set[str]) -> dict[str, str]:
    """Return {instance_id: gold_patch} for the requested ids from SWE-bench Verified."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return {}

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {
        r["instance_id"]: r.get("patch") or ""
        for r in ds
        if r["instance_id"] in instance_ids
    }
