#!/usr/bin/env python3
"""Generate a root_cause_attribution variant from a sample SWE-Bench task."""
from pathlib import Path

from daedalus.exporters.jsonl_exporter import export
from daedalus.loaders.swebench_loader import load_task
from daedalus.transforms.root_cause_attribution import transform

SAMPLE_TASK = {
    "instance_id": "django__django-11133",
    "repo": "django/django",
    "base_commit": "879cc3da6249e920b8d54518a0ae06de835d7373",
    "problem_statement": (
        "HttpResponse doesn't handle memoryview objects\n\n"
        "Description\n\n"
        "When a memoryview is passed as content to HttpResponse, the Content-Length "
        "header is set incorrectly. len() on a memoryview returns the number of "
        "elements, not bytes, when itemsize > 1. The fix should ensure bytes are "
        "counted correctly regardless of the memoryview's underlying type."
    ),
    "hints_text": "",
    "patch": "",
    "test_patch": "",
}

OUTPUT_PATH = Path(__file__).parent / "output" / "tasks.jsonl"


def main() -> None:
    task = load_task(SAMPLE_TASK)
    print(f"Loaded:   {task.instance_id}")

    print("Generating root_cause_attribution variant...")
    variant = transform(task, index=1)

    print(f"Variant:  {variant.variant_id}")
    print(f"Scope:    {variant.transformation_scope}")
    print(f"Target:   {variant.target_failure_mode}")
    print(f"Status:   {variant.quality_status}")
    print(f"\nDistractor: {variant.transformation_metadata.intended_distractor}")
    print(f"\nOriginal ({len(variant.original_problem_statement)} chars):")
    print(f"  {variant.original_problem_statement[:120]}...")
    print(f"\nModified ({len(variant.modified_problem_statement)} chars):")
    print(f"  {variant.modified_problem_statement[:120]}...")

    export([variant], OUTPUT_PATH)
    print(f"\nExported → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
