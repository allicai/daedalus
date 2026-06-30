"""Generate 10 tasks and print the framing starter used in each.

Run from the repo root:
    python examples/sample_framing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from daedalus.loaders.swebench_loader import load_from_hf
from daedalus.transforms.root_cause_attribution import transform as rca_transform
from daedalus.validators.variant_validator import validate

SAMPLE_SIZE = 10
SEED = 99  # different seed so we get fresh tasks


def extract_appended(original: str, modified: str) -> str:
    """Return the text appended after the original."""
    if modified.startswith(original):
        return modified[len(original):].strip()
    return modified[len(original):].strip() if len(modified) > len(original) else "(unchanged)"


def main() -> None:
    print(f"Loading {SAMPLE_SIZE} django tasks from HuggingFace...")
    tasks = load_from_hf(limit=SAMPLE_SIZE, repo="django/django", seed=SEED)
    print(f"Got {len(tasks)} tasks\n")

    for i, task in enumerate(tasks, 1):
        try:
            variant = rca_transform(task, index=1)
            scored = validate(variant, llm=False)  # heuristic only — fast
            appended = extract_appended(task.problem_statement, variant.modified_problem_statement)
            first_words = appended[:80].replace("\n", " ")
            print(f"[{i:2d}] {task.instance_id}")
            print(f"     Appended: {first_words!r}")
            print(f"     Status: {scored.quality_status} | {scored.validation_notes[:60]}")
            print()
        except Exception as exc:
            print(f"[{i:2d}] ERROR {task.instance_id}: {exc}\n")


if __name__ == "__main__":
    main()
