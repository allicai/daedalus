"""Re-validate existing tasks_v2.jsonl with the cross-model validator (DeepSeek V4 Pro).

Run from the repo root:
    python examples/revalidate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path so imports work without editable install
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from daedalus.exporters.jsonl_exporter import load
from daedalus.models.task_schema import DaedalusTask
from daedalus.validators.variant_validator import validate

TASKS_FILE = Path(__file__).parent / "output" / "tasks_v2.jsonl"
RESULTS_FILE = Path(__file__).parent / "output" / "revalidation_results.jsonl"


def main() -> None:
    all_tasks = load(TASKS_FILE, DaedalusTask)
    tasks = [t for t in all_tasks if t.quality_status == "validated"]
    print(f"Loaded {len(tasks)} validated tasks from {TASKS_FILE}")
    print(f"Validator model: DeepSeek-V4-Pro (cross-model, not self-grading)")
    print()

    passed = 0
    failed = 0
    errors = 0
    results = []

    for i, task in enumerate(tasks, 1):
        try:
            rescored = validate(task, llm=True)
            status = rescored.quality_status
            notes = rescored.validation_notes
            results.append({
                "source_instance_id": task.source_instance_id,
                "variant_id": task.variant_id,
                "new_status": status,
                "new_notes": notes,
                "original_notes": task.validation_notes,
            })
            if status == "validated":
                passed += 1
            else:
                failed += 1
            print(f"[{i:3d}/{len(tasks)}] {status.upper():9s} {task.source_instance_id}")
            if status == "rejected":
                print(f"            {notes}")
        except Exception as exc:
            errors += 1
            print(f"[{i:3d}/{len(tasks)}] ERROR     {task.source_instance_id}: {exc}", file=sys.stderr)

    RESULTS_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
        encoding="utf-8",
    )

    total = passed + failed
    print()
    print("=" * 60)
    print(f"Re-validation complete ({len(tasks)} tasks, {errors} errors)")
    print(f"  Pass: {passed}/{total} ({passed/total:.1%})" if total else "  No results")
    print(f"  Fail: {failed}/{total} ({failed/total:.1%})" if total else "")
    print(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
