#!/usr/bin/env python3
"""Regenerate 3 tasks with the new natural-framing prompt and show before/after."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daedalus.loaders.swebench_loader import SWEBenchTask
from daedalus.transforms.root_cause_attribution import transform

VALIDATED = Path(__file__).parent / "output" / "tasks_validated.jsonl"

TARGETS = [
    "django__django-11099",  # TRANSFORMER / parser-normalizer archetype
    "django__django-10999",  # PRODUCER / producer-consumer archetype
    "django__django-12858",  # VALIDATOR / validator-caller archetype
]

ARCHETYPE_LABEL = {
    "django__django-11099": "TRANSFORMER (parser-normalizer)",
    "django__django-10999": "PRODUCER (producer-consumer)",
    "django__django-12858": "VALIDATOR (validator-caller)",
}


def load_validated() -> dict[str, dict]:
    tasks = {}
    with VALIDATED.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                tasks[t["source_instance_id"]] = t
    return tasks


def sep(title: str, width: int = 72) -> None:
    print()
    print("━" * width)
    print(f"  {title}")
    print("━" * width)


def show_appended(label: str, orig: str, mod: str) -> None:
    if mod.startswith(orig):
        appended = mod[len(orig):].lstrip("\n")
    else:
        appended = "(modified in-place — diff not computable)"
    print(f"\n  [{label}]")
    for line in appended.splitlines():
        print(f"  {line}")


def main() -> None:
    validated = load_validated()

    for instance_id in TARGETS:
        old_task = validated.get(instance_id)
        if not old_task:
            print(f"  MISSING: {instance_id}")
            continue

        archetype = ARCHETYPE_LABEL[instance_id]
        sep(f"{instance_id}  —  {archetype}")

        orig = old_task["original_problem_statement"]
        old_mod = old_task["modified_problem_statement"]
        old_distractor = old_task["transformation_metadata"]["intended_distractor"]

        print(f"\n  Distractor (old): {old_distractor[:100]}...")
        show_appended("BEFORE", orig, old_mod)

        # Reconstruct a minimal SWEBenchTask (no patch → ungrounded path)
        swe_task = SWEBenchTask(
            instance_id=instance_id,
            repo=old_task["repo"],
            base_commit=old_task["base_commit"],
            problem_statement=orig,
            hints_text="",
            patch="",
            test_patch="",
        )

        print("\n  Generating new variant...")
        try:
            new_variant = transform(swe_task, index=2)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        new_mod = new_variant.modified_problem_statement
        new_distractor = new_variant.transformation_metadata.intended_distractor

        print(f"\n  Distractor (new): {new_distractor[:100]}...")

        # Detect what changed
        if new_mod.startswith(orig):
            appended_new = new_mod[len(orig):].lstrip("\n")
            print(f"\n  [AFTER — appended {len(appended_new)} chars]")
            for line in appended_new.splitlines():
                print(f"  {line}")
        else:
            # Find first divergence point
            for i, (a, b) in enumerate(zip(orig, new_mod)):
                if a != b:
                    ctx_start = max(0, i - 40)
                    print(f"\n  [AFTER — original was modified in-place at char {i}]")
                    print(f"  ORIGINAL[{ctx_start}:{i+80}] = {repr(orig[ctx_start:i+80])}")
                    print(f"  MODIFIED[{ctx_start}:{i+80}] = {repr(new_mod[ctx_start:i+80])}")
                    break
            # Still show the tail
            appended_new = new_mod
            tail = new_mod[len(orig):] if len(new_mod) > len(orig) else ""
            if tail:
                print(f"\n  [Tail appended]")
                for line in tail.lstrip("\n").splitlines():
                    print(f"  {line}")

        # Flag any forbidden patterns in anything beyond the original
        appended_new = new_mod[len(orig):] if new_mod.startswith(orig) else new_mod
        forbidden = ["Competing Hypothesis", "Alternative Explanation",
                     "hypothesis", "## ", "**Competing", "Hypothesis:"]
        violations = [p for p in forbidden if p.lower() in appended_new.lower()]
        if violations:
            print(f"\n  ⚠  VIOLATIONS FOUND: {violations}")
        else:
            print("\n  ✓  No forbidden patterns detected")


if __name__ == "__main__":
    main()
