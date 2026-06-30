#!/usr/bin/env python3
"""Re-label legacy evaluation pairs under the new ownership_label scheme."""
import json
import re
from pathlib import Path

RUNS_PATH  = Path(__file__).parent / "output" / "eval_runs_v0_legacy.jsonl"
PAIRS_PATH = Path(__file__).parent / "output" / "eval_pairs_v0_legacy.jsonl"

GOLD_FILES = {
    "django__django-16263": ["django/db/models/sql/query.py"],
    "django__django-15695": ["django/db/migrations/operations/models.py"],
    "django__django-11490": ["django/db/models/sql/compiler.py"],
    "django__django-10999": ["django/utils/dateparse.py"],
    "django__django-13109": ["django/db/models/fields/related.py"],
}


def _extract_ownership_file(ownership_assignment: str) -> str:
    if not ownership_assignment:
        return ""
    m = re.search(r"((?:\w+/)+\w+\.py)", ownership_assignment)
    return m.group(1) if m else ""


def _files_match(a: str, b: str) -> bool:
    a = a.replace("\\", "/")
    b = b.replace("\\", "/")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _classify(ownership_file: str, gold_files: list, distractor_file: str) -> str:
    if not ownership_file:
        return "unknown"
    if gold_files and any(_files_match(ownership_file, gf) for gf in gold_files):
        return "correct"
    if distractor_file and _files_match(ownership_file, distractor_file):
        return "distractor_adopted"
    return "other"


def _old_bucket_from_mentions(pair: dict) -> str:
    """Reconstruct legacy bucket from mentions_distractor booleans."""
    o = pair.get("original_mentions_distractor", False)
    v = pair.get("variant_mentions_distractor", False)
    var_visited = pair.get("variant_distractor_visited", False)
    if not var_visited and not v:
        return "bad_variant"
    if not o and v:
        return "clean_shift"
    if o and v:
        return "contaminated"
    if o and not v:
        return "adoption_reduced"
    return "no_effect"


def _new_bucket(orig_lbl: str, var_lbl: str, var_visited: bool) -> str:
    if not var_visited and var_lbl != "distractor_adopted":
        return "bad_variant"
    if orig_lbl != "distractor_adopted" and var_lbl == "distractor_adopted":
        return "clean_shift"
    if orig_lbl == "distractor_adopted" and var_lbl == "distractor_adopted":
        return "contaminated"
    if orig_lbl == "distractor_adopted" and var_lbl != "distractor_adopted":
        return "adoption_reduced"
    return "no_effect"


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    runs  = load_jsonl(RUNS_PATH)
    pairs = load_jsonl(PAIRS_PATH)
    run_by_id = {r["run_id"]: r for r in runs}

    print(f"Runs: {len(runs)}, Pairs: {len(pairs)}")

    print()
    print("=== RUN-LEVEL RE-LABELING ===")
    print()
    hdr = f"{'Variant ID':<52} {'Cond':<9} {'Old(mentions)':<14} {'OwnFile':<40} {'New label'}"
    print(hdr)
    print("-" * len(hdr))

    for pair in pairs:
        instance_id = pair["source_instance_id"]
        gold_files  = GOLD_FILES.get(instance_id, [])

        for cond, key in [("original", "original_run_id"), ("variant", "variant_run_id")]:
            run_id = pair[key]
            run = run_by_id.get(run_id)
            if not run:
                print(f"  MISSING run: {run_id}")
                continue

            distractor_file = run.get("distractor_file", "")
            old_mentions    = run.get("mentions_distractor", False)
            own_file        = _extract_ownership_file(run.get("ownership_assignment", ""))
            new_label       = _classify(own_file, gold_files, distractor_file)

            print(
                f"{run['variant_id']:<52} {cond:<9} {str(old_mentions):<14}"
                f" {own_file:<40} {new_label}"
            )

    print()
    print("=== PAIR-LEVEL RE-BUCKETING ===")
    print()
    hdr2 = f"{'Pair ID':<52} {'Old bucket':<22} {'Orig label':<22} {'Var label':<22} New bucket"
    print(hdr2)
    print("-" * len(hdr2))

    changed = 0
    for pair in pairs:
        instance_id = pair["source_instance_id"]
        gold_files  = GOLD_FILES.get(instance_id, [])

        orig_run = run_by_id.get(pair["original_run_id"])
        var_run  = run_by_id.get(pair["variant_run_id"])
        if not orig_run or not var_run:
            continue

        distractor_file = orig_run.get("distractor_file", "")
        orig_of  = _extract_ownership_file(orig_run.get("ownership_assignment", ""))
        var_of   = _extract_ownership_file(var_run.get("ownership_assignment", ""))
        orig_lbl = _classify(orig_of, gold_files, distractor_file)
        var_lbl  = _classify(var_of,  gold_files, distractor_file)

        bucket = _new_bucket(orig_lbl, var_lbl, var_run.get("distractor_file_visited", False))

        # Legacy pairs lack evaluation_bucket; reconstruct from mentions_distractor
        old_bucket = pair.get("evaluation_bucket") or _old_bucket_from_mentions(pair)
        marker = "  <-- CHANGED" if bucket != old_bucket else ""
        if bucket != old_bucket:
            changed += 1
        print(
            f"{pair['pair_id']:<52} {old_bucket:<22} {orig_lbl:<22} {var_lbl:<22} {bucket}{marker}"
        )

    print()
    print(f"Pairs re-bucketed: {changed} of {len(pairs)}")


if __name__ == "__main__":
    main()
