"""Fabrication candidate generator and manual review tool.

Usage:
    # Generate fabrication candidates for one or more tasks:
    python examples/fabricate_review.py generate django__django-13212
    python examples/fabricate_review.py generate django__django-13212 django__django-13810 django__django-15127

    # Skip Docker safety check (if Docker is unavailable):
    python examples/fabricate_review.py generate django__django-13212 --skip-safety

    # List all candidates and their status (including auto-rejection counts):
    python examples/fabricate_review.py list

    # Show full details for one artifact:
    python examples/fabricate_review.py show fab_django__django-13212_abc12345

    # Record a review decision:
    python examples/fabricate_review.py approve fab_django__django-13212_abc12345
    python examples/fabricate_review.py reject  fab_django__django-13212_abc12345 --notes "too obvious"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from daedalus.exporters.jsonl_exporter import load
from daedalus.fabricator.agent import propose_fabrications
from daedalus.fabricator.context import (
    fetch_distractor_file,
    load_gold_patch,
    load_pass_to_pass,
    parse_intended_distractor,
)
from daedalus.fabricator.safety import check_artifact
from daedalus.fabricator.schemas import (
    FabricatedArtifact,
    FabricationReview,
    RejectedArtifact,
)
from daedalus.models.task_schema import DaedalusTask

TASKS_FILE = Path(__file__).parent / "output" / "tasks_v2.jsonl"
FABRICATIONS_FILE = Path(__file__).parent / "output" / "fabrications.jsonl"
REJECTIONS_FILE = Path(__file__).parent / "output" / "rejected_fabrications.jsonl"
REVIEWS_FILE = Path(__file__).parent / "output" / "fabrication_approvals.jsonl"

VALIDATED_IDS = {
    "django__django-13212",
    "django__django-13810",
    "django__django-15127",
    "django__django-12965",
    "django__django-13346",
    "django__django-14580",
    "django__django-15268",
    "django__django-13028",
    "django__django-14122",
    "django__django-11133",
    "django__django-13964",
    "django__django-14608",
}


def _load_tasks() -> dict[str, DaedalusTask]:
    all_tasks = load(TASKS_FILE, DaedalusTask)
    return {
        t.source_instance_id: t
        for t in all_tasks
        if t.quality_status == "validated" and t.source_instance_id in VALIDATED_IDS
    }


def _load_existing_fabrications() -> dict[str, FabricatedArtifact]:
    if not FABRICATIONS_FILE.exists():
        return {}
    arts: dict[str, FabricatedArtifact] = {}
    with FABRICATIONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    a = FabricatedArtifact.model_validate_json(line)
                    arts[a.artifact_id] = a
                except Exception:
                    pass
    return arts


def _load_rejections() -> list[RejectedArtifact]:
    if not REJECTIONS_FILE.exists():
        return []
    rejs = []
    with REJECTIONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rejs.append(RejectedArtifact.model_validate_json(line))
                except Exception:
                    pass
    return rejs


def _save_artifact(artifact: FabricatedArtifact) -> None:
    FABRICATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FABRICATIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(artifact.model_dump_json() + "\n")


def _save_rejection(rejection: RejectedArtifact) -> None:
    REJECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REJECTIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(rejection.model_dump_json() + "\n")


def _save_review(review: FabricationReview) -> None:
    REVIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REVIEWS_FILE.open("a", encoding="utf-8") as f:
        f.write(review.model_dump_json() + "\n")


def _print_artifact(artifact: FabricatedArtifact, index: int) -> None:
    width = 72
    sep = "─" * width
    print(f"\n{'═' * width}")
    print(f"  Artifact [{index}]  ID: {artifact.artifact_id}")
    print(f"  Class: {artifact.artifact_class}   Safety: {artifact.safety_result}")
    print(f"  File: {artifact.target_file}")
    print(sep)
    print("REASONING:")
    print(artifact.reasoning)
    print(sep)
    print("DIFF:")
    print(artifact.diff[:3000] if len(artifact.diff) > 3000 else artifact.diff)
    if artifact.safety_notes and artifact.safety_notes != "skipped":
        print(sep)
        print("SAFETY NOTES:")
        print(artifact.safety_notes[:800])
    print(f"{'═' * width}")


def _print_rejection(rejection: RejectedArtifact, index: int) -> None:
    print(
        f"  [AUTO-REJECT {index}] stage={rejection.rejection_stage} "
        f"proposed_class={rejection.proposed_class} "
        f"file={rejection.target_file}"
    )
    print(f"    Reason: {rejection.rejection_reason}")


def cmd_generate(instance_ids: list[str], skip_safety: bool = False) -> None:
    tasks = _load_tasks()
    existing = _load_existing_fabrications()
    existing_rejections = {r.source_instance_id for r in _load_rejections()}
    already_done = {a.source_instance_id for a in existing.values()}

    need_ids = [
        iid for iid in instance_ids
        if iid not in already_done
    ]
    if not need_ids:
        print("All requested tasks already have fabrications. Use 'list' to review them.")
        return

    print(f"Loading SWE-bench metadata for {len(need_ids)} tasks...")
    from daedalus.evaluation.swe_runner import load_swebench_metadata
    meta = load_swebench_metadata(set(need_ids))
    p2p_map = load_pass_to_pass(set(need_ids))

    for iid in instance_ids:
        if iid in already_done:
            print(f"\n[SKIP] {iid} — fabrications already generated (use 'list' to see them)")
            continue

        task = tasks.get(iid)
        if task is None:
            print(f"\n[ERROR] {iid} — not found in validated task set", file=sys.stderr)
            continue

        print(f"\n{'=' * 60}")
        print(f"TASK: {iid}")
        raw_distractor = task.transformation_metadata.intended_distractor
        print(f"  Distractor: {raw_distractor[:80]}")

        distractor = parse_intended_distractor(raw_distractor)
        if not distractor.file:
            print(f"  [ERROR] Could not parse distractor file path from: {raw_distractor!r}", file=sys.stderr)
            continue

        print(f"  Fetching {distractor.file} ...")
        file_content = fetch_distractor_file(task.repo, task.base_commit, distractor.file)
        if not file_content:
            print(f"  [ERROR] Could not fetch {distractor.file}", file=sys.stderr)
            continue
        print(f"  Fetched {len(file_content.splitlines())} lines")

        print(f"  Loading gold patch...")
        gold_patch = load_gold_patch(iid)
        if not gold_patch:
            print(f"  [WARNING] No gold patch found for {iid}")

        print(f"  Calling fabricator LLM...")
        try:
            artifacts, auto_rejected = propose_fabrications(task, distractor, file_content, gold_patch)
        except Exception as exc:
            print(f"  [ERROR] Fabricator failed: {exc}", file=sys.stderr)
            continue

        # Report and save auto-rejections
        if auto_rejected:
            print(f"  Auto-rejected {len(auto_rejected)} artifact(s):")
            for i, rej in enumerate(auto_rejected, 1):
                _print_rejection(rej, i)
                _save_rejection(rej)

        if not artifacts:
            print(f"  No valid artifacts passed all gates.")
            continue

        print(f"  {len(artifacts)} artifact(s) passed gates")

        task_meta = meta.get(iid, {})
        fail_to_pass = task_meta.get("fail_to_pass", [])
        test_patch = task_meta.get("test_patch", "")
        pass_to_pass = p2p_map.get(iid, [])

        for i, artifact in enumerate(artifacts, 1):
            if skip_safety:
                artifact = artifact.model_copy(update={
                    "safety_result": "unavailable",
                    "safety_notes": "skipped (--skip-safety)",
                })
            else:
                print(f"  Running safety check for artifact {i}/{len(artifacts)}...")
                result, notes = check_artifact(
                    artifact, gold_patch, pass_to_pass, fail_to_pass, test_patch
                )
                artifact = artifact.model_copy(update={
                    "safety_result": result,
                    "safety_notes": notes,
                })

            _save_artifact(artifact)
            _print_artifact(artifact, i)

    print(f"\nFabrications → {FABRICATIONS_FILE}")
    print(f"Rejections   → {REJECTIONS_FILE}")
    print(f"\nReview with:  python examples/fabricate_review.py list")
    print(f"Approve with: python examples/fabricate_review.py approve <artifact_id>")


def cmd_list() -> None:
    existing = _load_existing_fabrications()
    rejections = _load_rejections()

    reviews: dict[str, str] = {}
    if REVIEWS_FILE.exists():
        with REVIEWS_FILE.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        reviews[r["artifact_id"]] = r["decision"]
                    except Exception:
                        pass

    if existing:
        print(f"\n{'ID':45s}  {'Cls':3s}  {'Safety':18s}  {'Review':10s}  File")
        print("─" * 110)
        for art in sorted(existing.values(), key=lambda a: a.source_instance_id):
            decision = reviews.get(art.artifact_id, "pending")
            print(
                f"{art.artifact_id:45s}  {art.artifact_class:3s}  "
                f"{art.safety_result:18s}  {decision:10s}  {art.target_file}"
            )
    else:
        print("No fabrications found.")

    if rejections:
        print(f"\n{'─' * 60}")
        print(f"AUTO-REJECTED ({len(rejections)} total):")
        stage_counts: dict[str, int] = {}
        for r in rejections:
            stage_counts[r.rejection_stage] = stage_counts.get(r.rejection_stage, 0) + 1
        for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
            print(f"  {stage:25s} {count:3d}")

    total = len(existing)
    if total:
        passed = sum(1 for a in existing.values() if a.safety_result == "passed")
        reviewed = len(reviews)
        approved = sum(1 for v in reviews.values() if v == "approved")
        print(f"\n{total} artifacts | {passed} passed safety | {reviewed} reviewed | {approved} approved")
        print(f"{len(rejections)} auto-rejected (see {REJECTIONS_FILE.name} for details)")


def cmd_show(artifact_id: str) -> None:
    existing = _load_existing_fabrications()
    if artifact_id not in existing:
        print(f"ERROR: {artifact_id!r} not found in {FABRICATIONS_FILE}", file=sys.stderr)
        sys.exit(1)
    _print_artifact(existing[artifact_id], 1)


def cmd_review(artifact_id: str, decision: str, notes: str) -> None:
    existing = _load_existing_fabrications()
    if artifact_id not in existing:
        print(f"ERROR: {artifact_id!r} not found in {FABRICATIONS_FILE}", file=sys.stderr)
        sys.exit(1)
    art = existing[artifact_id]
    review = FabricationReview(
        artifact_id=artifact_id,
        variant_id=art.variant_id,
        source_instance_id=art.source_instance_id,
        decision=decision,  # type: ignore[arg-type]
        notes=notes,
    )
    _save_review(review)
    print(f"Recorded: {decision.upper()} for {artifact_id}")
    if notes:
        print(f"Notes: {notes}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fabricate_review.py",
        description="Generate and review fabrication candidates for Daedalus tasks",
    )
    sub = parser.add_subparsers(dest="cmd")

    gen_p = sub.add_parser("generate", help="Generate fabrication candidates")
    gen_p.add_argument("task", nargs="+", metavar="INSTANCE_ID")
    gen_p.add_argument(
        "--skip-safety", action="store_true", dest="skip_safety",
        help="Skip Docker safety check (useful if Docker is unavailable)",
    )

    sub.add_parser("list", help="List all artifacts and auto-rejection stats")

    show_p = sub.add_parser("show", help="Show full details for one artifact")
    show_p.add_argument("artifact_id")

    approve_p = sub.add_parser("approve", help="Approve an artifact")
    approve_p.add_argument("artifact_id")
    approve_p.add_argument("--notes", default="")

    reject_p = sub.add_parser("reject", help="Reject an artifact")
    reject_p.add_argument("artifact_id")
    reject_p.add_argument("--notes", default="")

    args = parser.parse_args()

    if args.cmd == "generate":
        cmd_generate(args.task, skip_safety=args.skip_safety)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "show":
        cmd_show(args.artifact_id)
    elif args.cmd == "approve":
        cmd_review(args.artifact_id, "approved", args.notes)
    elif args.cmd == "reject":
        cmd_review(args.artifact_id, "rejected", args.notes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
