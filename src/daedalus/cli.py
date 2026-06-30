from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_generate(args: argparse.Namespace) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm is required. Run: pip install tqdm", file=sys.stderr)
        sys.exit(1)

    from daedalus.exporters.jsonl_exporter import export, load_source_ids
    from daedalus.loaders.swebench_loader import load_from_hf, load_tasks_from_jsonl
    from daedalus.transforms.root_cause_attribution import transform as rca_transform
    from daedalus.validators.variant_validator import validate

    TRANSFORMS = {
        "root_cause_attribution": rca_transform,
    }
    transform_fn = TRANSFORMS[args.variant_type]

    output_path = Path(args.output)

    done_ids = load_source_ids(output_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} source instances already in {output_path}")

    if args.dataset == "hf":
        print("Loading from HuggingFace (princeton-nlp/SWE-bench_Verified)...")
        tasks = load_from_hf(
            limit=args.limit,
            repo=args.repo or None,
            seed=args.seed,
            exclude_instance_ids=done_ids,
        )
    else:
        tasks = load_tasks_from_jsonl(args.dataset)
        if args.repo:
            tasks = [t for t in tasks if t.repo == args.repo]
        tasks = [t for t in tasks if t.instance_id not in done_ids]
        if args.limit:
            tasks = tasks[: args.limit]

    if not tasks:
        print("No new tasks to process.")
        return

    print(f"Processing {len(tasks)} tasks → {output_path}")
    if args.repo_dir:
        print(f"Note: --repo-dir is accepted but generation uses GitHub API for now.")

    generated = 0
    validated = 0
    errors = 0

    pbar = tqdm(tasks, desc="Generating", unit="task", dynamic_ncols=True)
    for task in pbar:
        try:
            variant = transform_fn(task, index=1)
            scored = validate(variant)
            export([scored], output_path, append=True)
            generated += 1
            if scored.quality_status == "validated":
                validated += 1
        except Exception as exc:
            errors += 1
            tqdm.write(f"  ERROR {task.instance_id}: {exc}", file=sys.stderr)

        if generated:
            pbar.set_postfix(
                {"yield": f"{validated / generated:.1%}", "err": errors},
                refresh=False,
            )

    if generated:
        print(
            f"\nDone — {generated} generated, {validated} validated "
            f"({validated / generated:.1%} yield), {errors} errors"
        )
    else:
        print(f"\nDone — 0 generated, {errors} errors")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm is required. Run: pip install tqdm", file=sys.stderr)
        sys.exit(1)

    from daedalus.evaluation.runner import compare_pair, run_condition
    from daedalus.exporters.jsonl_exporter import export, load, load_source_ids
    from daedalus.loaders.swebench_loader import load_patches_by_id
    from daedalus.models.task_schema import DaedalusTask

    tasks_path = Path(args.tasks)
    runs_path = Path(args.output)
    pairs_path = runs_path.parent / (runs_path.stem + "_pairs" + runs_path.suffix)
    repo_path = Path(args.repo_dir)

    if not repo_path.is_dir():
        print(f"ERROR: --repo-dir not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    all_tasks = load(tasks_path, DaedalusTask)
    tasks = [t for t in all_tasks if t.quality_status == "validated"]

    done_ids = load_source_ids(pairs_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} pairs already in {pairs_path}")
    tasks = [t for t in tasks if t.source_instance_id not in done_ids]

    if args.limit:
        tasks = tasks[: args.limit]

    if not tasks:
        print("No new tasks to evaluate.")
        return

    print(f"Loading gold patches for {len(tasks)} tasks...")
    gold_patches = load_patches_by_id({t.source_instance_id for t in tasks})
    missing = len(tasks) - len(gold_patches)
    if missing:
        print(f"  Warning: {missing} patches not found — ownership_label will be 'unknown' for those tasks")

    print(f"Evaluating {len(tasks)} pairs (2 runs each) → {runs_path}")
    print(f"  Pairs  → {pairs_path}")
    print(f"  Repo   → {repo_path}")

    bucket_counts: dict[str, int] = {}
    evaluated = 0
    errors = 0

    pbar = tqdm(tasks, desc="Evaluating", unit="pair", dynamic_ncols=True)
    for task in pbar:
        task_dict = task.model_dump()
        gold_patch = gold_patches.get(task.source_instance_id, "")
        try:
            original_run = run_condition(task_dict, "original", repo_path, gold_patch=gold_patch, model=args.model or None, max_turns=args.max_turns)
            variant_run = run_condition(task_dict, "variant", repo_path, gold_patch=gold_patch, model=args.model or None, max_turns=args.max_turns)
            pair = compare_pair(original_run, variant_run)

            export([original_run, variant_run], runs_path, append=True)
            export([pair], pairs_path, append=True)

            evaluated += 1
            bucket_counts[pair.evaluation_bucket] = bucket_counts.get(pair.evaluation_bucket, 0) + 1
        except Exception as exc:
            errors += 1
            tqdm.write(f"  ERROR {task.source_instance_id}: {exc}", file=sys.stderr)

        if evaluated:
            pbar.set_postfix(
                {b: n for b, n in bucket_counts.items()} | {"err": errors},
                refresh=False,
            )

    print(f"\nDone — {evaluated} pairs evaluated, {errors} errors")
    if bucket_counts:
        total = sum(bucket_counts.values())
        print("\nBucket distribution:")
        for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
            print(f"  {bucket:20s} {count:3d}  ({count / total:.0%})")


def _cmd_export_instances(args: argparse.Namespace) -> None:
    import json as _json

    from daedalus.exporters.jsonl_exporter import load
    from daedalus.models.task_schema import DaedalusTask

    tasks_path = Path(args.tasks)
    output_dir = Path(args.output)

    all_tasks = load(tasks_path, DaedalusTask)
    tasks = [t for t in all_tasks if t.quality_status == "validated"]

    if args.limit:
        tasks = tasks[: args.limit]

    if not tasks:
        print("No validated tasks found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for condition in ("original", "variant"):
        key = f"{condition}_problem_statement"
        instances = [
            {"instance_id": t.source_instance_id, "problem_statement": getattr(t, key)}
            for t in tasks
        ]
        out_path = output_dir / f"{condition}.json"
        out_path.write_text(_json.dumps(instances, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Exported {len(tasks)} instances ({len(tasks)} original + {len(tasks)} variant) to {output_dir}")


def _cmd_evaluate_swe(args: argparse.Namespace) -> None:
    from daedalus.evaluation.swe_runner import (
        build_swe_pairs,
        load_swebench_metadata,
        run_swe_condition,
    )
    from daedalus.exporters.jsonl_exporter import export, load, load_source_ids
    from daedalus.models.task_schema import DaedalusTask

    tasks_path = Path(args.tasks)
    output_dir = Path(args.output)
    runs_path = output_dir / "swe_runs.jsonl"
    pairs_path = output_dir / "swe_pairs.jsonl"

    all_tasks = load(tasks_path, DaedalusTask)
    tasks = [t for t in all_tasks if t.quality_status == "validated"]

    done_ids = load_source_ids(pairs_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} pairs already in {pairs_path}")
    tasks = [t for t in tasks if t.source_instance_id not in done_ids]

    if args.limit:
        tasks = tasks[: args.limit]

    if not tasks:
        print("No new tasks to evaluate.")
        return

    print(f"Loading SWE-bench metadata for {len(tasks)} tasks...")
    metadata = load_swebench_metadata({t.source_instance_id for t in tasks})
    missing = len(tasks) - len(metadata)
    if missing:
        print(f"  Warning: {missing} task(s) not found in SWE-bench metadata")

    task_dicts = [t.model_dump() for t in tasks]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running mini-SWE-agent (original condition) → {output_dir}/original/")
    orig_preds_path = run_swe_condition(
        task_dicts, "original", output_dir, args.model, workers=args.workers
    )

    print(f"Running mini-SWE-agent (variant condition) → {output_dir}/variant/")
    var_preds_path = run_swe_condition(
        task_dicts, "variant", output_dir, args.model, workers=args.workers
    )

    print("Evaluating patches and building pairs...")
    runs, pairs = build_swe_pairs(task_dicts, orig_preds_path, var_preds_path, metadata)

    export(runs, runs_path, append=True)
    export(pairs, pairs_path, append=True)

    bucket_counts: dict[str, int] = {}
    for pair in pairs:
        bucket_counts[pair.evaluation_bucket] = bucket_counts.get(pair.evaluation_bucket, 0) + 1

    print(f"\nDone — {len(pairs)} pairs evaluated")
    print(f"  Runs  → {runs_path}")
    print(f"  Pairs → {pairs_path}")
    if bucket_counts:
        total = sum(bucket_counts.values())
        print("\nBucket distribution:")
        for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
            print(f"  {bucket:20s} {count:3d}  ({count / total:.0%})")


def _cmd_report(args: argparse.Namespace) -> None:
    from collections import Counter

    from daedalus.evaluation.schemas import EvaluationPair, EvaluationRun
    from daedalus.exporters.jsonl_exporter import load

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
        sys.exit(1)

    pairs = load(pairs_path, EvaluationPair)
    if not pairs:
        print("No pairs found.")
        return

    # Auto-derive runs path: swe_pairs → swe_runs, foo_pairs → foo_runs, foo_pairs → foo
    runs_by_id: dict[str, EvaluationRun] = {}
    runs_path = Path(args.runs) if getattr(args, "runs", None) else None
    if runs_path is None:
        name = pairs_path.name
        for candidate_name in (
            name.replace("_pairs.", "_runs."),
            name.replace("_pairs.", "."),
        ):
            candidate = pairs_path.parent / candidate_name
            if candidate.exists() and candidate != pairs_path:
                runs_path = candidate
                break
    if runs_path and runs_path.exists():
        for run in load(runs_path, EvaluationRun):
            runs_by_id[run.run_id] = run

    total = len(pairs)
    bucket_counts: Counter = Counter(p.evaluation_bucket for p in pairs)

    W = 60
    print("=" * W)
    print("  Daedalus Evaluation Report")
    print(f"  {pairs_path.name}  ({total} pairs)")
    print("=" * W)

    print("\n  Bucket distribution")
    print("  " + "─" * (W - 2))
    for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        pct = count / total * 100
        print(f"  {bucket:<22} {count:3d}  ({pct:.0f}%)  {bar}")

    shifted  = sum(1 for p in pairs if p.hypothesis_shifted)
    diverged = sum(1 for p in pairs if p.trajectory_diverged)
    print()
    print(f"  hypothesis_shifted:   {shifted}/{total}")
    print(f"  trajectory_diverged:  {diverged}/{total}")

    if runs_by_id:
        resolved_orig = sum(
            1 for p in pairs
            if runs_by_id.get(p.original_run_id) and runs_by_id[p.original_run_id].resolved is True
        )
        resolved_var = sum(
            1 for p in pairs
            if runs_by_id.get(p.variant_run_id) and runs_by_id[p.variant_run_id].resolved is True
        )
        evaluated = sum(1 for p in pairs if p.original_resolved is not None)
        if evaluated:
            print(f"  resolved (original):  {resolved_orig}/{evaluated}")
            print(f"  resolved (variant):   {resolved_var}/{evaluated}")

    print(f"\n  Per-task")
    print("  " + "─" * (W - 2))
    abbrev = {"correct": "correct", "distractor_adopted": "distract", "other": "other", "unknown": "?"}
    print(f"  {'Instance':32s}  {'Bucket':20s}  {'Orig→Var':18s}  {'Dist':4s}  {'ΔT':>3s}")
    for p in sorted(pairs, key=lambda x: x.evaluation_bucket):
        iid = p.source_instance_id
        if len(iid) > 32:
            iid = "…" + iid[-31:]
        ol = abbrev.get(p.original_ownership_label, p.original_ownership_label[:8])
        vl = abbrev.get(p.variant_ownership_label, p.variant_ownership_label[:8])
        dist = f"{'Y' if p.original_distractor_visited else 'N'}/{'Y' if p.variant_distractor_visited else 'N'}"
        dt = f"{p.turns_delta:+d}" if p.turns_delta != 0 else "0"
        print(f"  {iid:32s}  {p.evaluation_bucket:20s}  {ol:8s}→{vl:8s}  {dist:4s}  {dt:>3s}")

    clean_pairs = [p for p in pairs if p.hypothesis_shifted]
    if clean_pairs:
        print(f"\n  {'═' * (W - 2)}")
        print(f"  CLEAN SHIFT CASES ({len(clean_pairs)})")
        print(f"  {'═' * (W - 2)}")
        for p in clean_pairs:
            print(f"\n  {p.source_instance_id}")
            for condition, run_id in (("original", p.original_run_id), ("variant", p.variant_run_id)):
                run = runs_by_id.get(run_id)
                if not run:
                    continue
                dist_note = " [distractor visited]" if run.distractor_file_visited else ""
                print(f"\n    [{condition.upper()}]  {run.turns} turns, {run.unique_files_opened} files{dist_note}")
                if run.ownership_assignment:
                    print(f"    Ownership: {run.ownership_assignment}")
                if run.reasoning:
                    text = run.reasoning.replace("\n", " ")
                    print(f"    Reasoning: {text[:280]}")
            print(f"\n  {'─' * (W - 2)}")


def _cmd_evaluate_fabricated(args: argparse.Namespace) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm is required. Run: pip install tqdm", file=sys.stderr)
        sys.exit(1)

    import json as _json

    from daedalus.evaluation.fabricated_runner import run_fabricated_pair
    from daedalus.exporters.jsonl_exporter import export, load, load_source_ids
    from daedalus.fabricator.schemas import FabricatedArtifact
    from daedalus.loaders.swebench_loader import load_patches_by_id
    from daedalus.models.task_schema import DaedalusTask

    tasks_path = Path(args.tasks)
    fab_path = Path(args.fabrications)
    repo_path = Path(args.repo_dir)
    output_dir = Path(args.output)
    runs_path = output_dir / "fab_runs.jsonl"
    pairs_path = output_dir / "fab_pairs.jsonl"

    if not repo_path.is_dir():
        print(f"ERROR: --repo-dir not found: {repo_path}", file=sys.stderr)
        sys.exit(1)
    if not fab_path.exists():
        print(f"ERROR: fabrications file not found: {fab_path}", file=sys.stderr)
        sys.exit(1)

    # Load tasks
    all_tasks = load(tasks_path, DaedalusTask)
    tasks_by_id = {
        t.source_instance_id: t.model_dump()
        for t in all_tasks
        if t.quality_status == "validated"
    }

    # Load fabrications
    artifacts: dict[str, FabricatedArtifact] = {}
    with fab_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    a = FabricatedArtifact.model_validate_json(line)
                    artifacts[a.artifact_id] = a
                except Exception:
                    pass

    # Load review decisions (from separate approvals file or --reviews arg)
    reviews_path = Path(args.reviews) if getattr(args, "reviews", None) else None
    if reviews_path is None:
        reviews_path = fab_path.parent / "fabrication_approvals.jsonl"
    reviews: dict[str, str] = {}
    if reviews_path.exists():
        with reviews_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = _json.loads(line)
                        reviews[r["artifact_id"]] = r["decision"]
                    except Exception:
                        pass

    # Filter to approved artifacts with a matching task
    approved = [
        a for a in artifacts.values()
        if reviews.get(a.artifact_id, a.review_decision) == "approved"
        and a.source_instance_id in tasks_by_id
        and a.safety_result not in ("failed_tests", "failed_gold_patch", "patch_error")
    ]

    if not approved:
        print("No approved artifacts with matching validated tasks found.")
        return

    # Resume support
    done_ids = load_source_ids(pairs_path)
    approved = [a for a in approved if a.artifact_id not in done_ids]
    if done_ids:
        print(f"Resuming: {len(done_ids)} artifact(s) already evaluated")

    if args.limit:
        approved = approved[: args.limit]

    if not approved:
        print("No new artifacts to evaluate.")
        return

    print(f"Loading gold patches for {len({a.source_instance_id for a in approved})} tasks...")
    gold_patches = load_patches_by_id({a.source_instance_id for a in approved})

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(approved)} fabricated artifact(s) → {output_dir}")

    bucket_counts: dict[str, int] = {}
    evaluated = 0
    errors = 0

    pbar = tqdm(approved, desc="Evaluating", unit="artifact", dynamic_ncols=True)
    for artifact in pbar:
        task = tasks_by_id[artifact.source_instance_id]
        gold_patch = gold_patches.get(artifact.source_instance_id, "")
        try:
            orig_run, var_run, pair = run_fabricated_pair(
                task, artifact, repo_path,
                gold_patch=gold_patch,
                model=args.model or None,
                max_turns=args.max_turns or None,
            )
            export([orig_run, var_run], runs_path, append=True)
            export([pair], pairs_path, append=True)

            evaluated += 1
            bucket_counts[pair.evaluation_bucket] = bucket_counts.get(pair.evaluation_bucket, 0) + 1
        except Exception as exc:
            errors += 1
            tqdm.write(f"  ERROR {artifact.artifact_id}: {exc}", file=sys.stderr)

        if evaluated:
            pbar.set_postfix(
                {b: n for b, n in bucket_counts.items()} | {"err": errors},
                refresh=False,
            )

    print(f"\nDone — {evaluated} pairs evaluated, {errors} errors")
    print(f"  Runs  → {runs_path}")
    print(f"  Pairs → {pairs_path}")
    if bucket_counts:
        total = sum(bucket_counts.values())
        print("\nBucket distribution:")
        for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
            print(f"  {bucket:20s} {count:3d}  ({count / total:.0%})")
    print(f"\nView results:  daedalus report --pairs {pairs_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="daedalus",
        description="Daedalus — coding-agent eval augmentation framework",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── generate ────────────────────────────────────────────────────────────
    gen = subparsers.add_parser(
        "generate",
        help="Batch generate task variants from SWE-bench",
    )
    gen.add_argument(
        "--dataset",
        default="hf",
        help="'hf' to load from HuggingFace, or path to a local JSONL file (default: hf)",
    )
    gen.add_argument(
        "--output",
        required=True,
        help="Output JSONL path. Appends to existing file for resume support.",
    )
    gen.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Max number of tasks to process",
    )
    gen.add_argument(
        "--variant-type",
        default="root_cause_attribution",
        choices=["root_cause_attribution"],
        dest="variant_type",
        help="Variant type to generate (default: root_cause_attribution)",
    )
    gen.add_argument(
        "--repo",
        default=None,
        metavar="OWNER/NAME",
        help="Filter to a specific repo, e.g. 'django/django'",
    )
    gen.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for HuggingFace dataset shuffle (default: 42)",
    )
    gen.add_argument(
        "--repo-dir",
        default=None,
        dest="repo_dir",
        metavar="DIR",
        help="Directory of locally checked-out repos (reserved for future local-mode generation)",
    )

    # ── evaluate ─────────────────────────────────────────────────────────────
    ev = subparsers.add_parser(
        "evaluate",
        help="Run paired agent evaluation on generated variants (stub — coming soon)",
    )
    ev.add_argument("--tasks", required=True, help="Path to DaedalusTask JSONL (output of daedalus generate)")
    ev.add_argument("--output", required=True, help="Output JSONL path for evaluation runs (pairs written to <stem>_pairs<ext>)")
    ev.add_argument("--repo-dir", required=True, dest="repo_dir", metavar="DIR",
                    help="Path to locally checked-out repository matching the tasks file")
    ev.add_argument(
        "--limit", type=int, default=None, metavar="N", help="Max pairs to evaluate"
    )
    ev.add_argument(
        "--max-turns", type=int, default=None, metavar="N", dest="max_turns",
        help=f"Max agent turns per run (default: {__import__('daedalus.evaluation.runner', fromlist=['MAX_TURNS']).MAX_TURNS})",
    )
    ev.add_argument("--model", default=None, help="Model for the read-only agent (e.g. deepseek-ai/DeepSeek-V4-Pro, anthropic/claude-sonnet-4-6). Any Together AI or OpenRouter model string works. Defaults to the LLM_PROVIDER-configured model.")

    # ── export-instances ──────────────────────────────────────────────────────
    ei = subparsers.add_parser(
        "export-instances",
        help="Export task instances as JSON for mini-SWE-agent input",
    )
    ei.add_argument("--tasks", required=True, help="Path to DaedalusTask JSONL")
    ei.add_argument("--output", required=True, help="Directory to write original.json and variant.json")
    ei.add_argument("--limit", type=int, default=None, metavar="N", help="Max instances to export")

    # ── evaluate-swe ─────────────────────────────────────────────────────────
    eswe = subparsers.add_parser(
        "evaluate-swe",
        help="Run mini-SWE-agent + Docker patch evaluation on generated variants",
    )
    eswe.add_argument("--tasks", required=True, help="Path to DaedalusTask JSONL")
    eswe.add_argument("--output", required=True, help="Directory for outputs (preds, runs, pairs)")
    eswe.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V4-Pro",
        help="Model for mini-SWE-agent (default: deepseek-ai/DeepSeek-V4-Pro). Any Together AI or OpenRouter model string works.",
    )
    eswe.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for mini-SWE-agent (default: 1)",
    )
    eswe.add_argument("--limit", type=int, default=None, metavar="N", help="Max pairs to evaluate")

    # ── evaluate-fabricated ───────────────────────────────────────────────────
    efab = subparsers.add_parser(
        "evaluate-fabricated",
        help="Run read-only agent evaluation with fabricated repo artifacts applied",
    )
    efab.add_argument("--tasks", required=True, help="Path to DaedalusTask JSONL")
    efab.add_argument(
        "--fabrications", required=True, metavar="FILE",
        help="Path to fabrications.jsonl (output of fabricate_review.py generate)",
    )
    efab.add_argument(
        "--reviews", default=None, metavar="FILE",
        help="Path to fabrication_approvals.jsonl (default: adjacent to --fabrications)",
    )
    efab.add_argument("--output", required=True, help="Directory for fab_runs.jsonl and fab_pairs.jsonl")
    efab.add_argument("--repo-dir", required=True, dest="repo_dir", metavar="DIR",
                      help="Path to locally checked-out repository")
    efab.add_argument("--model", default=None,
                      help="Model for the read-only agent (default: LLM_PROVIDER-configured model)")
    efab.add_argument("--max-turns", type=int, default=None, metavar="N", dest="max_turns",
                      help="Max agent turns per run")
    efab.add_argument("--limit", type=int, default=None, metavar="N",
                      help="Max artifacts to evaluate")

    # ── report ───────────────────────────────────────────────────────────────
    rep = subparsers.add_parser(
        "report",
        help="Print a human-readable summary of evaluation pairs output",
    )
    rep.add_argument("--pairs", required=True, metavar="FILE",
                     help="Path to *_pairs.jsonl (from evaluate, evaluate-swe, or evaluate-fabricated)")
    rep.add_argument("--runs", default=None, metavar="FILE",
                     help="Path to *_runs.jsonl for agent reasoning details (auto-derived if omitted)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "export-instances":
        _cmd_export_instances(args)
    elif args.command == "evaluate-swe":
        _cmd_evaluate_swe(args)
    elif args.command == "evaluate-fabricated":
        _cmd_evaluate_fabricated(args)
    elif args.command == "report":
        _cmd_report(args)
