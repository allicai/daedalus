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
