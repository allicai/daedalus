#!/usr/bin/env python3
"""Evaluation v0: paired agent runs on original vs. variant problem statements.

Runs the same agent on both the original SWE-bench problem statement and the
Daedalus modified statement for N validated tasks, then records trajectory and
diagnosis metrics for comparison.

No patch generation, no test execution. Read-only tool access only.

Usage
-----
# 5 tasks (default)
python examples/run_eval.py

# Specific number of tasks
python examples/run_eval.py --limit 3

# Specify output paths
python examples/run_eval.py --runs examples/output/eval_runs.jsonl --pairs examples/output/eval_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.runner import compare_pair, run_condition
from evaluation.schemas import EvaluationPair, EvaluationRun

EXAMPLES_DIR = Path(__file__).parent
TASKS_PATH = EXAMPLES_DIR / "output" / "tasks_main.jsonl"
DEFAULT_RUNS_PATH = EXAMPLES_DIR / "output" / "eval_runs.jsonl"
DEFAULT_PAIRS_PATH = EXAMPLES_DIR / "output" / "eval_pairs.jsonl"
REPO_CACHE_DIR = EXAMPLES_DIR / "repos"


def prepare_repo(repo: str, commit: str) -> Path:
    """Clone repo (once) and checkout to commit. Returns path to working copy."""
    REPO_CACHE_DIR.mkdir(exist_ok=True)
    repo_name = repo.replace("/", "__")
    repo_path = REPO_CACHE_DIR / repo_name

    if not repo_path.exists():
        token = os.environ.get("GITHUB_TOKEN", "")
        url = f"https://{token + '@' if token else ''}github.com/{repo}.git"
        print(f"  Cloning {repo} — this may take a few minutes ...", end=" ", flush=True)
        subprocess.run(
            ["git", "clone", url, str(repo_path)],
            check=True,
            capture_output=True,
        )
        print("done")

    # Fetch the exact commit in case it's not in the local clone
    subprocess.run(
        ["git", "fetch", "--quiet", "origin"],
        cwd=str(repo_path),
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", commit],
        cwd=str(repo_path),
        check=True,
        capture_output=True,
    )
    return repo_path


def load_eval_tasks(path: Path, limit: int) -> list[dict]:
    all_tasks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validated = [t for t in all_tasks if t["quality_status"] == "validated"]
    if not validated:
        raise RuntimeError(f"No validated tasks found in {path}")
    return validated[:limit]


def _append_jsonl(record: EvaluationRun | EvaluationPair, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def _print_run(run: EvaluationRun) -> None:
    dist   = "distractor=YES" if run.distractor_file_visited else "distractor=no "
    hyp    = "hypothesis=ADOPTED" if run.mentions_distractor else "hypothesis=rejected"
    stop   = "natural  " if run.stopped_reason == "natural" else "turn_limit"
    cost   = f"~${run.estimated_cost_usd:.3f}" if run.estimated_cost_usd else ""
    tokens = f"{run.input_tokens + run.output_tokens:,}tok" if run.input_tokens else ""
    file_summary = f"{run.unique_files_opened}read/{len(run.files_searched)}searched"
    print(
        f"    {run.condition:8s}  "
        f"{run.turns:2d} turns/{stop}  {run.tool_calls:3d} calls  "
        f"{file_summary}  "
        f"{dist}  {hyp}  [{run.confidence}]  "
        f"{run.wall_time_seconds:.0f}s  {tokens} {cost}"
    )
    if run.reasoning:
        snippet = run.reasoning[:160].replace("\n", " ")
        if len(run.reasoning) > 160:
            snippet += "…"
        print(f"    reasoning: {snippet}")


_BUCKET_ORDER = ["clean_shift", "no_effect", "contaminated", "adoption_reduced", "bad_variant"]

_BUCKET_LABEL = {
    "clean_shift":      "CLEAN SHIFT     — variant shifted agent to wrong hypothesis",
    "no_effect":        "NO EFFECT       — variant visited but didn't shift diagnosis",
    "contaminated":     "CONTAMINATED    — original already adopted distractor",
    "adoption_reduced": "ADOPT REDUCED   — variant reduced distractor adoption (unexpected)",
    "bad_variant":      "BAD VARIANT     — variant had no traction; distractor not visited",
}


def _print_summary(pairs: list[EvaluationPair]) -> None:
    print("\n" + "=" * 105)
    print("SUMMARY")
    print("=" * 105)
    header = (
        f"{'variant_id':<40} {'type':<20} "
        f"{'Δturn':>6} {'Δcall':>6} {'Δfile':>6} "
        f"{'bucket':<18}"
    )
    print(header)
    print("-" * 105)

    for p in pairs:
        short_id = p.variant_id.replace("__root_cause_attribution__001", "")
        print(
            f"{short_id:<40} {p.ownership_type:<20} "
            f"{p.turns_delta:+6d} {p.tool_calls_delta:+6d} {p.unique_files_delta:+6d} "
            f"{p.evaluation_bucket:<18}"
        )

    # Bucket counts
    from collections import Counter
    counts = Counter(p.evaluation_bucket for p in pairs)

    print("\n" + "=" * 105)
    print("\n## Evaluation Buckets\n")
    for bucket in _BUCKET_ORDER:
        n = counts.get(bucket, 0)
        bar = "█" * n
        label = _BUCKET_LABEL[bucket]
        print(f"  {bucket:<20} {n:>3}  {bar}  {label if n else ''}")

    total = len(pairs)
    n_clean = counts.get("clean_shift", 0)
    n_bad   = counts.get("bad_variant", 0)
    n_cont  = counts.get("contaminated", 0)
    print(f"\n  Total pairs: {total}  |  Actionable (clean_shift + no_effect): {total - n_cont - n_bad}")
    if n_clean:
        print(f"  → {n_clean} clean shift(s): Daedalus variants demonstrably altered agent diagnosis.")
    if n_bad:
        print(f"  → {n_bad} bad variant(s): review intended_distractor and modified_problem_statement.")
    if n_cont:
        print(f"  → {n_cont} contaminated: original already wrong — distractor too dominant, consider redesign.")


def _print_ownership_analysis(pairs: list[EvaluationPair]) -> None:
    from collections import defaultdict

    # Group pairs by ownership_type
    by_type: dict[str, list[EvaluationPair]] = defaultdict(list)
    for p in pairs:
        by_type[p.ownership_type].append(p)

    # Per-type bucket counts and success rate
    # success_rate = clean_shift / (clean_shift + no_effect); ignore contaminated/bad_variant
    rows = []
    for otype, type_pairs in by_type.items():
        bucket_counts: dict[str, int] = {}
        for b in _BUCKET_ORDER:
            bucket_counts[b] = sum(1 for p in type_pairs if p.evaluation_bucket == b)
        n_clean = bucket_counts["clean_shift"]
        n_no_effect = bucket_counts["no_effect"]
        denom = n_clean + n_no_effect
        rate = n_clean / denom if denom > 0 else None
        rows.append((otype, bucket_counts, n_clean, rate, len(type_pairs)))

    # Sort: primary = clean_shift desc, secondary = success_rate desc (None last)
    rows.sort(key=lambda r: (-(r[2]), -(r[3] if r[3] is not None else -1)))

    print("\n" + "=" * 105)
    print("OWNERSHIP TYPE ANALYSIS")
    print("=" * 105)

    # Cross-table header
    col_w = 10
    type_w = 22
    header = f"{'ownership_type':<{type_w}}"
    for b in _BUCKET_ORDER:
        short = b.replace("_", " ")[:col_w]
        header += f"  {short:>{col_w}}"
    header += f"  {'total':>{col_w}}  {'success%':>{col_w}}"
    print(header)
    print("-" * 105)

    for otype, bucket_counts, n_clean, rate, total in rows:
        row = f"{otype:<{type_w}}"
        for b in _BUCKET_ORDER:
            row += f"  {bucket_counts[b]:>{col_w}}"
        rate_str = f"{rate * 100:.0f}%" if rate is not None else "—"
        row += f"  {total:>{col_w}}  {rate_str:>{col_w}}"
        print(row)

    # Rankings
    print("\n## Rankings\n")
    print("  By clean_shift count:")
    for rank, (otype, _, n_clean, rate, _total) in enumerate(rows, 1):
        rate_str = f"{rate * 100:.0f}%" if rate is not None else "—"
        print(f"    {rank}. {otype:<22} clean_shift={n_clean}  success_rate={rate_str}")

    rate_rows = sorted(rows, key=lambda r: -(r[3] if r[3] is not None else -1))
    print("\n  By success rate (clean_shift / [clean_shift + no_effect]):")
    for rank, (otype, _, n_clean, rate, _total) in enumerate(rate_rows, 1):
        rate_str = f"{rate * 100:.0f}%" if rate is not None else "—"
        print(f"    {rank}. {otype:<22} success_rate={rate_str}  clean_shift={n_clean}")

    # Interpretation
    print("\n## Interpretation\n")
    if not rows:
        print("  No data.")
        return

    best_clean = rows[0]
    best_rate = rate_rows[0]
    all_no_effect = [r for r in rows if r[2] == 0 and r[3] == 0.0]
    all_contaminated = [r for r in rows if r[1]["contaminated"] == r[4]]

    if best_clean[2] > 0:
        print(f"  Most effective type by raw impact: {best_clean[0]} ({best_clean[2]} clean shift(s)).")
    if best_rate[3] is not None and best_rate[3] > 0:
        print(f"  Highest success rate: {best_rate[0]} ({best_rate[3] * 100:.0f}% of actionable pairs shifted).")
    if all_no_effect:
        types_str = ", ".join(r[0] for r in all_no_effect)
        print(f"  No hypothesis shift achieved for: {types_str} — variants visited but failed to redirect diagnosis.")
    if all_contaminated:
        types_str = ", ".join(r[0] for r in all_contaminated)
        print(f"  Fully contaminated type(s): {types_str} — original agent already adopts distractor independently.")

    total_pairs = len(pairs)
    total_clean = sum(r[2] for r in rows)
    total_actionable = sum(r[1]["clean_shift"] + r[1]["no_effect"] for r in rows)
    overall_rate = total_clean / total_actionable if total_actionable else None
    overall_str = f"{overall_rate * 100:.0f}%" if overall_rate is not None else "—"
    print(f"\n  Overall: {total_clean}/{total_actionable} actionable pairs shifted ({overall_str}) across {total_pairs} total.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daedalus Evaluation v0 — paired agent runs.")
    parser.add_argument("--limit", type=int, default=5, metavar="N",
                        help="Number of validated tasks to evaluate (default: 5)")
    parser.add_argument("--runs", type=str, default=str(DEFAULT_RUNS_PATH), metavar="PATH",
                        help="Output path for evaluation_runs JSONL")
    parser.add_argument("--pairs", type=str, default=str(DEFAULT_PAIRS_PATH), metavar="PATH",
                        help="Output path for evaluation_pairs JSONL")
    parser.add_argument("--max-turns", type=int, default=20, metavar="N",
                        help="Max agent turns per run (default: 20). Pass 0 for no cap.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 1 task, 3 turns. Quick wiring check before a full run.")
    args = parser.parse_args()

    if args.smoke:
        args.limit = 1
        args.max_turns = 5

    max_turns = None if args.max_turns == 0 else args.max_turns
    runs_path = Path(args.runs)
    pairs_path = Path(args.pairs)

    if not TASKS_PATH.exists():
        print(f"Tasks file not found: {TASKS_PATH}")
        sys.exit(1)

    tasks = load_eval_tasks(TASKS_PATH, args.limit)
    cap_str = f"{max_turns} turns" if max_turns else "no turn cap"
    print(f"Loaded {len(tasks)} validated task(s) for evaluation. ({cap_str} per run)\n")

    completed_pairs: list[EvaluationPair] = []

    for i, task in enumerate(tasks, 1):
        vid = task["variant_id"]
        ownership_type = (
            task.get("transformation_metadata", {})
            .get("ambiguity", {})
            .get("ownership_type", "unknown")
        )
        print(f"[{i}/{len(tasks)}] {vid}")
        print(f"  ownership_type: {ownership_type}")
        print(f"  repo:           {task['repo']}@{task['base_commit'][:12]}")

        repo_path = prepare_repo(task["repo"], task["base_commit"])

        original_run: EvaluationRun | None = None
        variant_run: EvaluationRun | None = None

        for condition in ("original", "variant"):
            print(f"  Running {condition} ...", flush=True)
            try:
                run = run_condition(task, condition, repo_path, max_turns=max_turns)
                _print_run(run)
                _append_jsonl(run, runs_path)
                if condition == "original":
                    original_run = run
                else:
                    variant_run = run
            except Exception as exc:
                print(f"    FAILED: {exc}")

        if original_run and variant_run:
            pair = compare_pair(original_run, variant_run, ownership_type)
            _append_jsonl(pair, pairs_path)
            completed_pairs.append(pair)
            shift_tag = " ← HYPOTHESIS SHIFTED" if pair.hypothesis_shifted else ""
            print(f"  diverged={pair.trajectory_diverged}{shift_tag}")

        print()
        if i < len(tasks):
            time.sleep(3)

    if completed_pairs:
        _print_summary(completed_pairs)
        _print_ownership_analysis(completed_pairs)

    print(f"\nRuns  → {runs_path}")
    print(f"Pairs → {pairs_path}")


if __name__ == "__main__":
    main()
