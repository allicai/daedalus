#!/usr/bin/env python3
"""Analyze a Daedalus output JSONL file.

Prints validation stats, score distributions, repo distribution, and
distractor patterns. Writes a CSV scores file and a Markdown review set
alongside the input file.

Usage
-----
python examples/analyze_output.py examples/output/tasks_verified.jsonl
python examples/analyze_output.py examples/output/tasks.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

_SCORE_RE = re.compile(
    r"plausibility=(\d+)\s+locality=(\d+)\s+competitiveness=(\d+)"
    r"(?:\s+semantic_fit=(\d+))?"
)


def load_variants(path: Path) -> list[dict]:
    variants = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                variants.append(json.loads(line))
    return variants


def parse_scores(notes: str) -> dict | None:
    m = _SCORE_RE.search(notes)
    if not m:
        return None
    result = {
        "plausibility": int(m.group(1)),
        "locality": int(m.group(2)),
        "competitiveness": int(m.group(3)),
    }
    if m.group(4) is not None:
        result["semantic_fit"] = int(m.group(4))
    return result


def parse_reason(notes: str) -> str:
    return re.sub(
        r"^\[.*?\]\s*(plausibility=\d+\s+locality=\d+\s+competitiveness=\d+\s*[—-]?\s*)?",
        "",
        notes,
    ).strip()


def distractor_category(text: str) -> str:
    if re.search(r"\w+\(\)", text) or re.search(r"_[a-z]\w+\b", text):
        return "function-level"
    if re.search(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", text):
        return "class-level"
    return "module/subsystem-level"


def score_dist_table(values: list[int], width: int = 5) -> str:
    dist = Counter(values)
    rows = []
    for score in range(1, 6):
        count = dist.get(score, 0)
        bar = "█" * count
        rows.append(f"    {score}  {bar:<{width}} {count}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a Daedalus output JSONL file.",
    )
    parser.add_argument("path", help="Path to tasks JSONL file")
    args = parser.parse_args()

    tasks_path = Path(args.path)
    if not tasks_path.exists():
        print(f"File not found: {tasks_path}")
        sys.exit(1)

    scores_csv = tasks_path.parent / (tasks_path.stem + "_scores.csv")
    review_md  = tasks_path.parent / (tasks_path.stem + "_review.md")

    variants = load_variants(tasks_path)
    total = len(variants)
    if total == 0:
        print("File is empty.")
        sys.exit(1)

    status_counts = Counter(v["quality_status"] for v in variants)

    scored = []
    for v in variants:
        s = parse_scores(v.get("validation_notes", ""))
        if s:
            scored.append({**v, **s})

    has_semantic_fit = any("semantic_fit" in (parse_scores(v.get("validation_notes", "")) or {}) for v in variants)
    dims = ("plausibility", "locality", "competitiveness", "semantic_fit") if has_semantic_fit else ("plausibility", "locality", "competitiveness")
    dim_values: dict[str, list[int]] = {d: [s[d] for s in scored if d in s] for d in dims}

    rejected = [v for v in variants if v["quality_status"] == "rejected"]
    heuristic_rejected = [v for v in rejected if v.get("validation_notes", "").startswith("[heuristic]")]
    llm_rejected = [v for v in rejected if v.get("validation_notes", "").startswith("[llm]")]
    min_pass = 4 if has_semantic_fit else 3
    failing_dims: Counter = Counter()
    for v in llm_rejected:
        s = parse_scores(v.get("validation_notes", ""))
        if s:
            for dim, val in s.items():
                if val < min_pass:
                    failing_dims[dim] += 1

    repo_counts   = Counter(v["repo"] for v in variants)
    categories    = Counter(
        distractor_category(v["transformation_metadata"]["intended_distractor"])
        for v in variants
    )

    W = 54
    print("=" * W)
    print("  Generation & Validation Report")
    print(f"  {tasks_path}")
    print("=" * W)
    print(f"\n  Total variants:  {total}")
    for status in ("validated", "rejected", "unverified"):
        n = status_counts.get(status, 0)
        pct = n / total * 100
        print(f"  {status:<14} {n:>3}  ({pct:.1f}%)")

    print(f"\n  LLM-scored:  {len(scored)} of {total}")

    if scored:
        print(f"\n{'─'*W}")
        threshold = 4 if has_semantic_fit else 3
        print(f"  Score Distributions  (1–5, pass threshold = {threshold})")
        print(f"{'─'*W}")
        headers = f"  {'':22} {'plaus':>6} {'local':>6} {'compete':>8}"
        if has_semantic_fit:
            headers += f" {'sem_fit':>8}"
        print(headers)

        def fmt(fn, vals):
            return f"{fn(vals):.2f}" if len(vals) > 1 else "n/a"

        for label, fn in [
            ("mean",   lambda v: f"{mean(v):.2f}" if v else "n/a"),
            ("median", lambda v: f"{median(v):.1f}" if v else "n/a"),
            ("stdev",  lambda v: fmt(stdev, v)),
        ]:
            row = f"  {label:<22}"
            for d in dims:
                row += f"  {fn(dim_values[d]):>6}"
            print(row)

        print()
        for d in dims:
            print(f"  {d}")
            print(score_dist_table(dim_values[d]))
            print()

        for d in dims:
            vals = dim_values[d]
            if len(vals) > 1 and len(set(vals)) == 1:
                print(f"  ⚠  WARNING: all {d} scores are {vals[0]}/5.")
                print(     "     Uniform scores may indicate a poorly calibrated validator.")
                print()

    print(f"{'─'*W}")
    print("  Rejection Breakdown")
    print(f"{'─'*W}")
    print(f"  Heuristic rejections:  {len(heuristic_rejected)}")
    print(f"  LLM rejections:        {len(llm_rejected)}")
    if failing_dims:
        print("  Failing dimensions:")
        for dim, n in failing_dims.most_common():
            print(f"    {dim}: {n}")

    print(f"\n{'─'*W}")
    print("  Repo Distribution")
    print(f"{'─'*W}")
    for repo, n in repo_counts.most_common():
        bar = "█" * n
        print(f"  {repo:<40} {n:>3}  {bar}")

    print(f"\n{'─'*W}")
    print("  Distractor Patterns")
    print(f"{'─'*W}")
    for cat, n in categories.most_common():
        print(f"  {cat:<26} {n:>3}  ({n/total*100:.1f}%)")

    fieldnames = [
        "variant_id", "repo", "quality_status",
        "plausibility", "locality", "competitiveness", "semantic_fit",
        "intended_distractor", "reason", "validation_notes",
    ]
    with scores_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in variants:
            notes = v.get("validation_notes", "")
            s = parse_scores(notes) or {}
            writer.writerow({
                "variant_id":          v["variant_id"],
                "repo":                v["repo"],
                "quality_status":      v["quality_status"],
                "plausibility":        s.get("plausibility", ""),
                "locality":            s.get("locality", ""),
                "competitiveness":     s.get("competitiveness", ""),
                "semantic_fit":        s.get("semantic_fit", ""),
                "intended_distractor": v["transformation_metadata"]["intended_distractor"],
                "reason":              parse_reason(notes),
                "validation_notes":    notes,
            })

    validated = [v for v in variants if v["quality_status"] == "validated"]

    lines: list[str] = [
        "# Daedalus — Manual Review Set\n",
        f"Source: `{tasks_path}`\n",
        f"{len(validated)} validated / {len(rejected)} rejected of {total} total variants.\n",
        "Score key: **P** = plausibility, **L** = locality, **C** = competitiveness "
        "(each 1–5, min 4 to pass)\n",
    ]

    def variant_block(v: dict) -> list[str]:
        meta  = v["transformation_metadata"]
        notes = v.get("validation_notes", "")
        s     = parse_scores(notes)
        score_str = (
            f"P={s['plausibility']} L={s['locality']} C={s['competitiveness']}"
            if s else "no scores"
        )
        orig = v["original_problem_statement"]
        mod  = v["modified_problem_statement"]
        added = mod[len(orig):].strip() if mod.startswith(orig) else "(modified in-place)"
        return [
            f"### {v['variant_id']}",
            f"**{v['quality_status'].upper()}** &nbsp; {score_str} &nbsp; `{v['repo']}`\n",
            f"**Intended distractor:** {meta['intended_distractor']}\n",
            f"**Text added to statement:**",
            f"> {added}\n",
            f"**Changes:** " + " · ".join(meta["changes_made"]) + "\n",
            f"**Validator:** {parse_reason(notes)}\n",
            "---\n",
        ]

    lines += ["\n## Validated\n"]
    for v in validated:
        lines += variant_block(v)

    lines += ["\n## Rejected\n"]
    for v in rejected:
        lines += variant_block(v)

    review_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'─'*W}")
    print("  Output files")
    print(f"{'─'*W}")
    print(f"  {scores_csv}")
    print(f"  {review_md}")


if __name__ == "__main__":
    main()
