#!/usr/bin/env python3
"""Analyze batch output: validation stats, score distributions, distractor patterns, review set."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

TASKS_PATH = Path(__file__).parent / "output" / "tasks.jsonl"
SCORES_CSV = Path(__file__).parent / "output" / "scores.csv"
REVIEW_MD  = Path(__file__).parent / "output" / "review_set.md"

_SCORE_RE = re.compile(
    r"plausibility=(\d+)\s+locality=(\d+)\s+competitiveness=(\d+)"
    r"(?:\s+semantic_fit=(\d+))?"
)


def load_variants() -> list[dict]:
    variants = []
    with TASKS_PATH.open(encoding="utf-8") as f:
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
    return re.sub(r"^\[.*?\]\s*(plausibility=\d+\s+locality=\d+\s+competitiveness=\d+\s*[—-]?\s*)?", "", notes).strip()


def distractor_category(text: str) -> str:
    if re.search(r"\w+\(\)", text) or re.search(r"_[a-z]\w+\b", text):
        return "function-level"
    if re.search(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", text):
        return "class-level"
    return "module/subsystem-level"


def score_dist_table(values: list[int], width: int = 4) -> str:
    dist = Counter(values)
    bars = []
    for score in range(1, 6):
        count = dist.get(score, 0)
        bar = "█" * count
        bars.append(f"    {score}  {bar:<{width}} {count}")
    return "\n".join(bars)


def main() -> None:
    if not TASKS_PATH.exists():
        print(f"No output at {TASKS_PATH}. Run batch_generate.py first.")
        sys.exit(1)

    variants = load_variants()
    total = len(variants)
    if total == 0:
        print("tasks.jsonl is empty.")
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

    categories = Counter(distractor_category(v["transformation_metadata"]["intended_distractor"]) for v in variants)

    print("=" * 52)
    print("  Batch Statistics")
    print("=" * 52)
    print(f"\n  Total variants:  {total}")
    for status in ("validated", "rejected", "unverified"):
        n = status_counts.get(status, 0)
        print(f"  {status:<14} {n:>3}  ({n/total*100:.1f}%)")

    print(f"\n  LLM-scored:      {len(scored)} of {total}")

    if scored:
        print(f"\n{'─'*52}")
        print("  Score Distributions")
        print(f"{'─'*52}")
        print(f"  {'':22} {'plaus':>6} {'local':>6} {'compete':>8}")
        for label, fn in [("mean", lambda v: f"{mean(v):.2f}"), ("median", lambda v: f"{median(v):.1f}"),
                          ("stdev", lambda v: f"{stdev(v):.2f}" if len(v) > 1 else "n/a")]:
            row = f"  {label:<22}"
            for d in dims:
                row += f"  {fn(dim_values[d]):>6}"
            print(row)

        print()
        for d in dims:
            print(f"  {d}")
            print(score_dist_table(dim_values[d]))
            print()

    print(f"{'─'*52}")
    print("  Rejection Breakdown")
    print(f"{'─'*52}")
    print(f"  Heuristic rejections:  {len(heuristic_rejected)}")
    print(f"  LLM rejections:        {len(llm_rejected)}")
    if failing_dims:
        print("  Failing dimensions:")
        for dim, n in failing_dims.most_common():
            print(f"    {dim}: {n}")

    print(f"\n{'─'*52}")
    print("  Distractor Patterns")
    print(f"{'─'*52}")
    for cat, n in categories.most_common():
        print(f"  {cat:<26} {n:>3}  ({n/total*100:.1f}%)")

    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant_id", "repo", "quality_status",
                  "plausibility", "locality", "competitiveness",
                  "intended_distractor", "reason", "validation_notes"]
    with SCORES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in variants:
            notes = v.get("validation_notes", "")
            s = parse_scores(notes) or {}
            writer.writerow({
                "variant_id": v["variant_id"],
                "repo": v["repo"],
                "quality_status": v["quality_status"],
                "plausibility": s.get("plausibility", ""),
                "locality": s.get("locality", ""),
                "competitiveness": s.get("competitiveness", ""),
                "intended_distractor": v["transformation_metadata"]["intended_distractor"],
                "reason": parse_reason(notes),
                "validation_notes": notes,
            })

    validated = [v for v in variants if v["quality_status"] == "validated"]

    lines: list[str] = [
        "# Daedalus — Manual Review Set\n",
        f"{len(validated)} validated / {len(rejected)} rejected of {total} total variants.\n",
        "Score key: **P** = plausibility, **L** = locality, **C** = competitiveness (each 1–5, min 3 to pass)\n",
    ]

    def variant_block(v: dict) -> list[str]:
        meta = v["transformation_metadata"]
        notes = v.get("validation_notes", "")
        s = parse_scores(notes)
        score_str = (
            f"P={s['plausibility']} L={s['locality']} C={s['competitiveness']}"
            if s else "no scores"
        )
        added_text = v["modified_problem_statement"][len(v["original_problem_statement"]):].strip()
        return [
            f"### {v['variant_id']}",
            f"**{v['quality_status'].upper()}** &nbsp; {score_str} &nbsp; `{v['repo']}`\n",
            f"**Intended distractor:** {meta['intended_distractor']}\n",
            f"**Text added to statement:**",
            f"> {added_text}\n",
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

    REVIEW_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'─'*52}")
    print("  Output files")
    print(f"{'─'*52}")
    print(f"  {SCORES_CSV}")
    print(f"  {REVIEW_MD}")


if __name__ == "__main__":
    main()
