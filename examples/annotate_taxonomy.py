#!/usr/bin/env python3
"""Annotate existing variants with ambiguity taxonomy.

Assigns ownership_type / failing_state / owner_a / owner_b / ownership_question
to every variant that does not already have them, then writes the file back in-place.
Pass --all to re-annotate entries that already have taxonomy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.task_schema import AmbiguityTaxonomy
from openrouter import MAX_TOKENS_VALIDATE, MODEL, PROVIDER_ROUTING, get_client

_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_taxonomy",
        "description": "Assign ambiguity taxonomy to a root_cause_attribution variant.",
        "parameters": {
            "type": "object",
            "properties": {
                "ownership_type": {
                    "type": "string",
                    "enum": [
                        "producer_consumer",
                        "validator_caller",
                        "operation_state",
                        "parser_normalizer",
                        "lifecycle_ordering",
                        "state_transition",
                        "schema_state",
                        "unknown",
                    ],
                    "description": (
                        "producer_consumer  — one component produces the state, another consumes it.\n"
                        "validator_caller   — a validator should have enforced an invariant before the caller.\n"
                        "operation_state    — a migration/DB operation owns the logic but state layer is contested.\n"
                        "parser_normalizer  — parser should accept a format, or normalizer should normalise it first.\n"
                        "lifecycle_ordering — two components share responsibility at different lifecycle phases.\n"
                        "state_transition   — a state transition (rename/delete/move) is contested between owners.\n"
                        "schema_state       — schema evolution or migration state boundary is ambiguous.\n"
                        "unknown            — does not fit the above."
                    ),
                },
                "failing_state": {
                    "type": "string",
                    "description": (
                        "Short phrase: the specific value, property, or invariant that is violated. "
                        "Domain terms, not code terms. "
                        "E.g. 'Negative duration representation not parseable after round-trip'"
                    ),
                },
                "owner_a": {
                    "type": "string",
                    "description": "The TRUE bug site — the function or class actually responsible.",
                },
                "owner_b": {
                    "type": "string",
                    "description": "The DISTRACTOR — the competing owner the variant implicates.",
                },
                "ownership_question": {
                    "type": "string",
                    "description": (
                        "Single sentence capturing the ambiguity a developer must resolve. "
                        "Format: 'Should [action] happen in [owner_a] or [owner_b]?' "
                        "E.g. 'Should negative durations be normalised during formatting or accepted during parsing?'"
                    ),
                },
            },
            "required": [
                "ownership_type",
                "failing_state",
                "owner_a",
                "owner_b",
                "ownership_question",
            ],
        },
    },
}

_SYSTEM = """\
You are extracting structured metadata from a root_cause_attribution variant.

Given the variant's description and intended_distractor, assign the five fields below.

For ownership_type, choose the MOST SPECIFIC type that applies.
Use 'producer_consumer' only as a last resort — it is the fallback, not the default.

Decision guide (check in this order):
  1. parser_normalizer  — Is the ambiguity about whether to normalise/transform at format-time
                          or accept a wider format at parse-time?
                          Example: duration_string() vs parse_duration() for negative durations.

  2. validator_caller   — Is the ambiguity about which component should enforce a constraint
                          or check membership before the caller receives the state?
                          Example: ForeignKey.validate() vs descriptor.get_queryset() for manager selection.

  3. operation_state    — Is the ambiguity between a migration/database OPERATION and a
                          STATE LAYER (ProjectState, schema graph) about where logic lives?
                          Example: RenameModel.database_forwards() vs ProjectState.rename_model() for noop detection.

  4. state_transition   — Is the ambiguity about which component should update a key/name/id
                          during a rename, move, or delete operation?
                          Example: autodetector vs ProjectState for dictionary key after rename.

  5. schema_state       — Is the ambiguity specifically about schema migration state tracking
                          (index names, table names, constraint names across migrations)?

  6. lifecycle_ordering — Is the ambiguity about WHEN (setup vs. execution, pre vs. post)
                          something should happen, not just which component does it?

  7. producer_consumer  — Generic: one component produces the state, another consumes it,
                          and neither more-specific type applies.

Assign:
  • ownership_type      — most specific applicable type
  • failing_state       — the invariant violated, in domain terms (not code)
  • owner_a             — the TRUE bug site
  • owner_b             — the DISTRACTOR
  • ownership_question  — single sentence: "Should [action] happen in [owner_a] or [owner_b]?"

Examples of ownership_question:
  "Should negative durations be normalised during formatting or accepted during parsing?"
  "Should unused annotations be stripped before building the count subquery or on receipt?"
  "Should FK validation use the descriptor's queryset or perform its own manager lookup?"
  "Should noop detection live in the migration operation or in the state layer?"
"""


def _annotate(variant: dict, client) -> dict:
    meta = variant["transformation_metadata"]
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_VALIDATE,
        extra_body=PROVIDER_ROUTING,
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_taxonomy"}},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Variant ID: {variant['variant_id']}\n"
                    f"Repo: {variant['repo']}\n\n"
                    f"Description:\n{meta['description']}\n\n"
                    f"Intended distractor:\n{meta['intended_distractor']}\n\n"
                    f"Changes made:\n" + "\n".join(f"- {c}" for c in meta["changes_made"])
                ),
            },
        ],
    )
    result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    taxonomy = AmbiguityTaxonomy(**result)
    meta["ambiguity"] = taxonomy.model_dump()
    return variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate Daedalus variants with ambiguity taxonomy.")
    parser.add_argument("path", help="Path to JSONL file to annotate")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Annotate ALL variants (default: only those without existing taxonomy)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    variants = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                variants.append(json.loads(line))

    client = get_client()
    annotated = 0
    skipped = 0

    for i, v in enumerate(variants):
        meta = v["transformation_metadata"]
        already_has = meta.get("ambiguity") is not None
        if already_has and not args.all:
            skipped += 1
            continue

        print(f"[{i+1}/{len(variants)}] Annotating {v['variant_id']} ({v['quality_status']}) ...",
              end=" ", flush=True)
        try:
            variants[i] = _annotate(v, client)
            tax = variants[i]["transformation_metadata"]["ambiguity"]
            print(f"{tax['ownership_type']}")
            annotated += 1
        except Exception as exc:
            print(f"FAILED: {exc}")

    path.write_text(
        "\n".join(json.dumps(v) for v in variants) + "\n",
        encoding="utf-8",
    )
    print(f"\nDone. annotated={annotated}  skipped={skipped}")
    print(f"Written → {path}")


if __name__ == "__main__":
    main()
