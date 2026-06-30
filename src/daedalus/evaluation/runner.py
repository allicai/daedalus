"""Read-only agent runner: measures exploration trajectory and stated diagnosis."""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Literal

from daedalus.evaluation.schemas import EvaluationBucket, EvaluationPair, EvaluationRun
from daedalus.openrouter import MAX_TOKENS_GENERATE, MODEL, PROVIDER_ROUTING, get_client, get_client_for_model

MAX_TURNS = 12
MAX_FILE_LINES = 200
MAX_SEARCH_RESULTS = 20

_SYSTEM = (
    "You are an expert software engineer. Your task is to investigate a bug report "
    "and identify its root cause in the codebase.\n\n"
    "Use the available tools to read files, search for relevant code, and trace the "
    "data flow between components. When you have gathered sufficient evidence to form "
    "a conclusion, stop using tools and summarize your findings."
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the repository root, e.g. 'django/db/models/query.py'",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to return (1-indexed). Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": f"Last line to return. Defaults to start_line + {MAX_FILE_LINES - 1}.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a string or pattern across Python files in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "String or regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search within (relative to repo root). Defaults to the whole repo.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a path in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the repository root. Defaults to the root.",
                    },
                },
                "required": [],
            },
        },
    },
]

_DIAGNOSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_diagnosis",
        "description": "Submit your final diagnosis of the root cause.",
        "parameters": {
            "type": "object",
            "properties": {
                "stated_root_cause": {
                    "type": "string",
                    "description": (
                        "One to three sentences: what is the specific bug, what invariant "
                        "is violated, and what observable symptom does it produce?"
                    ),
                },
                "ownership_assignment": {
                    "type": "string",
                    "description": (
                        "The specific function or class you believe is responsible for the fix. "
                        "Be precise — name the symbol and its file."
                    ),
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "How confident are you in this diagnosis based on your investigation?",
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "2-3 sentences. Did you investigate multiple candidate locations or focus on one? "
                        "If a competing hypothesis was visible, did you verify or dismiss it, and on what basis? "
                        "Describe the specific evidence that led to your conclusion — not what the problem "
                        "statement said, but what you found in the code."
                    ),
                },
            },
            "required": ["stated_root_cause", "ownership_assignment", "confidence", "reasoning"],
        },
    },
}


def _read_file(path: str, start: int, end: int, repo_path: Path, files_opened: list[str]) -> str:
    full = (repo_path / path).resolve()
    if not full.is_relative_to(repo_path.resolve()) or not full.is_file():
        return f"Error: file not found: {path}"

    if path not in files_opened:
        files_opened.append(path)

    lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    start = max(1, start)
    end = min(total, end)

    chunk = lines[start - 1 : end]
    numbered = "\n".join(f"{start + i:5d} | {line}" for i, line in enumerate(chunk))
    header = f"# {path}  (lines {start}–{end} of {total})\n"
    if end < total:
        header += f"# [{total - end} more lines — use start_line={end + 1} to continue]\n"
    return header + numbered


def _list_directory(path: str, repo_path: Path) -> str:
    target = (repo_path / path).resolve() if path else repo_path.resolve()
    if not target.is_relative_to(repo_path.resolve()) or not target.is_dir():
        return f"Error: not a directory: {path}"

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = []
    for e in entries:
        rel = e.relative_to(repo_path)
        lines.append(f"{'DIR' if e.is_dir() else 'FILE':4s}  {rel}")
    return "\n".join(lines) if lines else "(empty directory)"


def _parse_search_files(output: str) -> list[str]:
    """Extract unique file paths from grep-style output (file:line:content)."""
    seen: dict[str, None] = {}
    for line in output.splitlines():
        parts = line.split(":", 1)
        if parts:
            candidate = parts[0].replace("\\", "/")
            if candidate.endswith(".py") and candidate not in seen:
                seen[candidate] = None
    return list(seen)


def _search_code(pattern: str, path: str, repo_path: Path, files_searched: list[str]) -> str:
    output = ""
    try:
        cmd = ["git", "grep", "-n", "-i", "--", pattern]
        if path and path not in (".", ""):
            cmd.append(path)
        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if not output:
        search_root = (repo_path / path).resolve() if path and path not in (".", "") else repo_path.resolve()
        if not search_root.is_relative_to(repo_path.resolve()):
            search_root = repo_path.resolve()
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        hits = []
        for fp in search_root.rglob("*.py"):
            try:
                for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if regex.search(line):
                        hits.append(f"{fp.relative_to(repo_path)}:{i}: {line.rstrip()}")
                        if len(hits) >= MAX_SEARCH_RESULTS:
                            output = "\n".join(hits) + "\n[More matches — refine your pattern]"
                            break
            except Exception:
                continue
            if output:
                break
        if not output:
            output = "\n".join(hits) if hits else f"No matches for: {pattern}"

    if not output or output.startswith("No matches"):
        return output

    lines = output.splitlines()
    if len(lines) > MAX_SEARCH_RESULTS:
        output = "\n".join(lines[:MAX_SEARCH_RESULTS]) + f"\n[{len(lines) - MAX_SEARCH_RESULTS} more matches — refine your pattern]"

    for fp in _parse_search_files(output):
        if fp not in files_searched:
            files_searched.append(fp)

    return output


def _execute_tool(
    tool_call,
    repo_path: Path,
    files_opened: list[str],
    files_searched: list[str],
) -> str:
    try:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "read_file":
            start = args.get("start_line", 1)
            end = args.get("end_line", start + MAX_FILE_LINES - 1)
            return _read_file(args["path"], start, end, repo_path, files_opened)

        if name == "list_directory":
            return _list_directory(args.get("path", ""), repo_path)

        if name == "search_code":
            return _search_code(args["pattern"], args.get("path", ""), repo_path, files_searched)

        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool error: {exc}"


def _to_dict(msg) -> dict:
    """Convert an API response message object to a plain dict for subsequent calls."""
    d: dict = {"role": msg.role, "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def _extract_diagnosis(
    messages: list[dict],
    client,
    model: str,
    provider_routing: dict,
) -> tuple[dict, int, int]:
    """Append a diagnosis request to the conversation and return structured output.

    Returns (diagnosis_dict, input_tokens, output_tokens).
    """
    diag_messages = messages + [
        {
            "role": "user",
            "content": (
                "Based on your investigation, please submit your diagnosis using the tool. "
                "Name the specific function or class you believe is responsible."
            ),
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=512,
            extra_body=provider_routing,
            tools=[_DIAGNOSIS_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_diagnosis"}},
            messages=diag_messages,
        )
        result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        return (
            {
                "stated_root_cause": result.get("stated_root_cause", ""),
                "ownership_assignment": result.get("ownership_assignment", ""),
                "confidence": result.get("confidence", "unknown"),
                "reasoning": result.get("reasoning", ""),
            },
            in_tok,
            out_tok,
        )
    except Exception:
        return {"stated_root_cause": "", "ownership_assignment": "", "confidence": "unknown", "reasoning": ""}, 0, 0


def _parse_distractor(intended_distractor: str) -> tuple[str, str, list[str]]:
    """Extract (symbol, file_path, search_terms) from intended_distractor field.

    Format: 'Symbol in path/to/file.py — ROLE: sentence'
    """
    m = re.match(r"^(.+?)\s+in\s+([\w/\\.]+\.py)", intended_distractor)
    if not m:
        return "", "", []

    symbol = m.group(1).rstrip("()")
    file_path = m.group(2)

    terms: set[str] = set()
    terms.add(file_path.lower())

    basename = file_path.split("/")[-1]
    if basename.endswith(".py"):
        terms.add(basename[:-3].lower())

    for part in symbol.split("."):
        clean = part.strip("()")
        if len(clean) > 3 and not (clean.startswith("__") and clean.endswith("__")):
            terms.add(clean.lower())

    return symbol, file_path, sorted(terms)


def _parse_gold_files(patch: str) -> list[str]:
    """Extract unique patched file paths from a unified diff."""
    matches = re.findall(r'^\+\+\+\s+b/(.+)', patch, re.MULTILINE)
    seen: dict[str, None] = {}
    for m in matches:
        m = m.strip()
        if m and not m.startswith('/dev/'):
            seen[m] = None
    return list(seen)


def _extract_ownership_file(ownership_assignment: str) -> str:
    """Extract a Python file path from ownership_assignment.

    Handles slash paths (django/db/models/query.py) and dotted module paths
    (django.db.models.sql.query.Query.method — strips class/method suffix).
    """
    if not ownership_assignment:
        return ""

    # Slash-separated path is unambiguous — prefer it
    m = re.search(r'((?:\w+/)+\w+\.py)', ownership_assignment)
    if m:
        return m.group(1)

    # Dotted module path: match only the lowercase prefix before any CamelCase class name
    # e.g. "django.db.models.sql.query.Query.resolve_lookup_value" → "django/db/models/sql/query.py"
    m = re.search(r'\b([a-z]\w*(?:\.[a-z]\w*)+)', ownership_assignment)
    if m:
        parts = m.group(1).split('.')
        if len(parts) >= 2:
            return '/'.join(parts) + '.py'

    return ""


def _classify_ownership(
    ownership_file: str,
    gold_files: list[str],
    distractor_file: str,
) -> Literal["correct", "distractor_adopted", "other", "unknown"]:
    if not ownership_file:
        return "unknown"

    def _files_match(a: str, b: str) -> bool:
        a, b = a.replace("\\", "/"), b.replace("\\", "/")
        return a == b or a.endswith("/" + b) or b.endswith("/" + a)

    if gold_files and any(_files_match(ownership_file, gf) for gf in gold_files):
        return "correct"
    if distractor_file and _files_match(ownership_file, distractor_file):
        return "distractor_adopted"
    return "other"



def run_condition(
    task: dict,
    condition: Literal["original", "variant"],
    repo_path: Path,
    max_turns: int | None = MAX_TURNS,
    gold_patch: str = "",
    model: str | None = None,
) -> EvaluationRun:
    """Run the agent on one condition (original or variant) and return an EvaluationRun."""
    problem_statement = (
        task["original_problem_statement"]
        if condition == "original"
        else task["modified_problem_statement"]
    )

    meta = task["transformation_metadata"]
    intended_distractor = meta.get("intended_distractor", "")
    distractor_symbol, distractor_file, _ = _parse_distractor(intended_distractor)

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Repository: {task['repo']} (commit {task['base_commit'][:12]})\n\n"
                f"Bug report:\n{problem_statement}"
            ),
        },
    ]

    if model is not None:
        client, _routing = get_client_for_model(model)
        _model = model
    else:
        client = get_client()
        _model = MODEL
        _routing = PROVIDER_ROUTING

    files_opened: list[str] = []
    files_searched: list[str] = []
    tool_call_count = 0
    turns = 0
    input_tokens = 0
    output_tokens = 0
    stopped_naturally = False
    t0 = time.time()

    while max_turns is None or turns < max_turns:
        response = client.chat.completions.create(
            model=_model,
            max_tokens=MAX_TOKENS_GENERATE,
            extra_body=_routing,
            tools=_TOOLS,
            messages=messages,
        )
        turns += 1
        if response.usage:
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens

        msg = response.choices[0].message
        messages.append(_to_dict(msg))

        if not msg.tool_calls:
            stopped_naturally = True
            break

        for tc in msg.tool_calls:
            tool_call_count += 1
            result = _execute_tool(tc, repo_path, files_opened, files_searched)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    diagnosis, diag_in, diag_out = _extract_diagnosis(messages, client, _model, _routing)
    input_tokens += diag_in
    output_tokens += diag_out
    wall_time = time.time() - t0

    all_touched = set(files_opened) | set(files_searched)
    distractor_visited = (
        any(distractor_file in f or f in distractor_file for f in all_touched)
        if distractor_file else False
    )

    gold_files = _parse_gold_files(gold_patch)
    ownership_file = _extract_ownership_file(diagnosis.get("ownership_assignment", ""))
    ownership_label = _classify_ownership(ownership_file, gold_files, distractor_file)

    run_id = f"{task['variant_id']}__{condition}__{int(t0)}"

    return EvaluationRun(
        run_id=run_id,
        variant_id=task["variant_id"],
        source_instance_id=task["source_instance_id"],
        condition=condition,
        agent_id=_model,
        files_opened=list(dict.fromkeys(files_opened)),
        files_searched=list(dict.fromkeys(files_searched)),
        unique_files_opened=len(set(files_opened)),
        tool_calls=tool_call_count,
        turns=turns,
        wall_time_seconds=round(wall_time, 1),
        stopped_reason="natural" if stopped_naturally else "turn_limit",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        distractor_symbol=distractor_symbol,
        distractor_file=distractor_file,
        distractor_file_visited=distractor_visited,
        stated_root_cause=diagnosis["stated_root_cause"],
        ownership_assignment=diagnosis["ownership_assignment"],
        confidence=diagnosis.get("confidence", "unknown"),
        reasoning=diagnosis.get("reasoning", ""),
        gold_files=gold_files,
        ownership_file=ownership_file,
        ownership_label=ownership_label,
    )


def _assign_bucket(original: EvaluationRun, variant: EvaluationRun) -> EvaluationBucket:
    # Resolved-based: highest confidence signal
    if original.resolved is not None and variant.resolved is not None:
        if original.resolved and not variant.resolved:
            return "clean_shift"
        if original.resolved and variant.resolved:
            return "no_effect"
        if not original.resolved and variant.resolved:
            return "adoption_reduced"
        return "no_signal"

    # Ownership-label-based: fallback when resolved is unavailable
    if not variant.distractor_file_visited and variant.ownership_label != "distractor_adopted":
        return "bad_variant"

    o = original.ownership_label == "distractor_adopted"
    v = variant.ownership_label == "distractor_adopted"

    if not o and v:
        return "clean_shift"
    if o and v:
        return "contaminated"
    if o and not v:
        return "adoption_reduced"
    return "no_effect"


def compare_pair(
    original: EvaluationRun,
    variant: EvaluationRun,
) -> EvaluationPair:
    bucket = _assign_bucket(original, variant)

    trajectory_diverged = (
        original.turns != variant.turns
        or original.tool_calls != variant.tool_calls
        or original.unique_files_opened != variant.unique_files_opened
        or original.ownership_label != variant.ownership_label
        or original.distractor_file_visited != variant.distractor_file_visited
        or original.resolved != variant.resolved
    )

    return EvaluationPair(
        pair_id=f"{original.variant_id}__pair",
        variant_id=original.variant_id,
        source_instance_id=original.source_instance_id,
        original_run_id=original.run_id,
        variant_run_id=variant.run_id,
        turns_delta=variant.turns - original.turns,
        tool_calls_delta=variant.tool_calls - original.tool_calls,
        unique_files_delta=variant.unique_files_opened - original.unique_files_opened,
        original_distractor_visited=original.distractor_file_visited,
        variant_distractor_visited=variant.distractor_file_visited,
        original_ownership_label=original.ownership_label,
        variant_ownership_label=variant.ownership_label,
        original_resolved=original.resolved,
        variant_resolved=variant.resolved,
        evaluation_bucket=bucket,
        hypothesis_shifted=bucket == "clean_shift",
        trajectory_diverged=trajectory_diverged,
    )
