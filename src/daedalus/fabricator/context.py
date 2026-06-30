"""Fetch distractor file content and gold patch for fabrication context."""
from __future__ import annotations

import re

from daedalus.fabricator.schemas import DistractorInfo
from daedalus.repo_analyzer import _fetch_content


_DISTRACTOR_RE = re.compile(
    r'^(.+?)\s+in\s+([\w/.\-]+\.py)\s*[—–\-]+\s*([A-Z]+)\s*:\s*(.+)$',
    re.DOTALL,
)


def parse_intended_distractor(raw: str) -> DistractorInfo:
    """Parse 'Symbol in path/to/file.py — ROLE: ownership sentence'."""
    m = _DISTRACTOR_RE.match(raw.strip())
    if not m:
        # Fallback: try to extract at least the file path
        file_m = re.search(r'in\s+([\w/.\-]+\.py)', raw)
        file_path = file_m.group(1) if file_m else ""
        return DistractorInfo(symbol="", file=file_path, role="", ownership=raw)
    return DistractorInfo(
        symbol=m.group(1).strip(),
        file=m.group(2).strip(),
        role=m.group(3).strip(),
        ownership=m.group(4).strip(),
    )


def fetch_distractor_file(repo: str, commit: str, file_path: str) -> str | None:
    """Return the full source of the distractor file at the task's base commit."""
    owner, name = repo.split("/", 1)
    return _fetch_content(owner, name, commit, file_path)


def load_gold_patch(instance_id: str) -> str:
    """Load the gold fix patch from the cached HuggingFace SWE-bench dataset."""
    from daedalus.loaders.swebench_loader import load_patches_by_id
    patches = load_patches_by_id({instance_id})
    return patches.get(instance_id, "")


def load_pass_to_pass(instance_ids: set[str]) -> dict[str, list[str]]:
    """Load PASS_TO_PASS test lists from SWE-bench Verified."""
    import json
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return {}

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    result: dict[str, list[str]] = {}
    for row in ds:
        iid = row["instance_id"]
        if iid not in instance_ids:
            continue
        raw = row.get("PASS_TO_PASS", "[]")
        try:
            result[iid] = json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            result[iid] = []
    return result
