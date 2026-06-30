from __future__ import annotations

import json
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def export(records: list[BaseModel], path: str | Path, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def load_source_ids(path: str | Path) -> set[str]:
    """Return the set of source_instance_ids already written to a JSONL file."""
    path = Path(path)
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["source_instance_id"])
                except Exception:
                    pass
    return ids


def load(path: str | Path, model_class: Type[M]) -> list[M]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(model_class.model_validate_json(line))
    return records
