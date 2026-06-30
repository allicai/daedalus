"""Disk cache for GitHub API responses (~/.daedalus/github_cache/)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CACHE_DIR = Path.home() / ".daedalus" / "github_cache"


def _key(url: str, params: dict | None) -> str:
    raw = url + "|" + json.dumps(params, sort_keys=True) if params else url
    return hashlib.sha256(raw.encode()).hexdigest()


def get(url: str, params: dict | None = None) -> dict | list | None:
    path = _CACHE_DIR / (_key(url, params) + ".json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
    return None


def set(url: str, params: dict | None, data: dict | list) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / (_key(url, params) + ".json")
    path.write_text(json.dumps(data), encoding="utf-8")


def stats() -> dict[str, int]:
    if not _CACHE_DIR.exists():
        return {"entries": 0, "size_mb": 0}
    files = list(_CACHE_DIR.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {"entries": len(files), "size_mb": round(total_bytes / 1_048_576, 1)}
