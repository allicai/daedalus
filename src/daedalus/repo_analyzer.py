from __future__ import annotations

import ast
import base64
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import requests

from daedalus import github_cache

GITHUB_API = "https://api.github.com"
_MAX_SYMBOLS_PER_FILE = 25
_FETCH_POOL = 14
_DEFAULT_MAX_FILES = 7
_INTER_REQUEST_DELAY = 0.12  # seconds

_STOP = frozenset({
    "this", "that", "with", "from", "when", "have", "been", "will", "would",
    "should", "could", "there", "their", "they", "them", "than", "then",
    "also", "into", "over", "after", "before", "about", "which", "where",
    "what", "while", "does", "issue", "error", "problem", "using", "used",
    "some", "each", "make", "made", "line", "lines", "code", "file", "test",
    "class", "method", "return", "value", "type", "data", "object", "instance",
    "variable", "parameter", "result", "output", "input", "behavior", "expected",
    "actual", "current", "default", "change", "following", "below", "above",
    "raise", "raised", "raises", "called", "call", "calls", "just", "only",
    "when", "such", "other", "more", "less", "very", "been", "being", "both",
})


@dataclass
class FileSymbols:
    path: str
    symbols: list[str] = field(default_factory=list)
    import_connected: bool = False   # True → patched file directly imports this file


def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(url: str, params: dict | None = None) -> dict | list:
    cached = github_cache.get(url, params)
    if cached is not None:
        return cached

    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    if resp.status_code == 403:
        msg = resp.json().get("message", "")
        if "rate limit" in msg.lower():
            raise RuntimeError(
                "GitHub API rate limit exceeded. Set GITHUB_TOKEN in .env for 5000 req/hr."
            )
    resp.raise_for_status()
    data = resp.json()
    github_cache.set(url, params, data)
    return data


def _patched_files(patch: str) -> list[str]:
    matches = re.findall(r'^(?:\+\+\+|---)\s+[ab]/(.+)$', patch, re.MULTILINE)
    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        m = m.strip()
        if m and m not in seen and not m.startswith("/dev/"):
            seen.add(m)
            out.append(m)
    return out


def _extract_python_symbols(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    symbols: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if len(symbols) >= _MAX_SYMBOLS_PER_FILE:
            break
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(f"{node.name}()")
        elif isinstance(node, ast.ClassDef):
            symbols.append(node.name)
            for child in ast.iter_child_nodes(node):
                if len(symbols) >= _MAX_SYMBOLS_PER_FILE:
                    break
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(f"{node.name}.{child.name}()")
    return symbols


def _imported_paths(source: str, all_paths: set[str]) -> list[str]:
    # Relative imports are skipped — resolving them requires knowing the package root.
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    found: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]

        for module in modules:
            candidate = module.replace(".", "/") + ".py"
            if candidate in all_paths and candidate not in seen:
                seen.add(candidate)
                found.append(candidate)

    return found


def _fetch_content(owner: str, name: str, commit: str, path: str) -> str | None:
    url = f"{GITHUB_API}/repos/{owner}/{name}/contents/{path}"
    try:
        data = _get(url, params={"ref": commit})
        if isinstance(data, dict) and data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def _fetch_symbols(owner: str, name: str, commit: str, path: str) -> list[str]:
    content = _fetch_content(owner, name, commit, path)
    if content and path.endswith(".py"):
        return _extract_python_symbols(content)
    return []


def _keywords(text: str) -> set[str]:
    words: set[str] = set()
    words.update(re.findall(r'\b[a-z][a-z]{3,}\b', text.lower()))
    for m in re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', text):
        words.add(m.lower())
        for part in re.findall(r'[A-Z][a-z]+', m):
            words.add(part.lower())
    for m in re.findall(r'\b[a-z][a-z0-9]*(?:_[a-z][a-z0-9]+)+\b', text):
        for part in m.split('_'):
            if len(part) >= 3:
                words.add(part)
    return words - _STOP


def _score(path: str, symbols: list[str], keywords: set[str]) -> int:
    text = path.lower() + " " + " ".join(s.lower() for s in symbols)
    return sum(1 for kw in keywords if kw in text)


def analyze(
    repo: str,
    commit: str,
    patch: str,
    problem_statement: str = "",
    max_files: int = _DEFAULT_MAX_FILES,
) -> list[FileSymbols]:
    # The gold patch is used only to locate the subsystem — never forwarded to the LLM.
    owner, name = repo.split("/", 1)

    patched = _patched_files(patch)
    if not patched:
        return []

    subsystem_dirs = {
        str(PurePosixPath(p).parent)
        for p in patched
        if "/" in p
    }
    if not subsystem_dirs:
        subsystem_dirs = {"."}

    try:
        tree_data = _get(
            f"{GITHUB_API}/repos/{owner}/{name}/git/trees/{commit}",
            params={"recursive": "1"},
        )
    except Exception as exc:
        raise RuntimeError(f"Could not fetch repo tree for {repo}@{commit[:8]}: {exc}") from exc

    all_blobs: list[str] = [
        item["path"]
        for item in tree_data.get("tree", [])
        if item.get("type") == "blob"
    ]
    all_paths = set(all_blobs)
    patched_set = set(patched)

    import_connected: list[str] = []
    for pf in patched:
        if not pf.endswith(".py"):
            continue
        content = _fetch_content(owner, name, commit, pf)
        time.sleep(_INTER_REQUEST_DELAY)
        if content:
            for imp_path in _imported_paths(content, all_paths):
                if imp_path not in patched_set and imp_path not in import_connected:
                    import_connected.append(imp_path)

    def in_subsystem(path: str) -> bool:
        return str(PurePosixPath(path).parent) in subsystem_dirs

    keywords = _keywords(problem_statement)

    def path_score(p: str) -> int:
        return sum(1 for kw in keywords if kw in p.lower())

    neighbours = [
        f for f in all_blobs
        if in_subsystem(f) and f not in patched_set and f not in import_connected
    ]
    neighbours.sort(key=lambda p: (-path_score(p), not p.endswith(".py"), len(p)))

    pool_paths = import_connected[:_FETCH_POOL]
    remaining = _FETCH_POOL - len(pool_paths)
    if remaining > 0:
        pool_paths += neighbours[:remaining]

    if not pool_paths:
        pool_paths = patched[:_FETCH_POOL]

    pool: list[FileSymbols] = []
    import_connected_set = set(import_connected)
    for path in pool_paths:
        symbols = _fetch_symbols(owner, name, commit, path)
        pool.append(FileSymbols(
            path=path,
            symbols=symbols,
            import_connected=(path in import_connected_set),
        ))
        time.sleep(_INTER_REQUEST_DELAY)

    if keywords:
        # +1000 bonus keeps import-connected files above neighbours with equal semantic score.
        pool.sort(key=lambda fs: -(
            _score(fs.path, fs.symbols, keywords) + (1000 if fs.import_connected else 0)
        ))

    return pool[:max_files]


def format_for_prompt(candidates: list[FileSymbols], commit: str) -> str:
    if not candidates:
        return ""
    lines = [
        f"Candidate files and symbols (verified at commit {commit[:8]}):"
    ]
    for fs in candidates:
        tag = " [imports this]" if fs.import_connected else ""
        if fs.symbols:
            lines.append(f"  {fs.path}{tag}: {', '.join(fs.symbols)}")
        else:
            lines.append(f"  {fs.path}{tag}")
    return "\n".join(lines)
