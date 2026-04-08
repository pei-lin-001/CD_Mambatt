from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    search_from = current if current.is_dir() else current.parent
    for candidate in [search_from, *search_from.parents]:
        if (candidate / "train_supervised.py").exists() and (candidate / "cd_mambatt").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {__file__}")


def ensure_repo_root() -> Path:
    repo_root = find_repo_root()
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    return repo_root
