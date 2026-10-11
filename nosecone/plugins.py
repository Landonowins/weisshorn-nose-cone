"""Loading user plugins: Python files listed in ``project.plugins`` (e.g. new profile families).

A plugin is an ordinary Python file that imports from ``nosecone`` and registers what it adds,
for example a profile family with :func:`nosecone.geometry.register_family`. Plugins are loaded
before the configuration is validated and again in every CFD worker process.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

_LOADED: set[str] = set()


def load_plugins(paths: Iterable[str | Path]) -> None:
    """Import every plugin file once (by absolute path)."""
    for p in paths or []:
        path = Path(p).resolve()
        key = str(path)
        if key in _LOADED:
            continue
        if not path.exists():
            raise FileNotFoundError(f"plugin not found: {path}")
        spec = importlib.util.spec_from_file_location(f"nosecone_plugin_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        _LOADED.add(key)
