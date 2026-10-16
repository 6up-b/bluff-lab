"""Load settings.env into os.environ (never committed). Tolerates `KEY = value` spacing."""
from __future__ import annotations

import os
from pathlib import Path

ALIASES = {
    # SurrealDB Agent Memory was formerly "Spectron"; accept either spelling.
    "SURREALDB_API_KEY": "SPECTRON_API_KEY",
    "SURREALDB_URL": "SPECTRON_URL",
    "SURREALDB_CONTEXT_ID": "SPECTRON_CONTEXT_ID",
}


def load_env(path: str | os.PathLike | None = None) -> dict[str, str]:
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "settings.env"
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        loaded[k] = v
        os.environ.setdefault(k, v)
        if k in ALIASES:
            os.environ.setdefault(ALIASES[k], v)
    os.environ.setdefault("ARIZE_PROJECT_NAME", "bluff-lab")
    return loaded


def has(*keys: str) -> bool:
    return all(os.environ.get(k) for k in keys)
