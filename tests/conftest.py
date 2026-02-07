from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = (BASE_DIR / "src").resolve()

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: List[pytest.Item],
) -> None:
    """Fail fast if pytest ever generates duplicate node IDs."""
    del session
    del config

    seen: Dict[str, int] = {}
    duplicates: List[str] = []
    for item in items:
        nodeid = item.nodeid
        if nodeid in seen:
            duplicates.append(nodeid)
            continue
        seen[nodeid] = 1

    if not duplicates:
        return

    lines = "\n".join(f"- {nodeid}" for nodeid in sorted(set(duplicates)))
    raise pytest.UsageError(
        "Duplicate pytest nodeids detected during collection:\n" f"{lines}"
    )
