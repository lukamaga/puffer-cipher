"""Run server output through the dashboard's real TypeScript decoders (Node >= 22.18)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "decode_with_dashboard.mjs"
PROTOCOL_TS = HERE.parents[1] / "dashboard" / "src" / "lib" / "protocol.ts"
MIN_NODE = (22, 18)


def node_skip_reason() -> str | None:
    """Why the real-decoder tests cannot run here, or ``None`` if they can."""
    if not PROTOCOL_TS.is_file():
        return "dashboard/src/lib/protocol.ts not found"
    node = shutil.which("node")
    if node is None:
        return "node is not installed"
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.\d+", version)
    if match is None or (int(match[1]), int(match[2])) < MIN_NODE:
        return f"node {version or '?'} cannot import TypeScript (needs >= 22.18)"
    return None


requires_node = pytest.mark.skipif(node_skip_reason() is not None, reason=str(node_skip_reason()))


def decode_with_dashboard(
    messages: list[str], rest: list[tuple[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """One result per message / REST body: ``{"ok": bool, "detail"?: str}``."""
    node = shutil.which("node")
    assert node is not None
    payload = json.dumps({"messages": messages, "rest": rest or []})
    completed = subprocess.run(
        [node, "--no-warnings", str(SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    results: list[dict[str, Any]] = json.loads(completed.stdout)
    return results
