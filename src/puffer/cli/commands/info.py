"""``puffer info``: versions, optional components, configuration and data directory.

The first thing to paste into a bug report, so it must work in a broken
installation too: every probe is independent and reports "unavailable"
instead of failing, and optional modules are located with
:func:`importlib.util.find_spec` (no import side effects, no OpenCV start-up
for a version string). Nothing secret is shown: the configuration holds no
secrets (``PufferConfig.to_dict`` even redacts stream-URL credentials) and
key files in the data directory are reported by existence only.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import typer
from rich.table import Table
from rich.text import Text

import puffer
from puffer.cli.commands.common import JsonOption, emit_json
from puffer.cli.commands.fish import DB_NAME
from puffer.cli.context import CliState, get_state
from puffer.cli.errors import CliError

__all__ = ["collect_info", "info", "libsodium_version"]

_COMPONENTS = {
    "engine": "puffer.engine",
    "runtime": "puffer.runtime",
    "api_server": "puffer.api.server",
    "analysis_report": "puffer.analysis.report",
    "yolo (ultralytics)": "ultralytics",
    "gpio (gpiozero)": "gpiozero",
    "picamera2": "picamera2",
}
_TOOLS = ("dieharder", "ent")
_BEACON_KEY = "beacon.key"


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def libsodium_version() -> str:
    """libsodium's version string, or where it comes from when that is not exposed.

    PyNaCl wheels link libsodium statically and do not export
    ``sodium_version_string``; rather than guess a number we say which
    PyNaCl build bundles it.
    """
    try:
        from nacl._sodium import ffi, lib  # PyNaCl's cffi module
    except ImportError:  # pragma: no cover - PyNaCl is a hard dependency
        return "unavailable"
    getter = getattr(lib, "sodium_version_string", None)
    if getter is not None:
        return str(ffi.string(getter()).decode("ascii"))
    return f"bundled with PyNaCl {_version('pynacl') or '?'}"


def _opencv_version() -> str | None:
    if importlib.util.find_spec("cv2") is None:
        return None
    import cv2

    return str(cv2.__version__)


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a parent package failed to import
        return False


def _config_section(state: CliState) -> tuple[dict[str, Any] | None, str | None, Path]:
    """``(config dict or None, error or None, data_dir)``."""
    try:
        config = state.config()
    except CliError as exc:
        data_dir = state.options.data_dir or Path("data")
        return None, exc.message, data_dir
    to_dict = getattr(config, "to_dict", None)
    values: dict[str, Any] = to_dict() if callable(to_dict) else dict(vars(config))
    return values, None, Path(config.data_dir)


def collect_info(state: CliState) -> dict[str, Any]:
    """Everything ``puffer info`` reports, as JSON-ready data."""
    config, config_error, data_dir = _config_section(state)
    config_file = state.options.config_path or os.environ.get("PUFFER_CONFIG") or None
    return {
        "puffer": {"version": puffer.__version__, "location": str(Path(puffer.__file__).parent)},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "libraries": {
            "libsodium": libsodium_version(),
            "pynacl": _version("pynacl"),
            "opencv": _opencv_version(),
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "fastapi": _version("fastapi"),
            "typer": _version("typer"),
            "rich": _version("rich"),
        },
        "components": {name: _available(module) for name, module in _COMPONENTS.items()},
        "tools": {tool: shutil.which(tool) for tool in _TOOLS},
        "config_file": os.fspath(config_file) if config_file else None,
        "config": config,
        "config_error": config_error,
        "data_dir": {
            "path": os.fspath(data_dir.resolve()),
            "exists": data_dir.is_dir(),
            "database": (data_dir / DB_NAME).is_file(),
            "beacon_key": (data_dir / _BEACON_KEY).is_file(),
        },
    }


def _yes(value: bool) -> Text:
    return Text("yes", style="green") if value else Text("no", style="dim")


def info(ctx: typer.Context, json_output: JsonOption = False) -> None:
    """Show versions (libsodium, OpenCV), optional components, configuration and data dir."""
    state = get_state(ctx)
    data = collect_info(state)
    if json_output:
        emit_json(data)
        return

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold", no_wrap=True)
    table.add_column()
    table.add_row("puffer-cipher", f"{data['puffer']['version']}  ({data['puffer']['location']})")
    py = data["python"]
    table.add_row("python", f"{py['implementation']} {py['version']}  ({py['executable']})")
    table.add_row("platform", data["platform"])
    for name, version in data["libraries"].items():
        table.add_row(name, Text(version) if version else Text("not installed", style="dim"))
    state.out.print(table)

    extras = Table.grid(padding=(0, 2))
    extras.add_column(style="bold", no_wrap=True)
    extras.add_column()
    for name, present in data["components"].items():
        extras.add_row(name, _yes(present))
    for tool, path in data["tools"].items():
        extras.add_row(tool, Text(path) if path else Text("not on PATH", style="dim"))
    state.out.print(Text("\ncomponents and tools", style="bold underline"))
    state.out.print(extras)

    state.out.print(Text("\nconfiguration", style="bold underline"))
    if data["config_file"]:
        state.out.print(Text(f"file {data['config_file']}", style="dim"))
    if data["config"] is None:
        state.out.print(Text(f"unavailable: {data['config_error']}", style="yellow"))
    else:
        settings = Table.grid(padding=(0, 2))
        settings.add_column(style="bold", no_wrap=True)
        settings.add_column()
        for key, value in data["config"].items():
            settings.add_row(key, Text(str(value)))
        state.out.print(settings)

    dd = data["data_dir"]
    state.out.print(Text("\ndata directory", style="bold underline"))
    state.out.print(
        Text.assemble(
            (dd["path"], ""),
            ("" if dd["exists"] else "  (does not exist yet)", "dim"),
            (f"\n{DB_NAME}: ", "dim"),
            ("present" if dd["database"] else "absent", ""),
            (f" · {_BEACON_KEY}: ", "dim"),
            ("present" if dd["beacon_key"] else "absent", ""),
        )
    )
