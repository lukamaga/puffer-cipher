"""The root command: help, version, global error handling."""

from __future__ import annotations

import subprocess
import sys

import pytest

from puffer import __version__
from puffer.cli import main

from .conftest import Invoke, Tank


def test_help_lists_every_command_and_the_exit_codes(cli: Invoke) -> None:
    result = cli("--help")

    assert result.exit_code == 0
    for command in (
        "keygen", "encrypt", "decrypt", "random", "simulate", "serve", "beacon", "analyze",
        "fish", "info",
    ):  # fmt: skip
        assert command in result.stdout
    assert "Exit status" in result.stdout
    assert "130 interrupted" in result.stdout
    assert "PUFFER_PASSWORD" in result.stdout
    assert "[puffer] table" in result.stdout  # literal brackets survive Rich markup


def test_no_arguments_shows_help(cli: Invoke) -> None:
    result = cli()
    assert "Usage" in result.output


def test_version(cli: Invoke) -> None:
    result = cli("--version")
    assert result.exit_code == 0
    assert result.stdout.strip() == f"puffer-cipher {__version__}"


def test_unknown_liveness_mode_is_a_usage_error(cli: Invoke) -> None:
    result = cli("--liveness", "maybe", "info")
    assert result.exit_code == 2


def test_internal_errors_are_one_line_and_leak_no_locals(cli: Invoke, tank: Tank) -> None:
    tank.error = ZeroDivisionError("division by zero")

    result = cli("keygen")

    assert result.exit_code == 1
    assert "internal error: ZeroDivisionError: division by zero" in result.stderr
    assert "re-run with -v" in result.stderr
    assert "Traceback" not in result.output


def test_verbose_shows_the_traceback(cli: Invoke, tank: Tank) -> None:
    tank.error = ZeroDivisionError("division by zero")

    result = cli("-v", "keygen")

    assert result.exit_code == 1
    assert "Traceback" in result.stderr
    assert "re-run with -v" not in result.stderr


def test_main_entry_point(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_python_dash_m() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "puffer.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert __version__ in completed.stdout


def test_startup_does_not_import_the_heavy_stack() -> None:
    """``puffer --help`` must stay fast: no OpenCV, SciPy, FastAPI or engine at import."""
    code = (
        "import sys, puffer.cli; "
        "print(sorted(m for m in ('cv2', 'scipy', 'fastapi', 'puffer.engine', 'puffer.sources') "
        "if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert completed.stdout.strip() == "[]"
