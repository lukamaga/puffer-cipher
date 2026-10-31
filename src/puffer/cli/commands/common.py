"""Option types and helpers shared by several commands."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import typer

from puffer.cli import wiring
from puffer.cli.context import PASSWORD_ENV, CliState
from puffer.cli.errors import CliError, ExitCode
from puffer.cli.protocols import ConfigLike, EngineLike

__all__ = [
    "ForceOption",
    "JsonOption",
    "MinBitsOption",
    "MinFishBitsOption",
    "SourceOption",
    "StrictOption",
    "TimeoutOption",
    "check_reachable",
    "emit_json",
    "open_engine",
    "read_password",
    "refuse_existing",
]

SourceOption = Annotated[
    str | None,
    typer.Option(
        "--source",
        metavar="SPEC",
        help="Frame source for this command (overrides the global --source), "
        "e.g. synthetic:seed=7, video:tank.mp4, camera:0.",
    ),
]
MinBitsOption = Annotated[
    float,
    typer.Option(
        "--min-bits",
        min=0.0,
        help="Camera credit (sensor + fish, in bits) to collect before drawing.",
    ),
]
MinFishBitsOption = Annotated[
    float,
    typer.Option(
        "--min-fish-bits",
        min=0.0,
        help="Of that, credit the fish channel itself must earn (the sensor alone fills the "
        "camera target within a frame; fish need ~1000 observations to pass start-up tests).",
    ),
]
TimeoutOption = Annotated[
    float,
    typer.Option("--timeout", min=0.0, help="Seconds to wait for that camera credit."),
]
StrictOption = Annotated[
    bool,
    typer.Option(
        "--strict",
        help="Fail (exit 8) if the camera target is not met, instead of falling back to "
        "a key that is exactly as strong as os.urandom.",
    ),
]
ForceOption = Annotated[bool, typer.Option("--force", help="Overwrite existing output files.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON on stdout.")]


def refuse_existing(*paths: Path, force: bool) -> None:
    """Fail fast, before any slow work, if an output exists and ``--force`` is not set.

    The writers check again atomically when they publish the file; this
    early check only spares the user a minute of entropy collection that
    would end in "file exists".
    """
    if force:
        return
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(17, "refusing to overwrite existing file", os.fspath(path))


def check_reachable(target_bits: float, config: ConfigLike, fish_bits: float = 0.0) -> float:
    """Reject targets the saturating credit ledger can never hold; return the pool maximum.

    Each camera channel's credit saturates at ``pool_capacity_bits``
    (DESIGN.md 4.4), so the camera total is capped at twice that and the
    fish channel alone at once that. Waiting for more could only time out.
    """
    capacity = float(config.pool_capacity_bits)
    reachable = capacity * 2  # sensor + fish channels
    hint = "lower the target or raise pool_capacity_bits in the configuration"
    if target_bits > reachable:
        raise CliError(
            f"--min-bits {target_bits:g} can never be reached: each camera channel's credit "
            f"saturates at {capacity:g} bits ({reachable:g} in total)",
            ExitCode.USAGE,
            hint=hint,
        )
    if fish_bits > capacity:
        raise CliError(
            f"--min-fish-bits {fish_bits:g} can never be reached: the fish channel's credit "
            f"saturates at {capacity:g} bits",
            ExitCode.USAGE,
            hint=hint,
        )
    return reachable


@contextlib.contextmanager
def open_engine(state: CliState, config: ConfigLike) -> Iterator[EngineLike]:
    """Build the engine for ``config`` and always close it (or a camera stays locked)."""
    with state.err.status("Opening the aquarium...", spinner="dots"):
        engine = wiring.build_engine(config)
    try:
        yield engine
    finally:
        engine.close()


def read_password(*, confirm: bool) -> str:
    """The password from ``$PUFFER_PASSWORD``, else a hidden prompt on the terminal.

    ``confirm`` asks twice (encryption: a typo would make the file
    unrecoverable). The environment variable exists for scripts and tests;
    note that other processes of the same user can read a process's
    environment, so interactive use should prefer the prompt.
    """
    value = os.environ.get(PASSWORD_ENV)
    if value is not None:
        if not value:
            raise CliError(f"${PASSWORD_ENV} is set but empty", ExitCode.USAGE)
        return value
    password: str = typer.prompt("Password", hide_input=True, confirmation_prompt=confirm, err=True)
    if not password:
        raise CliError("the password must not be empty", ExitCode.USAGE)
    return password


def emit_json(document: Any) -> None:
    """Write one JSON document to stdout (strict JSON: no NaN/Infinity)."""
    sys.stdout.write(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()
