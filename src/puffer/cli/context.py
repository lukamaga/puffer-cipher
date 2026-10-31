"""Per-invocation state shared by every command: global options, consoles, config.

The root callback stores one :class:`CliState` in the Typer context; commands
fetch it with :func:`get_state`. Two consoles keep the Unix contract:

* ``state.out`` (stdout) carries the *result* - a table, hex bytes, JSON - so
  ``puffer random | xxd`` and ``puffer keygen --json | jq`` work;
* ``state.err`` (stderr) carries everything for the human at the terminal:
  the live collection panel, progress, warnings and errors.

Configuration is loaded lazily, because half the commands (``decrypt -k``,
``beacon verify``, ``simulate``) never need it and must work even when the
configuration module or file is unavailable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from puffer.cli import wiring
from puffer.cli.protocols import ConfigLike

__all__ = ["PASSWORD_ENV", "CliState", "GlobalOptions", "LivenessMode", "get_state"]

PASSWORD_ENV = "PUFFER_PASSWORD"  # noqa: S105 - the variable's name, not a password
"""Environment variable holding the password for ``--password`` (scripts and tests)."""


class LivenessMode(enum.StrEnum):
    """``--liveness`` choices, mirroring ``PufferConfig.liveness`` (DESIGN.md 9.1)."""

    AUTO = "auto"
    REQUIRED = "required"
    ADVISORY = "advisory"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class GlobalOptions:
    """Options given before the command name; each overrides a ``PufferConfig`` field."""

    config_path: Path | None = None
    data_dir: Path | None = None
    source: str | None = None
    liveness: LivenessMode | None = None
    verbose: int = 0


class CliState:
    """Consoles plus the configuration with command-line overrides applied."""

    def __init__(self, options: GlobalOptions | None = None) -> None:
        self.options = options if options is not None else GlobalOptions()
        # highlight=False: no automatic colouring of numbers and paths - the
        # CLI chooses its colours deliberately (fish colours, health states).
        self.out = Console(highlight=False)
        self.err = Console(stderr=True, highlight=False)
        self._loaded: dict[Path | None, ConfigLike] = {}

    @property
    def verbose(self) -> int:
        return self.options.verbose

    def config(self, *, config_path: Path | None = None, **overrides: Any) -> ConfigLike:
        """The effective configuration for one command.

        Precedence, lowest to highest: dataclass defaults, TOML file, ``PUFFER_*``
        environment, global options (``--source`` before the command), then the
        command's own options passed here as ``overrides`` (``None`` = not given).
        """
        path = config_path if config_path is not None else self.options.config_path
        base = self._loaded.get(path)
        if base is None:
            base = self._loaded[path] = wiring.load_config(path)
        liveness = self.options.liveness.value if self.options.liveness is not None else None
        merged: dict[str, Any] = {
            "source": self.options.source,
            "liveness": liveness,
            "data_dir": self.options.data_dir,
        }
        merged.update({name: value for name, value in overrides.items() if value is not None})
        return wiring.apply_overrides(base, **merged)


def get_state(ctx: typer.Context) -> CliState:
    """The :class:`CliState` of this invocation (created on demand for direct calls)."""
    root = ctx.find_root()
    if not isinstance(root.obj, CliState):
        root.obj = CliState()
    return root.obj
