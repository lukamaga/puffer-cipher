"""The seam between the CLI and the heavy parts of puffer.

Everything the CLI needs from the engine, the configuration, the fish
statistics, the API server and the analysis report goes through the
functions in this module, for three reasons:

* **Lazy imports.** ``puffer --help``, ``puffer beacon verify`` or
  ``puffer decrypt`` must not pay for importing OpenCV, the tracker and
  FastAPI, and must keep working when an optional part (the YOLO detector,
  the report builder) is not installed. Each import happens inside the
  function that needs it and failures become a friendly
  :class:`~puffer.cli.errors.CliError` with exit code
  :attr:`~puffer.cli.errors.ExitCode.UNAVAILABLE`.
* **Test hooks.** ``engine_factory``, ``config_loader`` and
  ``fish_stats_factory`` are module-level hooks. Unit tests set them (with
  ``monkeypatch``) to factories of fakes; ``None`` means "use the real
  implementation". No production code path ever sets them.
* **One place to adapt.** If an interface of DESIGN.md 9 changes, only this
  file changes; the commands only see :mod:`puffer.cli.protocols`.
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from puffer.cli.errors import CliError, ExitCode
from puffer.cli.protocols import ConfigLike, EngineLike, FishStatsLike
from puffer.hardware import LedHardwareError

__all__ = [
    "ConfigLoader",
    "EngineFactory",
    "FishStatsFactory",
    "apply_overrides",
    "build_engine",
    "config_loader",
    "engine_factory",
    "fish_stats_factory",
    "load_config",
    "open_fish_stats",
    "run_report",
    "serve",
]

EngineFactory = Callable[[ConfigLike], EngineLike]
ConfigLoader = Callable[[Path | None], ConfigLike]
FishStatsFactory = Callable[[Path], FishStatsLike]

engine_factory: EngineFactory | None = None
"""Test hook: builds the engine from a config instead of ``puffer.runtime.build_engine``."""

config_loader: ConfigLoader | None = None
"""Test hook: loads the configuration instead of ``puffer.config.PufferConfig``."""

fish_stats_factory: FishStatsFactory | None = None
"""Test hook: opens the fish statistics database instead of ``puffer.fishstats.FishStats``."""


_EXTRAS = {"ultralytics": "yolo", "gpiozero": "pi", "matplotlib": "analysis"}
"""Optional packages and the ``pyproject.toml`` extra that installs each."""

_C = TypeVar("_C")


def _unavailable(exc: ImportError, module: str, what: str) -> CliError:
    """Explain an import failure: our own module missing vs. a missing dependency."""
    missing = exc.name or ""
    if isinstance(exc, ModuleNotFoundError) and (
        missing == module or module.startswith(missing + ".")
    ):
        return CliError(
            f"{what} is not available in this installation ({module} is missing)",
            ExitCode.UNAVAILABLE,
        )
    if isinstance(exc, ModuleNotFoundError) and missing:
        extra = _EXTRAS.get(missing.partition(".")[0])
        return CliError(
            f"{what} needs the Python package {missing!r}, which is not installed",
            ExitCode.UNAVAILABLE,
            hint=f"pip install 'puffer-cipher[{extra}]'" if extra else f"pip install {missing}",
        )
    return CliError(f"{what} could not be loaded: {exc}", ExitCode.UNAVAILABLE)


_OWN_ENV = frozenset({"PUFFER_PASSWORD"})
"""``PUFFER_*`` variables owned by the CLI, not by ``PufferConfig`` (never passed to it)."""


def _config_env() -> dict[str, str]:
    """``os.environ`` without the CLI's own variables.

    ``PUFFER_PASSWORD`` shares the prefix of configuration overrides but is
    not a setting: the password must never flow into config parsing (where it
    would at best trigger an "unknown setting" warning).
    """
    return {name: value for name, value in os.environ.items() if name not in _OWN_ENV}


def load_config(path: Path | None) -> ConfigLike:
    """Load ``PufferConfig`` from ``path`` (TOML ``[puffer]`` table) and ``PUFFER_*`` env vars."""
    if config_loader is not None:
        return config_loader(path)
    try:
        from puffer.config import PufferConfig
    except ImportError as exc:
        raise _unavailable(exc, "puffer.config", "the configuration module") from exc
    try:
        config: ConfigLike = PufferConfig.load(path, env=_config_env())
    except ValueError as exc:  # ConfigError: the message names the field and its origin
        raise CliError(f"configuration error: {exc}", ExitCode.USAGE) from exc
    return config


def apply_overrides(config: ConfigLike, **changes: Any) -> ConfigLike:
    """Return ``config`` with the non-``None`` ``changes`` applied (command-line overrides).

    ``PufferConfig`` is a dataclass, so :func:`dataclasses.replace` re-runs its
    validation: an invalid override (``--port 0``) is reported like an invalid
    config file instead of surfacing later inside the engine.
    """
    updates = {name: value for name, value in changes.items() if value is not None}
    if not updates:
        return config
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"cannot override fields of {type(config).__name__}")
    try:
        updated: ConfigLike = dataclasses.replace(config, **updates)
    except (ValueError, TypeError) as exc:
        raise CliError(f"invalid option value: {exc}", ExitCode.USAGE) from exc
    return updated


def build_engine(config: ConfigLike) -> EngineLike:
    """Wire a concrete engine for ``config`` (``puffer.runtime.build_engine``)."""
    if engine_factory is not None:
        return engine_factory(config)
    try:
        from puffer.config import PufferConfig
        from puffer.runtime import build_engine as real_build_engine
    except ImportError as exc:
        raise _unavailable(exc, "puffer.runtime", "the entropy engine") from exc
    try:
        engine: EngineLike = real_build_engine(_real_config(config, PufferConfig))
    except LedHardwareError as exc:
        raise CliError(
            f"the liveness LED is not available: {exc}",
            ExitCode.UNAVAILABLE,
            hint="set led = none (and liveness = advisory or off), or install gpiozero",
        ) from exc
    except ImportError as exc:  # e.g. detector = "yolo:..." without ultralytics
        raise _unavailable(exc, "puffer.runtime", "the entropy engine") from exc
    return engine


def _real_config(config: ConfigLike, cls: type[_C]) -> _C:
    """Narrow a :class:`ConfigLike` to the real ``PufferConfig`` the runtime expects.

    Only test hooks ever produce other config types, and they replace the
    functions that call this; anything else reaching here is a bug.
    """
    if not isinstance(config, cls):
        raise TypeError(f"expected {cls.__name__}, got {type(config).__name__}")
    return config


def open_fish_stats(db_path: Path) -> FishStatsLike:
    """Open the fish statistics stored in ``db_path`` (``data_dir/puffer.db``)."""
    if fish_stats_factory is not None:
        return fish_stats_factory(db_path)
    try:
        from puffer.fishstats import FishStats, FishStatsError
    except ImportError as exc:
        raise _unavailable(exc, "puffer.fishstats", "fish statistics") from exc
    try:
        stats: FishStatsLike = FishStats(db_path)
    except (FishStatsError, sqlite3.DatabaseError) as exc:
        raise CliError(
            f"cannot read fish statistics from {db_path}: {exc}", ExitCode.FORMAT
        ) from exc
    return stats


def serve(config: ConfigLike) -> None:
    """Run engine + beacon + API + dashboard until interrupted (``puffer.api.server.serve``)."""
    try:
        from puffer.api.server import serve as real_serve
        from puffer.config import PufferConfig
    except ImportError as exc:
        raise _unavailable(exc, "puffer.api.server", "the API server") from exc
    real_serve(_real_config(config, PufferConfig))


def run_report(
    *,
    sources: list[str],
    frames: int,
    out_dir: Path,
    media_dir: Path,
    nist_sequences: int,
    dieharder: bool,
    seed: int | None,
    progress: Callable[[str], None],
) -> Path:
    """Build the DESIGN.md section 10 statistics report; return the path of ``RESULTS.md``."""
    try:
        from puffer.analysis.report import ReportConfig
        from puffer.analysis.report import run_report as real_run_report
    except ImportError as exc:
        raise _unavailable(exc, "puffer.analysis.report", "the analysis report") from exc
    report_config = ReportConfig(
        sources=sources,
        frames=frames,
        out_dir=out_dir,
        media_dir=media_dir,
        nist_sequences=nist_sequences,
        dieharder=dieharder,
        seed=seed,
    )
    return Path(real_run_report(report_config, progress=progress))
