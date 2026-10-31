"""Fixtures for the CLI tests: isolated environment, fake tank, CLI invoker."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from puffer.cli import app, wiring

from .fakes import FakeConfig, FakeEngine, FakeFishStats

# Plain, wide, colourless output so assertions do not depend on the terminal.
BASE_ENV: dict[str, str | None] = {
    "COLUMNS": "160",
    "NO_COLOR": "1",
    "FORCE_COLOR": None,
    "TTY_COMPATIBLE": None,
    "TTY_INTERACTIVE": None,
}

Invoke = Callable[..., Result]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """No PUFFER_* variables from the developer's shell, cwd in tmp, logging restored."""
    for name in list(os.environ):
        if name.startswith("PUFFER_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@dataclass
class Tank:
    """Installs the fake config loader and engine factory; records what was built."""

    config: FakeConfig
    options: dict[str, Any] = field(default_factory=dict)
    engines: list[FakeEngine] = field(default_factory=list)
    configs: list[Any] = field(default_factory=list)
    config_paths: list[Path | None] = field(default_factory=list)
    error: BaseException | None = None

    def load(self, path: Path | None) -> FakeConfig:
        self.config_paths.append(path)
        return self.config

    def build(self, config: Any) -> FakeEngine:
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        engine = FakeEngine(config, **self.options)
        self.engines.append(engine)
        return engine

    @property
    def engine(self) -> FakeEngine:
        assert self.engines, "no engine was built"
        return self.engines[-1]


@pytest.fixture
def tank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Tank:
    fake = Tank(FakeConfig(data_dir=tmp_path / "data"))
    monkeypatch.setattr(wiring, "config_loader", fake.load)
    monkeypatch.setattr(wiring, "engine_factory", fake.build)
    return fake


@pytest.fixture
def fish_stats(monkeypatch: pytest.MonkeyPatch) -> FakeFishStats:
    stats = FakeFishStats()
    opened: list[Path] = []

    def factory(path: Path) -> FakeFishStats:
        opened.append(path)
        return stats

    monkeypatch.setattr(wiring, "fish_stats_factory", factory)
    stats.opened = opened  # type: ignore[attr-defined]
    return stats


@pytest.fixture
def cli() -> Invoke:
    runner = CliRunner()

    def invoke(*args: str, input: str | None = None, env: dict[str, str] | None = None) -> Result:
        return runner.invoke(app, list(args), input=input, env={**BASE_ENV, **(env or {})})

    return invoke
