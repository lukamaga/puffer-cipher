"""``puffer serve`` and ``puffer analyze`` delegate to lazily imported modules."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from puffer.config import PufferConfig

from .conftest import Invoke


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> list[PufferConfig]:
    served: list[PufferConfig] = []
    module = types.ModuleType("puffer.api.server")
    module.serve = served.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "puffer.api.server", module)
    return served


def test_serve_applies_overrides_to_the_real_configuration(
    cli: Invoke, fake_server: list[PufferConfig]
) -> None:
    Path("puffer.toml").write_text('[puffer]\nport = 9000\nsource = "synthetic:seed=3"\n')

    result = cli("--liveness", "advisory", "serve", "--config", "puffer.toml", "--host", "0.0.0.0")

    assert result.exit_code == 0, result.output
    [config] = fake_server
    assert isinstance(config, PufferConfig)
    assert (config.host, config.port, config.source, config.liveness) == (
        "0.0.0.0",
        9000,
        "synthetic:seed=3",
        "advisory",
    )
    assert "http://0.0.0.0:9000" in result.stderr


def test_serve_reports_invalid_configuration(cli: Invoke, fake_server: list[Any]) -> None:
    Path("bad.toml").write_text("[puffer]\nprot = 1\n")

    result = cli("serve", "--config", "bad.toml")

    assert result.exit_code == 2
    assert "unknown setting 'prot' (did you mean 'port'?)" in result.stderr
    assert fake_server == []


def test_serve_without_the_api_module(cli: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "puffer.api.server", None)

    result = cli("serve")

    assert result.exit_code == 9
    assert "the API server is not available" in result.stderr


@dataclass
class _ReportConfig:
    sources: list[str]
    frames: int = 3000
    out_dir: Path = Path("docs/results")
    media_dir: Path = Path("docs/media")
    nist_sequences: int = 10
    dieharder: bool = False
    seed: int | None = None


@dataclass
class _Report:
    configs: list[_ReportConfig] = field(default_factory=list)

    def run_report(self, config: _ReportConfig, *, progress: Any = None) -> Path:
        self.configs.append(config)
        progress("SP 800-90B: sensor channel")
        return config.out_dir / "RESULTS.md"


@pytest.fixture
def fake_report(monkeypatch: pytest.MonkeyPatch) -> _Report:
    report = _Report()
    module = types.ModuleType("puffer.analysis.report")
    module.ReportConfig = _ReportConfig  # type: ignore[attr-defined]
    module.run_report = report.run_report  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "puffer.analysis.report", module)
    return report


def test_analyze_defaults_to_simulator_day_and_night(cli: Invoke, fake_report: _Report) -> None:
    result = cli("analyze", "--frames", "100", "--out", "res", "--dieharder", "--seed", "4")

    assert result.exit_code == 0, result.output
    [config] = fake_report.configs
    assert config.sources == ["synthetic:seed=1,mode=day", "synthetic:seed=1,mode=night"]
    assert (config.frames, config.out_dir, config.dieharder, config.seed) == (
        100,
        Path("res"),
        True,
        4,
    )
    assert "Report written to res/RESULTS.md" in result.stdout
    assert "SP 800-90B: sensor channel" in result.stderr


def test_analyze_sources(cli: Invoke, fake_report: _Report) -> None:
    assert cli("--source", "video:tank.mkv", "analyze").exit_code == 0
    assert cli("analyze", "--source", "a", "--source", "b").exit_code == 0

    assert [c.sources for c in fake_report.configs] == [["video:tank.mkv"], ["a", "b"]]


def test_analyze_unavailable_is_friendly(cli: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "puffer.analysis.report", None)

    result = cli("analyze")

    assert result.exit_code == 9
    assert "the analysis report is not available in this installation" in result.stderr


def test_analyze_missing_dependency_names_the_extra(
    cli: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Finder:
        """Makes ``import puffer.analysis.report`` fail on a missing matplotlib."""

        def find_spec(self, name: str, path: object = None, target: object = None) -> None:
            if name == "puffer.analysis.report":
                raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")

    monkeypatch.delitem(sys.modules, "puffer.analysis.report", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Finder(), *sys.meta_path])

    result = cli("analyze")

    assert result.exit_code == 9
    assert "needs the Python package 'matplotlib'" in result.stderr
    assert "pip install 'puffer-cipher[analysis]'" in result.stderr
