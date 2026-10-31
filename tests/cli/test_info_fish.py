"""``puffer info`` and ``puffer fish``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from puffer.cli import wiring
from puffer.cli.commands.info import libsodium_version
from puffer.cli.errors import CliError, ExitCode
from puffer.fishstats import FishStats

from .conftest import Invoke, Tank
from .fakes import FakeFishStats


def test_info_json(cli: Invoke, tank: Tank) -> None:
    result = cli("--data-dir", "d", "info", "--json")

    assert result.exit_code == 0, result.output
    info = json.loads(result.stdout)
    assert info["puffer"]["version"]
    assert info["libraries"]["libsodium"] == libsodium_version()
    assert info["libraries"]["pynacl"]
    assert set(info["components"]) >= {"engine", "api_server", "analysis_report"}
    assert set(info["tools"]) == {"dieharder", "ent"}
    assert info["config"]["source"] == "synthetic:seed=7"
    assert info["config"]["data_dir"] == "d"
    assert info["data_dir"] == {
        "path": str(Path("d").resolve()),
        "exists": False,
        "database": False,
        "beacon_key": False,
    }


def test_info_human(cli: Invoke, tank: Tank) -> None:
    Path("data").mkdir()
    (Path("data") / "beacon.key").write_text("secret, never printed")

    result = cli("--data-dir", "data", "info")

    assert result.exit_code == 0, result.output
    for word in ("libsodium", "opencv", "components and tools", "configuration", "data directory"):
        assert word in result.stdout
    assert "beacon.key: present" in result.stdout
    assert "never printed" not in result.stdout


def test_info_survives_a_broken_configuration(cli: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(path: Path | None) -> None:
        raise CliError("configuration error: port: expected an integer", ExitCode.USAGE)

    monkeypatch.setattr(wiring, "config_loader", broken)

    result = cli("info", "--json")

    assert result.exit_code == 0, result.output
    info = json.loads(result.stdout)
    assert info["config"] is None
    assert "expected an integer" in info["config_error"]


def test_libsodium_version_is_never_guessed() -> None:
    version = libsodium_version()
    assert version.startswith("bundled with PyNaCl") or version[0].isdigit()


# -------------------------------------------------------------------- fish


def test_fish_without_statistics(cli: Invoke, tank: Tank, fish_stats: FakeFishStats) -> None:
    result = cli("fish")

    assert result.exit_code == 0, result.output
    assert "No fish statistics" in result.stdout
    assert fish_stats.opened == []  # type: ignore[attr-defined]  # no empty DB is created
    assert not (tank.config.data_dir / "puffer.db").exists()


def test_fish_leaderboard(cli: Invoke, tank: Tank, fish_stats: FakeFishStats) -> None:
    tank.config.data_dir.mkdir()
    (tank.config.data_dir / "puffer.db").write_bytes(b"")
    fish_stats.rows = [
        {"rank": 1, "fish_id": "f7", "name": "Zap", "chaos_score": 0.83, "bits": 9123.5},
        {"rank": 2, "fish_id": "f3", "name": "[red]Puff[/red]", "chaos_score": None, "bits": 12.0},
    ]

    result = cli("fish", "--days", "3")

    assert result.exit_code == 0, result.output
    assert fish_stats.days == [3]
    assert fish_stats.closed
    out = result.stdout
    assert "Fish leaderboard - last 3 day(s)" in out
    assert "Zap" in out
    assert "0.83" in out
    assert "9,123.5" in out
    assert "[red]Puff[/red]" in out  # names are data, never Rich markup

    result = cli("fish", "--json")
    assert json.loads(result.stdout)[0]["fish_id"] == "f7"


def test_fish_reads_a_real_statistics_database(cli: Invoke, tank: Tank) -> None:
    data = tank.config.data_dir
    data.mkdir()
    now = time.time()
    with FishStats(data / "puffer.db") as stats:
        stats.record("f1", t=0.0, wall_time=now, name="Biscuit", bits=40.0, chaos_score=0.9)
        stats.record("f2", t=0.1, wall_time=now, name="Sunny", bits=10.0, chaos_score=0.2)

    result = cli("fish", "--json")

    assert result.exit_code == 0, result.output
    board = json.loads(result.stdout)
    assert [row["name"] for row in board] == ["Biscuit", "Sunny"]


def test_fish_reports_an_unreadable_database(cli: Invoke, tank: Tank) -> None:
    data = tank.config.data_dir
    data.mkdir()
    (data / "puffer.db").write_bytes(b"this is not sqlite" * 100)

    result = cli("fish")

    assert result.exit_code == 7
    assert "cannot read fish statistics" in result.stderr
