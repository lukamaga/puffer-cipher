"""``puffer.api.server``: start order, beacon thread and shutdown, with the runtime faked."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from nacl.signing import SigningKey

from puffer.api import server
from puffer.beacon import BeaconService, BeaconStore
from puffer.config import PufferConfig

from .fakes import FakeEngine


class LifecycleEngine(FakeEngine):
    """A fake engine that records start/close in a shared event log."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    def start(self) -> None:
        self.log.append("engine.start")

    def close(self) -> None:
        self.log.append("engine.close")


class RecordingStore(BeaconStore):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    def close(self) -> None:
        self.log.append("store.close")
        super().close()


@pytest.fixture
def log() -> list[str]:
    return []


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> dict[str, Any]:
    """Replace ``puffer.runtime`` (imported lazily by ``running``) and ``uvicorn.run``."""
    made: dict[str, Any] = {}

    def build_engine(config: PufferConfig) -> LifecycleEngine:
        made["engine"] = LifecycleEngine(log)
        return made["engine"]

    def build_beacon(config: PufferConfig, engine: LifecycleEngine) -> BeaconService:
        def entropy(n: int, purpose: str) -> tuple[bytes, dict[str, Any]]:
            return engine.rng.generate(n, purpose)[0], {"fish_bits": 0}

        made["beacon"] = BeaconService(
            RecordingStore(log), SigningKey(bytes(32)), entropy, period_ms=60_000
        )
        return made["beacon"]

    runtime = types.ModuleType("puffer.runtime")
    runtime.build_engine = build_engine  # type: ignore[attr-defined]
    runtime.build_beacon = build_beacon  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "puffer.runtime", runtime)

    def fake_uvicorn_run(app: Any, **kwargs: Any) -> None:
        threads = {t.name for t in threading.enumerate()}
        log.append("uvicorn.run" + (" +beacon" if "puffer-beacon" in threads else ""))
        made["uvicorn"] = kwargs
        if made.get("raise"):
            raise made["raise"]

    monkeypatch.setattr(server.uvicorn, "run", fake_uvicorn_run)
    return made


def test_serve_starts_engine_and_beacon_then_shuts_down_in_reverse(
    fake_runtime: dict[str, Any], log: list[str], tmp_path: Path
) -> None:
    config = PufferConfig(host="127.0.0.1", port=8765, static_dir=tmp_path, data_dir=tmp_path)
    server.serve(config)
    assert log == ["engine.start", "uvicorn.run +beacon", "engine.close", "store.close"]
    options = fake_runtime["uvicorn"]
    assert (options["host"], options["port"]) == ("127.0.0.1", 8765)
    assert options["ws_max_size"] == 64 * 1024
    assert options["server_header"] is False
    assert not any(t.name == "puffer-beacon" for t in threading.enumerate())


def test_serve_without_beacon(fake_runtime: dict[str, Any], log: list[str], tmp_path: Path) -> None:
    server.serve(PufferConfig(beacon_enabled=False, static_dir=tmp_path, data_dir=tmp_path))
    assert log == ["engine.start", "uvicorn.run", "engine.close"]
    assert "beacon" not in fake_runtime


def test_cleanup_runs_when_the_server_fails(
    fake_runtime: dict[str, Any], log: list[str], tmp_path: Path
) -> None:
    fake_runtime["raise"] = OSError("address already in use")
    with pytest.raises(OSError, match="address already in use"):
        server.serve(PufferConfig(static_dir=tmp_path, data_dir=tmp_path))
    assert log[-2:] == ["engine.close", "store.close"]


def test_running_yields_an_app_wired_to_the_engine_and_beacon(
    fake_runtime: dict[str, Any], tmp_path: Path
) -> None:
    with server.running(PufferConfig(static_dir=tmp_path, data_dir=tmp_path)) as app:
        assert app.state.settings.frame_hz == 10.0  # from config.stream_fps
        assert app.state.settings.pool_capacity_bits == 4096.0


def test_a_crashing_beacon_is_logged_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        def run(self, stop: threading.Event) -> None:
            raise RuntimeError("stored pending reveal is inconsistent")

    server._run_beacon(Broken(), threading.Event())  # type: ignore[arg-type]
    assert "beacon stopped" in caplog.text


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("synthetic", "synthetic:realtime=1"),
        ("synthetic:seed=7", "synthetic:seed=7,realtime=1"),
        ("synthetic:seed=7,realtime=0", "synthetic:seed=7,realtime=0"),
        ("camera:0", "camera:0"),
        ("video:tank.mp4", "video:tank.mp4"),
    ],
)
def test_serve_paces_only_the_simulator(spec: str, expected: str) -> None:
    from puffer.api.server import live_source_spec

    assert live_source_spec(spec) == expected
