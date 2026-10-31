"""Fixtures for the API tests: a fake engine, a real in-memory beacon, test clients."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from nacl.signing import SigningKey

from puffer.api import ApiSettings, create_app
from puffer.beacon import BeaconService, BeaconStore, summary_from_receipt

from .dashboard_protocol import parse_server_message
from .fakes import FakeEngine

with warnings.catch_warnings():
    # Starlette suggests httpx2 for its TestClient; httpx is what the project pins.
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

FAST = ApiSettings(
    frame_hz=50.0,
    entropy_hz=50.0,
    bits_hz=50.0,
    health_hz=50.0,
    led_hz=50.0,
    fish_hz=20.0,
    hello_wait_s=0.2,
)
"""Every stream at >= 20 Hz so tests see each message type within milliseconds."""

BEACON_T0_MS = 1_793_454_000_000
BEACON_PERIOD_MS = 60_000


class ManualClock:
    """Millisecond clock for the beacon, advanced by the test."""

    def __init__(self, now_ms: int = BEACON_T0_MS) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def beacon_clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def beacon(engine: FakeEngine, beacon_clock: ManualClock) -> Iterator[BeaconService]:
    """A real beacon over an in-memory store, drawing from the fake engine's real RNG."""

    def entropy(n: int, purpose: str) -> tuple[bytes, dict[str, Any]]:
        value, receipt = engine.rng.inner.generate(n, purpose)
        return value, summary_from_receipt(receipt.to_dict())

    with BeaconStore() as store:
        yield BeaconService(
            store,
            SigningKey(bytes(range(32))),  # deterministic test key, never a real beacon key
            entropy,
            period_ms=BEACON_PERIOD_MS,
            clock_ms=beacon_clock,
        )


ClientFactory = Callable[..., TestClient]


@pytest.fixture
def make_client(engine: FakeEngine) -> Iterator[ClientFactory]:
    """``make_client(**create_app_kwargs)``: a started TestClient (lifespan running)."""
    clients: list[TestClient] = []

    def factory(**kwargs: Any) -> TestClient:
        kwargs.setdefault("settings", FAST)
        target = kwargs.pop("engine", engine)
        client = TestClient(create_app(target, **kwargs))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in reversed(clients):
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    return make_client()


def receive_until(ws: Any, message_type: str, *, limit: int = 2000) -> dict[str, Any]:
    """Read messages until one of ``message_type`` arrives; every message must decode."""
    for _ in range(limit):
        raw = ws.receive_text()
        decoded_type, _ = parse_server_message(raw)
        message: dict[str, Any] = json.loads(raw)
        assert decoded_type == message["type"]
        if message["type"] == message_type:
            return message
    raise AssertionError(f"no {message_type!r} message within {limit} messages")
