"""The live hub in isolation: slow clients, encode-once fan-out, history, resilience."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest
from starlette.datastructures import Headers

from puffer.api.hub import ClientChannel, LiveHub, origin_allowed
from puffer.beacon import canonical_summary, create_pulse
from puffer.beacon.pulse import GENESIS_PREV_OUTPUT
from puffer.beacon.vectors import vector_signing_key

from .conftest import FAST
from .fakes import FakeEngine


class FakeSocket:
    """A WebSocket whose writes can get stuck, like a peer that stopped reading."""

    def __init__(self, *, stuck_after: int | None = None, headers: dict[str, str] | None = None):
        self.headers = Headers(headers or {"host": "testserver"})
        self.stuck_after = stuck_after  # writes block forever once this many were sent
        self.sent: list[str] = []
        self.closed: tuple[int, str | None] | None = None
        self._incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._never = asyncio.Event()

    async def accept(self) -> None:
        pass

    def stall(self) -> None:
        """From now on every write blocks (the peer's receive window is full)."""
        self.stuck_after = len(self.sent)

    async def send_text(self, data: str) -> None:
        if self.stuck_after is not None and len(self.sent) >= self.stuck_after:
            await self._never.wait()
        self.sent.append(data)

    async def receive(self) -> dict[str, Any]:
        return await self._incoming.get()

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = (code, reason)

    def leave(self) -> None:
        self._incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

    def types(self) -> list[str]:
        return [json.loads(m)["type"] for m in self.sent]


def make_hub(engine: FakeEngine | None = None, **changes: Any) -> tuple[LiveHub, FakeEngine]:
    engine = engine or FakeEngine()
    hub = LiveHub(engine, replace(FAST, **changes), version="test", beacon_public_key=lambda: None)
    return hub, engine


async def run_clients(hub: LiveHub, sockets: list[FakeSocket], seconds: float) -> list[Any]:
    """Serve ``sockets`` for ``seconds``, then disconnect them; returns the handler tasks."""
    await hub.start()
    tasks = [asyncio.create_task(hub.serve(s)) for s in sockets]  # type: ignore[arg-type]
    await asyncio.sleep(seconds)
    return tasks


async def finish(hub: LiveHub, sockets: list[FakeSocket], tasks: list[Any]) -> None:
    for socket in sockets:
        socket.leave()
    await hub.stop()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)


# --------------------------------------------------------------------------
# ClientChannel
# --------------------------------------------------------------------------


def test_channel_keeps_only_the_newest_frame() -> None:
    async def scenario() -> ClientChannel:
        channel = ClientChannel(max_queue=10)
        channel.offer_frame("frame-1")
        channel.offer("entropy-1")
        channel.offer_frame("frame-2")
        channel.offer_frame("frame-3")
        channel.offer("health-1")
        got = [await channel.next_message() for _ in range(3)]
        assert got == ["entropy-1", "health-1", "frame-3"]  # ordered first, then newest frame
        return channel

    channel = asyncio.run(scenario())
    assert channel.dropped_frames == 2
    assert channel.pending == 0


def test_channel_overflow_requests_a_close() -> None:
    async def scenario() -> ClientChannel:
        channel = ClientChannel(max_queue=3)
        for i in range(4):
            channel.offer(f"m{i}")
        assert await channel.next_message() is None  # closed: the backlog is discarded
        channel.offer("late")  # ignored after close
        await asyncio.wait_for(channel.wait_closed(), timeout=1)
        return channel

    channel = asyncio.run(scenario())
    assert channel.close_request == (1013, "client too slow")


# --------------------------------------------------------------------------
# Slow clients
# --------------------------------------------------------------------------


async def wait_registered(hub: LiveHub, count: int) -> None:
    while hub.client_count < count:
        await asyncio.sleep(0.005)


def test_a_stuck_client_does_not_slow_down_the_others() -> None:
    hub, _ = make_hub(client_queue=1000)
    fast, stuck = FakeSocket(), FakeSocket()

    async def scenario() -> None:
        tasks = await run_clients(hub, [fast, stuck], 0.0)
        await wait_registered(hub, 2)
        stuck.stall()
        before = fast.types().count("frame")
        await asyncio.sleep(0.5)
        (stuck_channel,) = (c for c in hub._clients if c.sent <= 1)
        assert stuck_channel.dropped_frames >= 10  # stale frames are dropped...
        assert stuck_channel.pending <= stuck_channel.max_queue + 1  # ...never queued
        assert fast.types().count("frame") - before >= 10  # ~50 Hz meanwhile
        await finish(hub, [fast, stuck], tasks)

    asyncio.run(scenario())


def test_a_client_that_falls_too_far_behind_is_closed_with_1013() -> None:
    hub, _ = make_hub(client_queue=8)
    stuck = FakeSocket()

    async def scenario() -> None:
        await hub.start()
        task = asyncio.create_task(hub.serve(stuck))  # type: ignore[arg-type]
        await wait_registered(hub, 1)
        stuck.stall()
        await asyncio.wait_for(task, timeout=3)  # ends by itself, although stuck in send
        assert hub.client_count == 0
        await hub.stop()

    asyncio.run(scenario())
    assert stuck.closed == (1013, "client too slow")


def test_a_client_that_never_reads_the_handshake_is_dropped() -> None:
    hub, _ = make_hub(handshake_timeout_s=0.2)
    stuck = FakeSocket(stuck_after=0)

    async def scenario() -> None:
        await hub.start()
        await asyncio.wait_for(hub.serve(stuck), timeout=3)  # type: ignore[arg-type]
        assert hub.client_count == 0
        await hub.stop()

    asyncio.run(scenario())
    assert stuck.sent == []


def test_shutdown_closes_clients_with_1001() -> None:
    hub, _ = make_hub()
    socket = FakeSocket()

    async def scenario() -> None:
        await hub.start()
        task = asyncio.create_task(hub.serve(socket))  # type: ignore[arg-type]
        await asyncio.sleep(0.1)
        await hub.stop()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())
    assert socket.closed == (1001, "server shutting down")
    assert socket.types()[0] == "hello"


def test_a_pulse_published_during_the_handshake_arrives_once_after_hello() -> None:
    hub, _ = make_hub()
    socket = FakeSocket(stuck_after=0)  # the hello write blocks until we release it
    pulse = create_pulse(
        vector_signing_key(),
        index=0,
        timestamp_ms=1_793_454_000_000,
        period_ms=60_000,
        local_random=bytes(64),
        next_local_random=bytes(range(64)),
        prev_output=GENESIS_PREV_OUTPUT,
        summary=canonical_summary({"fish_bits": 0}),
    )

    async def scenario() -> None:
        await hub.start()
        task = asyncio.create_task(hub.serve(socket))  # type: ignore[arg-type]
        await wait_registered(hub, 1)  # registered while still writing hello
        hub.publish_pulse(pulse)
        await asyncio.sleep(0.05)
        socket.stuck_after = None
        socket._never.set()
        while "beacon" not in socket.types():
            await asyncio.sleep(0.005)
        await finish(hub, [socket], [task])

    asyncio.run(scenario())
    types = socket.types()
    assert types[0] == "hello"
    assert types.count("beacon") == 1


# --------------------------------------------------------------------------
# Encode once
# --------------------------------------------------------------------------


def test_each_frame_is_encoded_once_and_shared_by_all_clients() -> None:
    hub, engine = make_hub()
    sockets = [FakeSocket() for _ in range(3)]

    async def scenario() -> None:
        tasks = await run_clients(hub, sockets, 0.4)
        await finish(hub, sockets, tasks)

    asyncio.run(scenario())
    assert hub.frames_encoded >= 5
    assert engine.jpeg_calls == hub.frames_encoded  # one JPEG per broadcast frame
    frames = [[m for m in s.sent if m.startswith('{"type":"frame"')] for s in sockets]
    shared = set.intersection(*({id(m) for m in f} for f in frames))
    assert len(shared) >= 3  # literally the same str objects in every outbox


# --------------------------------------------------------------------------
# History and producers
# --------------------------------------------------------------------------


def test_history_is_a_time_bounded_ring_buffer() -> None:
    now = [100.0]
    engine = FakeEngine()
    hub = LiveHub(
        engine,
        replace(FAST, history_max_s=10.0, replay_entropy_s=5.0),
        version="t",
        beacon_public_key=lambda: None,
        clock=lambda: now[0],
    )
    for _ in range(30):  # one entry per simulated second
        hub._entropy_tick()
        now[0] += 1.0
    assert len(hub.history(600)) == 10  # capped at history_max_s
    recent = hub.history(3)
    assert len(recent) == 3
    assert [e["t"] for e in recent] == sorted(e["t"] for e in recent)
    assert len([m for m in hub.welcome() if '"entropy"' in m]) == 5  # replay_entropy_s


def test_unchanged_snapshots_add_no_history() -> None:
    engine = FakeEngine(auto_advance=False)
    hub, _ = make_hub(engine)
    for _ in range(5):
        hub._entropy_tick()
    assert len(hub.history(600)) == 1
    engine.advance()
    hub._entropy_tick()
    assert len(hub.history(600)) == 2


def test_a_failing_producer_does_not_stop_the_others(caplog: pytest.LogCaptureFixture) -> None:
    engine = FakeEngine()

    def broken() -> list[dict[str, Any]]:
        raise RuntimeError("database is locked")

    engine.fish_cards = broken  # type: ignore[method-assign]
    hub, _ = make_hub(engine)
    socket = FakeSocket()

    async def scenario() -> None:
        tasks = await run_clients(hub, [socket], 0.4)
        await finish(hub, [socket], tasks)

    asyncio.run(scenario())
    kinds = set(socket.types())
    assert {"hello", "frame", "entropy", "bits", "health", "led"} <= kinds
    assert "fish" not in kinds
    assert "live fish producer failed" in caplog.text


def test_engine_and_beacon_callbacks_before_start_are_ignored() -> None:
    hub, engine = make_hub()
    calls: list[int] = []
    hub._call_soon(lambda: calls.append(1))  # no loop yet: dropped, no exception
    hub._on_engine_snapshot(engine.advance())
    assert calls == []


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"host": "aquarium:8000"}, True),
        ({"host": "aquarium:8000", "origin": "http://aquarium:8000"}, True),
        ({"host": "aquarium:8000", "origin": "https://AQUARIUM:8000"}, True),
        ({"host": "aquarium:8000", "origin": "http://aquarium:8001"}, False),
        ({"host": "aquarium:8000", "origin": "http://127.0.0.1:5173/"}, True),
        ({"host": "aquarium:8000", "origin": "file://"}, False),
    ],
)
def test_origin_policy(headers: dict[str, str], expected: bool) -> None:
    allowed = ("http://localhost:5173", "http://127.0.0.1:5173")
    assert origin_allowed(Headers(headers), allowed) is expected
