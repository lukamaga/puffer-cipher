"""The live feed ``/ws/live``: one broadcaster, many clients (DESIGN.md 7.2).

Shape of the data flow::

    engine thread ──subscribe──► call_soon_threadsafe ──► frame loop ─┐
    beacon thread ──on_pulse───► call_soon_threadsafe ──► broadcast   │  pre-encoded str
    periodic loops (entropy 2 Hz, bits 5 Hz, health 1 Hz, led 2 Hz,   ├──► ClientChannel ─► socket
                    fish 0.2 Hz) ─────────────────────────────────────┘    (one per client)

Design decisions:

* **Encode once, fan out strings.** Each message is serialised exactly once
  per tick - for a frame that includes the JPEG encode and its base64 - and
  the same ``str`` is handed to every client. Cost per extra viewer is a
  queue append, not an image encode.
* **Nothing a client does can block the hub.** Broadcasting is a
  synchronous, non-blocking append to each client's :class:`ClientChannel`;
  a per-client task does the actual (awaiting) socket writes. Frames use a
  *latest-wins* slot: a client that cannot keep up silently skips stale
  frames instead of building a backlog. Everything else (entropy, health,
  beacon...) is ordered and small; if even that queue overflows, the client
  is closed with 1013 "try again later" and reconnects to a fresh state.
* **Blocking work stays off the loop.** JPEG encoding and ``fish_cards``
  (SQLite) run in worker threads; frames are only encoded while somebody is
  watching.
* **``hello`` is always first, and nothing is lost behind it.** A client is
  registered at the same instant its welcome is assembled (``hello``, the
  recent ``entropy`` history, the latest state messages - so a new dashboard
  is populated at once instead of after the 5 s ``fish`` period). The welcome
  is written directly; broadcasts meanwhile wait in the client's outbox,
  which is drained only afterwards. A pulse committed mid-handshake is
  therefore delivered exactly once.
* **Threads talk to the loop only through ``call_soon_threadsafe``.**
  Engine and beacon callbacks never touch asyncio objects directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from dataclasses import dataclass
from typing import Any, Final, Protocol
from urllib.parse import urlsplit

import anyio
from starlette.datastructures import Headers
from starlette.websockets import WebSocket

from puffer.api import wire
from puffer.api.ports import EngineLike, SnapshotLike
from puffer.api.settings import ApiSettings
from puffer.beacon.pulse import Pulse

logger = logging.getLogger(__name__)

__all__ = ["ClientChannel", "LiveHub", "origin_allowed"]

CLOSE_GOING_AWAY: Final = 1001
CLOSE_POLICY_VIOLATION: Final = 1008
CLOSE_TRY_AGAIN_LATER: Final = 1013
_CLOSE_TIMEOUT_S: Final = 2.0
_STATE_TYPES: Final = ("health", "led", "fish", "bits")
"""Message types whose latest instance is replayed to a new client (after ``entropy``)."""


class TextSocket(Protocol):
    """The part of a WebSocket a :class:`ClientChannel` writes to."""

    async def send_text(self, data: str) -> None: ...


class ClientChannel:
    """One client's outbox: an ordered bounded queue plus a latest-wins frame slot.

    All methods except :meth:`pump` are synchronous and never block, so the
    hub can broadcast from the event loop without awaiting any client.
    """

    def __init__(self, *, max_queue: int) -> None:
        self.max_queue = max_queue
        self.sent = 0
        self.dropped_frames = 0
        self._queue: deque[str] = deque()
        self._frame: str | None = None
        self._close: tuple[int, str] | None = None
        self._wake = asyncio.Event()
        self._closed = asyncio.Event()

    @property
    def close_request(self) -> tuple[int, str] | None:
        """``(code, reason)`` once the server decided to close this client."""
        return self._close

    @property
    def pending(self) -> int:
        """Messages waiting to be written (queued plus the frame slot)."""
        return len(self._queue) + (self._frame is not None)

    def offer(self, message: str) -> None:
        """Queue an ordered message; a client that is this far behind gets closed."""
        if self._close is not None:
            return
        if len(self._queue) >= self.max_queue:
            self.close(CLOSE_TRY_AGAIN_LATER, "client too slow")
            return
        self._queue.append(message)
        self._wake.set()

    def offer_frame(self, message: str) -> None:
        """Replace any unsent frame: slow clients drop stale frames, never queue them."""
        if self._close is not None:
            return
        if self._frame is not None:
            self.dropped_frames += 1
        self._frame = message
        self._wake.set()

    def close(self, code: int, reason: str) -> None:
        """Ask :meth:`pump` to stop; unsent messages are discarded."""
        if self._close is None:
            self._close = (code, reason)
            self._wake.set()
            self._closed.set()

    async def wait_closed(self) -> None:
        """Return once :meth:`close` was called (even while :meth:`pump` is stuck writing)."""
        await self._closed.wait()

    async def next_message(self) -> str | None:
        """The next message to write, or ``None`` once a close was requested."""
        while True:
            if self._close is not None:
                return None
            if self._queue:
                return self._queue.popleft()
            if self._frame is not None:
                message, self._frame = self._frame, None
                return message
            self._wake.clear()
            await self._wake.wait()

    async def pump(self, socket: TextSocket) -> None:
        """Write messages until a close is requested (or the socket raises)."""
        while (message := await self.next_message()) is not None:
            await socket.send_text(message)
            self.sent += 1


def origin_allowed(headers: Headers, allowed: tuple[str, ...]) -> bool:
    """Same-origin or explicitly allowed browser origin (blocks cross-site WebSocket hijacking).

    Browsers do not apply CORS to WebSockets, so without this check any web
    page could open the live feed through a visitor's browser. Requests
    without an ``Origin`` header come from non-browser clients, which the
    check cannot and need not stop.
    """
    origin = headers.get("origin")
    if origin is None:
        return True
    host = headers.get("host", "")
    if host and urlsplit(origin).netloc.lower() == host.lower():
        return True
    normalized = origin.rstrip("/").lower()
    return any(normalized == candidate.rstrip("/").lower() for candidate in allowed)


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    recorded_at: float  # hub clock (monotonic), independent of the frame clock
    data: dict[str, Any]
    text: str


class _FailureLog:
    """Logs a failing producer on its 1st, 2nd, 4th, 8th... failure, not ten times a second."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def __call__(self, name: str) -> None:
        count = self._counts.get(name, 0) + 1
        self._counts[name] = count
        if count & (count - 1) == 0 or count % 1000 == 0:  # 1, 2, 4, 8, ... then every 1000
            logger.exception("live %s producer failed (%d times)", name, count)


class LiveHub:
    """Produces the ``/ws/live`` messages from engine snapshots and fans them out.

    ``start``/``stop`` are called from the app's lifespan; ``serve`` runs one
    connection; ``publish_pulse`` is a :meth:`BeaconService.on_pulse
    <puffer.beacon.service.BeaconService.on_pulse>` listener; ``history``
    backs ``GET /api/entropy/history``.
    """

    def __init__(
        self,
        engine: EngineLike,
        settings: ApiSettings,
        *,
        version: str,
        beacon_public_key: Callable[[], str | None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._settings = settings
        self._version = version
        self._beacon_public_key = beacon_public_key
        self._clock = clock
        self._clients: set[ClientChannel] = set()
        self._history: deque[_HistoryEntry] = deque(
            maxlen=int(settings.history_max_s * settings.entropy_hz) + 8
        )
        self._last: dict[str, str] = {}
        self._identities: dict[str, tuple[str, str | None]] = {}
        self._last_entropy_t: float | None = None
        self._last_fish_at = -float("inf")
        self._tasks: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._snapshot_ready = asyncio.Event()
        self._first_snapshot = asyncio.Event()
        self._fish_wake = asyncio.Event()
        self._unsubscribe: Callable[[], None] | None = None
        self._closing = False
        self._failed = _FailureLog()
        self.frames_encoded = 0

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the engine and start the producer loops (call once, from the loop)."""
        if self._tasks:
            raise RuntimeError("LiveHub is already running")
        self._closing = False
        self._loop = asyncio.get_running_loop()
        # Fresh events for this loop (an app may be started by several test clients).
        self._snapshot_ready = asyncio.Event()
        self._first_snapshot = asyncio.Event()
        self._fish_wake = asyncio.Event()
        if self._engine.snapshot() is not None:
            self._first_snapshot.set()
        self._unsubscribe = self._engine.subscribe(self._on_engine_snapshot)
        s = self._settings
        producers: dict[str, Coroutine[Any, Any, None]] = {
            "frame": self._frame_loop(),
            "fish": self._fish_loop(),
            "entropy": self._every(1.0 / s.entropy_hz, "entropy", self._entropy_tick),
            "bits": self._every(1.0 / s.bits_hz, "bits", self._bits_tick),
            "health": self._every(1.0 / s.health_hz, "health", self._health_tick),
            "led": self._every(1.0 / s.led_hz, "led", self._led_tick),
        }
        self._tasks = [
            asyncio.create_task(coro, name=f"puffer-live-{name}")
            for name, coro in producers.items()
        ]

    async def stop(self) -> None:
        """Stop producing, unsubscribe and ask every client to close (1001)."""
        self._closing = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        for client in tuple(self._clients):
            client.close(CLOSE_GOING_AWAY, "server shutting down")
        self._loop = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # -- thread entry points ---------------------------------------------------

    def _on_engine_snapshot(self, _snapshot: SnapshotLike) -> None:
        """Engine-thread callback: only wakes the frame loop."""
        self._call_soon(self._snapshot_arrived)

    def publish_pulse(self, pulse: Pulse) -> None:
        """Beacon-thread listener: broadcast a committed pulse, byte-identical to REST."""
        message = wire.raw_envelope("beacon", wire.pulse_text(pulse))
        self._call_soon(self._broadcast, message)

    def _call_soon(self, callback: Callable[..., None], *args: Any) -> None:
        loop = self._loop
        if loop is None or self._closing:
            return
        with contextlib.suppress(RuntimeError):  # loop already closed during shutdown
            loop.call_soon_threadsafe(callback, *args)

    def _snapshot_arrived(self) -> None:
        self._snapshot_ready.set()
        self._first_snapshot.set()

    # -- broadcasting ----------------------------------------------------------

    def _broadcast(self, message: str, *, cache_as: str | None = None) -> None:
        if cache_as is not None:
            self._last[cache_as] = message
        for client in tuple(self._clients):
            client.offer(message)

    def _broadcast_frame(self, message: str) -> None:
        for client in tuple(self._clients):
            client.offer_frame(message)

    # -- producers ---------------------------------------------------------------

    async def _every(self, period: float, name: str, tick: Callable[[], None]) -> None:
        """Run ``tick`` every ``period`` seconds; never bursts to catch up after a stall."""
        loop = asyncio.get_running_loop()
        due = loop.time()
        while True:
            try:
                tick()
            except Exception:
                self._failed(name)
            due += period
            delay = due - loop.time()
            if delay < 0:
                due, delay = loop.time(), 0.0
            await asyncio.sleep(delay)

    async def _frame_loop(self) -> None:
        """Send each new engine frame, at most ``frame_hz`` times per second.

        Woken by the engine's subscription; the timeout doubles as a polling
        fallback, so an engine that never calls subscribers still streams.
        """
        loop = asyncio.get_running_loop()
        period = 1.0 / self._settings.frame_hz
        last_index: int | None = None
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._snapshot_ready.wait(), timeout=period)
            self._snapshot_ready.clear()
            if not self._clients:
                continue
            snapshot = self._engine.snapshot()
            if snapshot is None or snapshot.frame_index < 0 or snapshot.frame_index == last_index:
                continue  # nothing new (a -1 index is a blank snapshot of a failed engine)
            started = loop.time()
            last_index = snapshot.frame_index
            # Labels the engine attached to this very frame win over the fish-card cache.
            identities = {
                **self._identities,
                **wire.identities_from_labels(getattr(snapshot, "fish", None)),
            }
            try:
                message = await asyncio.to_thread(self._encode_frame, snapshot, identities)
            except Exception:
                self._failed("frame")
            else:
                self.frames_encoded += 1
                self._broadcast_frame(message)
                if any(
                    t.confirmed and t.fish_id is not None and t.fish_id not in self._identities
                    for t in snapshot.tracks
                ):
                    self._fish_wake.set()  # a fish without a card yet: refresh (rate-limited)
            await asyncio.sleep(max(0.0, started + period - loop.time()))

    def _encode_frame(
        self, snapshot: SnapshotLike, identities: dict[str, tuple[str, str | None]]
    ) -> str:
        """Worker thread: JPEG once, base64 once, JSON once - shared by every client."""
        s = self._settings
        jpeg = self._engine.jpeg(max_width=s.stream_width, quality=s.jpeg_quality)
        return wire.envelope("frame", wire.frame_message(snapshot, jpeg, identities))

    async def _fish_loop(self) -> None:
        """``fish`` cards every 1/fish_hz s, or sooner (at most 1 Hz) when woken."""
        period = 1.0 / self._settings.fish_hz
        min_gap = min(1.0, period)
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._fish_wake.wait(), timeout=period)
            self._fish_wake.clear()
            wait = self._last_fish_at + min_gap - self._clock()
            if wait > 0:
                await asyncio.sleep(wait)
            if not self._clients:
                continue
            self._last_fish_at = self._clock()
            try:
                cards = await asyncio.to_thread(self._engine.fish_cards)
                self._identities = wire.identities_from_cards(cards)
                fish = [wire.fish_card(card) for card in cards]
                self._broadcast(wire.envelope("fish", {"fish": fish}), cache_as="fish")
            except Exception:
                self._failed("fish")

    def _entropy_tick(self) -> None:
        snapshot = self._engine.snapshot()
        if snapshot is None or snapshot.t == self._last_entropy_t:
            return  # the engine has not moved: no duplicate history points
        self._last_entropy_t = snapshot.t
        data = wire.entropy_message(snapshot, self._settings.pool_capacity_bits)
        message = wire.envelope("entropy", data)
        now = self._clock()
        self._history.append(_HistoryEntry(now, data, message))
        self._prune_history(now)
        self._broadcast(message, cache_as="entropy")

    def _bits_tick(self) -> None:
        if not self._clients:
            return
        snapshot = self._engine.snapshot()
        bits = self._engine.rng.display_bits(self._settings.display_bits)
        t = snapshot.t if snapshot is not None else 0.0
        self._broadcast(wire.envelope("bits", wire.bits_message(t, bits)), cache_as="bits")

    def _health_tick(self) -> None:
        snapshot = self._engine.snapshot()
        if snapshot is not None:
            message = wire.envelope("health", wire.health_message(snapshot))
            self._broadcast(message, cache_as="health")

    def _led_tick(self) -> None:
        snapshot = self._engine.snapshot()
        if snapshot is not None:
            message = wire.envelope("led", wire.liveness_status(snapshot.liveness))
            self._broadcast(message, cache_as="led")

    # -- entropy history ---------------------------------------------------------

    def _prune_history(self, now: float) -> None:
        horizon = now - self._settings.history_max_s
        while self._history and self._history[0].recorded_at < horizon:
            self._history.popleft()

    def _recent(self, seconds: float) -> Iterator[_HistoryEntry]:
        now = self._clock()
        self._prune_history(now)
        horizon = now - seconds
        return (entry for entry in self._history if entry.recorded_at >= horizon)

    def history(self, seconds: float) -> list[dict[str, Any]]:
        """``entropy`` data of the last ``seconds`` (at most ``history_max_s``), oldest first."""
        return [entry.data for entry in self._recent(min(seconds, self._settings.history_max_s))]

    # -- connections ---------------------------------------------------------------

    def source_label(self) -> str:
        """Where the data comes from, for ``hello`` and ``/api/status``."""
        frame = self._engine.latest_frame()
        if frame is not None:
            return wire.describe_source(frame.source_id)
        # Before the first frame only the configured kind is known; the rest of
        # the spec may hold paths or stream URLs that are nobody's business.
        return wire.describe_source(self._engine.config.source.partition(":")[0])

    async def hello(self) -> str:
        """The ``hello`` message, waiting briefly for the engine's first frame."""
        if not self._first_snapshot.is_set():
            with anyio.move_on_after(self._settings.hello_wait_s):
                await self._first_snapshot.wait()
        snapshot = self._engine.snapshot()
        fps = self._settings.frame_hz
        if snapshot is not None and snapshot.fps > 0:
            fps = min(fps, round(snapshot.fps, 1))
        data = wire.hello_message(
            version=self._version,
            source=self.source_label(),
            width=snapshot.width if snapshot is not None else 0,
            height=snapshot.height if snapshot is not None else 0,
            fps=fps,
            beacon_public_key=self._beacon_public_key(),
        )
        return wire.envelope("hello", data)

    def welcome(self) -> list[str]:
        """What a new client gets right after ``hello``: entropy history, then latest state."""
        replay = self._settings.replay_entropy_s
        messages = [entry.text for entry in self._recent(replay)] if replay > 0 else []
        if not messages and "entropy" in self._last:
            messages.append(self._last["entropy"])
        messages.extend(self._last[kind] for kind in _STATE_TYPES if kind in self._last)
        return messages

    async def serve(self, websocket: WebSocket) -> None:
        """Run one ``/ws/live`` connection from handshake to disconnect."""
        if not origin_allowed(websocket.headers, self._settings.allowed_origins):
            await websocket.close(code=CLOSE_POLICY_VIOLATION)  # before accept: HTTP 403
            return
        await websocket.accept()
        if self._closing or len(self._clients) >= self._settings.max_clients:
            reason = "server shutting down" if self._closing else "too many clients"
            await websocket.close(code=CLOSE_TRY_AGAIN_LATER, reason=reason)
            return
        hello = await self.hello()
        channel = ClientChannel(max_queue=self._settings.client_queue)
        # Snapshot the welcome and register in one step - no await in between -
        # so every broadcast is either in the welcome or queued in the channel:
        # a beacon pulse committed during the handshake is never lost.
        welcome = self.welcome()
        self._clients.add(channel)
        self._fish_wake.set()  # a new viewer gets current fish cards (rate-limited)
        try:
            try:
                # Written directly while the channel only queues, so hello is
                # first; bounded, so a peer that never reads cannot pin us.
                with anyio.fail_after(self._settings.handshake_timeout_s):
                    await websocket.send_text(hello)
                    for message in welcome:
                        await websocket.send_text(message)
            except Exception:
                logger.debug("live client failed the handshake", exc_info=True)
                return
            await self._run_client(websocket, channel)
        finally:
            self._clients.discard(channel)

    async def _run_client(self, websocket: WebSocket, channel: ClientChannel) -> None:
        """Pump messages until the client leaves or the server closes it.

        Sender, receiver and close watcher run in one anyio task group, the
        structured-concurrency model Starlette itself runs on: whichever ends
        first cancels the others, and a cancellation from outside (server
        shutdown) tears all three down instead of orphaning them. A close
        request cancels the sender even while it is stuck writing to a client
        that stopped reading, so a dead peer cannot pin a handler.
        """
        client_left = False

        async def receive() -> None:
            nonlocal client_left
            await _until_disconnect(websocket)
            client_left = True

        async with anyio.create_task_group() as group:

            async def first_to_finish(job: Callable[[], Awaitable[None]]) -> None:
                try:
                    await job()
                except Exception:
                    logger.debug("live client ended", exc_info=True)
                finally:
                    group.cancel_scope.cancel()

            group.start_soon(first_to_finish, lambda: channel.pump(websocket))
            group.start_soon(first_to_finish, receive)
            group.start_soon(first_to_finish, channel.wait_closed)

        close = channel.close_request
        if close is not None and not client_left:
            with anyio.move_on_after(_CLOSE_TIMEOUT_S), contextlib.suppress(Exception):
                await websocket.close(code=close[0], reason=close[1])


async def _until_disconnect(websocket: WebSocket) -> None:
    """Read (and ignore) client messages until the client goes away.

    The protocol is server-to-client only; reading is still required to
    notice disconnects promptly. Message size is bounded by the server
    (uvicorn ``ws_max_size``).
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
