"""Collecting camera entropy before drawing a secret, and accounting for it.

``puffer keygen`` promises "a key the fish helped make", so before drawing it
waits until the camera channels (sensor noise + fish motion) have earned
``--min-bits`` of credit in the pool. Three outcomes are possible and each is
handled explicitly:

1. **Target reached** - draw the key; its receipt says which fish helped.
2. **Timeout or the source ended** (liveness never verified, the tank is
   dark, a video file ran out) - the key is *still* drawn, because PufferRNG
   keys every output with 256 fresh bits from ``os.urandom`` and is
   therefore never weaker than the OS. The shortfall is shown loudly and
   written into the receipt, so nobody later believes the fish contributed
   what they did not. ``--strict`` turns this into an error instead.
3. **Interrupted** - nothing is drawn or written.

Two targets must both be met: ``--min-bits`` of camera credit (sensor +
fish) and ``--min-fish-bits`` from the fish channel alone. The second exists
because the sensor channel saturates within a frame or two, while the fish
earn credit only after their start-up health tests (about a thousand
observations): without it, "a key the fish helped make" would usually be a
key the fish never touched.

Frames are processed by :meth:`EngineLike.collect` - the engine's own
synchronous loop, which retries a live camera that missed a frame and skips
isolated failing frames - asked for one frame at a time, so both targets are
re-checked after every frame (no overshoot) and the panel can redraw. The
engine signals the end of a finite source by returning ``False`` without
having read a frame before its time budget is over.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from puffer.cli.errors import CliError, ExitCode
from puffer.cli.protocols import EngineLike, RngLike, SnapshotLike
from puffer.cli.render import (
    LIVENESS_TARGET_BITS,
    CollectionView,
    FishLabel,
    collection_panel,
    fish_labels,
    shortfall_panel,
)
from puffer.types import (
    CAMERA_CHANNELS,
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    HealthState,
    LivenessState,
)

if TYPE_CHECKING:
    from puffer.entropy.receipt import EntropyReceipt

__all__ = [
    "KEY_BYTES",
    "CollectionResult",
    "FishKey",
    "SourceInfo",
    "StopReason",
    "camera_credit",
    "collect",
    "collect_with_panel",
    "collection_record",
    "describe_source",
    "explain_shortfall",
    "fish_credit",
    "make_fish_key",
    "public_random",
    "shortfall",
    "warm_up",
]

KEY_BYTES = 32
StopReason = Literal["target", "timeout", "source_ended"]
Clock = Callable[[], float]
Progress = Callable[[SnapshotLike | None, int, float], None]

_SLICE_S = 0.25  # longest single engine.collect() call (bounds a stalled camera's wait)
_REFRESH_S = 0.1  # live panel redraw interval (10 Hz is smooth and cheap)
_CARDS_REFRESH_S = 1.0  # fish cards (names, colours) change slowly
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")
_KNOWN_KINDS = ("synthetic", "video", "camera", "picamera")


@dataclass(frozen=True, slots=True)
class CollectionResult:
    """How a collection ended; credits are the pool's undrawn credit at the end."""

    target_bits: float
    camera_bits: float
    frames: int
    elapsed_s: float
    stop: StopReason
    fish_target_bits: float = 0.0
    fish_bits: float = 0.0
    snapshot: SnapshotLike | None = field(default=None, compare=False, repr=False)

    @property
    def met(self) -> bool:
        return self.camera_bits >= self.target_bits and self.fish_bits >= self.fish_target_bits


def _pool_bits(snapshot: SnapshotLike | None, channel: str) -> float:
    if snapshot is None:
        return 0.0
    value = snapshot.pool.get(channel, 0.0)
    return value if math.isfinite(value) and value > 0 else 0.0


def camera_credit(snapshot: SnapshotLike | None) -> float:
    """Undrawn credit of the camera channels (sensor + fish) in a snapshot."""
    return sum(_pool_bits(snapshot, name) for name in CAMERA_CHANNELS)


def fish_credit(snapshot: SnapshotLike | None) -> float:
    """Undrawn credit of the fish channel in a snapshot."""
    return _pool_bits(snapshot, CHANNEL_FISH)


def _index(snapshot: SnapshotLike | None) -> int | None:
    return snapshot.frame_index if snapshot is not None else None


def collect(
    engine: EngineLike,
    target_bits: float,
    *,
    timeout_s: float,
    reachable_bits: float,
    fish_bits: float = 0.0,
    on_progress: Progress | None = None,
    clock: Clock = time.monotonic,
    slice_s: float = _SLICE_S,
) -> CollectionResult:
    """Let the engine process frames until both credit targets are met.

    ``target_bits`` is camera credit (sensor + fish), ``fish_bits`` fish
    credit alone; ``reachable_bits`` is the most the pool can ever hold
    (``2 * pool_capacity_bits``). Stops at ``timeout_s`` (wall time) or when
    a finite source ends. Credit already in the pool counts, so zero targets
    are met without reading a frame. ``on_progress(snapshot, frames,
    elapsed)`` runs after every frame.
    """
    start = clock()
    deadline = start + timeout_s
    snapshot = engine.snapshot()
    first = _index(snapshot)
    frames = 0
    stop: StopReason
    while True:
        if camera_credit(snapshot) >= target_bits and fish_credit(snapshot) >= fish_bits:
            stop = "target"
            break
        now = clock()
        budget = min(slice_s, deadline - now)
        if budget <= 0:
            stop = "timeout"
            break
        # Once the camera total is met only the fish target is outstanding;
        # aim at "full" so the engine still steps (it returns True at once for
        # a goal that is already met). One frame per call: no overshoot.
        goal = target_bits if camera_credit(snapshot) < target_bits else reachable_bits
        before = _index(snapshot)
        reached = engine.collect(goal, timeout_s=budget, max_frames=1)
        spent = clock() - now
        snapshot = engine.snapshot()
        latest = _index(snapshot)
        if latest is not None:
            frames = latest - (first if first is not None else -1)
        if on_progress is not None:
            on_progress(snapshot, frames, clock() - start)
        if not reached and spent < budget and latest == before:
            stop = "source_ended"  # collect() gave up early without reading a frame
            break
    return CollectionResult(
        target_bits=target_bits,
        camera_bits=camera_credit(snapshot),
        frames=frames,
        elapsed_s=clock() - start,
        stop=stop,
        fish_target_bits=fish_bits,
        fish_bits=fish_credit(snapshot),
        snapshot=snapshot,
    )


def warm_up(engine: EngineLike, *, frames: int, timeout_s: float, reachable_bits: float) -> None:
    """Feed up to ``frames`` frames (or ``timeout_s``) so the pool holds fresh tank state.

    Used before drawing *public* values (an Argon2 salt): they consume no
    credit, but mixing the current pool snapshot is the point of asking the
    fish at all. Implemented as a collection towards a full pool that stops
    after ``frames`` frames - the engine's own robust stepping loop.
    """
    engine.collect(reachable_bits, timeout_s=timeout_s, max_frames=frames)


def explain_shortfall(result: CollectionResult) -> list[str]:
    """Human-readable reasons why ``result`` did not reach its target."""
    reasons: list[str] = []
    if result.stop == "source_ended":
        reasons.append(f"the frame source ended after {result.frames} frame(s)")
    elif result.stop == "timeout":
        reasons.append(f"timed out after {result.elapsed_s:.1f} s ({result.frames} frames)")
    snap = result.snapshot
    if snap is None:
        reasons.append("no frame was processed")
        return reasons
    live = snap.liveness
    if live.state in (LivenessState.CALIBRATING, LivenessState.FAILED):
        reasons.append(
            f"liveness never verified (state {live.state.value}, {live.confidence_bits:.1f} of "
            f"{LIVENESS_TARGET_BITS:g} confidence bits): the LED challenge was not seen in the "
            "video - a replayed or frozen feed, a hidden LED, or simply not enough time"
        )
    for name in CAMERA_CHANNELS:
        report = snap.health.get(name)
        stats = snap.channels.get(name)
        state = report.state if report is not None else stats.health if stats else None
        if state in (HealthState.FAILED, HealthState.STARTUP, HealthState.DEGRADED):
            why = ", ".join(report.reasons) if report is not None and report.reasons else ""
            reasons.append(f"{name} channel {state.value}" + (f": {why}" if why else ""))
    if result.camera_bits >= result.target_bits and result.fish_bits < result.fish_target_bits:
        reasons.append(
            f"the fish earned only {result.fish_bits:.1f} of {result.fish_target_bits:g} bits "
            "(their health tests need about a thousand observations before credit starts)"
        )
    elif len(reasons) == 1 and result.stop == "timeout":
        reasons.append(
            f"credit accrued too slowly (sensor {_pool_bits(snap, CHANNEL_SENSOR):.1f}, "
            f"fish {_pool_bits(snap, CHANNEL_FISH):.1f} bits)"
        )
    return reasons


def shortfall(
    result: CollectionResult,
    reasons: list[str] | tuple[str, ...],
    *,
    what: str = "key",
    recorded_in: str = "the key file's receipt",
) -> Panel:
    """The loud "target not met" warning for ``result``."""
    return shortfall_panel(
        camera_bits=result.camera_bits,
        target_bits=result.target_bits,
        fish_bits=result.fish_bits,
        fish_target_bits=result.fish_target_bits,
        elapsed_s=result.elapsed_s,
        reasons=reasons,
        what=what,
        recorded_in=recorded_in,
    )


# --------------------------------------------------------------------------
# Source description (for honesty labels and receipts)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Which data produced an output (DESIGN.md rule 4)."""

    spec: str
    kind: str

    @property
    def simulated(self) -> bool:
        return self.kind == "synthetic"


def redact_spec(spec: str) -> str:
    """Hide credentials embedded in a stream URL (``rtsp://user:pass@host``)."""
    return _URL_CREDENTIALS.sub(r"\g<scheme>***@", spec)


def describe_source(spec: str, engine: EngineLike | None = None) -> SourceInfo:
    """Classify the source, preferring what the frames themselves report.

    ``Frame.source_id`` (``"synthetic"``, ``"video:tank.mp4"``, ``"camera:0"``)
    is authoritative; the configured spec is the fallback before any frame.
    """
    kind = ""
    frame = engine.latest_frame() if engine is not None else None
    if frame is not None:
        kind = frame.source_id.partition(":")[0].strip().lower()
    if kind not in _KNOWN_KINDS:
        prefix = spec.strip().partition(":")[0].strip().lower()
        if prefix in _KNOWN_KINDS:
            kind = prefix
        elif spec.strip().isdigit():
            kind = "camera"
        else:
            kind = kind or "unknown"
    return SourceInfo(spec=redact_spec(spec), kind=kind)


def collection_record(
    result: CollectionResult, source: SourceInfo, reasons: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """The ``collection`` block added to receipts: what was asked for and what happened."""
    record: dict[str, Any] = {
        "source": source.spec,
        "source_kind": source.kind,
        "simulated": source.simulated,
        "target_bits": result.target_bits,
        "camera_bits": round(result.camera_bits, 3),
        "fish_target_bits": result.fish_target_bits,
        "fish_bits": round(result.fish_bits, 3),
        "target_met": result.met,
        "stopped": result.stop,
        "frames": result.frames,
        "elapsed_s": round(result.elapsed_s, 3),
    }
    if not result.met:
        record["warning"] = (
            "camera entropy target not met: this output's strength rests on os.urandom alone"
        )
        record["reasons"] = list(reasons)
    return record


# --------------------------------------------------------------------------
# Live panel glue
# --------------------------------------------------------------------------


class _FishNames:
    """Fish names and colours from ``engine.fish_cards()``, refreshed at most once a second."""

    def __init__(self, engine: EngineLike, clock: Clock) -> None:
        self._engine = engine
        self._clock = clock
        self._at = -math.inf
        self.labels: dict[str, FishLabel] = {}

    def refresh(self, *, force: bool = False) -> dict[str, FishLabel]:
        now = self._clock()
        if force or now - self._at >= _CARDS_REFRESH_S:
            self.labels = fish_labels(self._engine.fish_cards())
            self._at = now
        return self.labels


def collect_with_panel(
    engine: EngineLike,
    *,
    target_bits: float,
    timeout_s: float,
    reachable_bits: float,
    console: Console,
    source: SourceInfo,
    fish_bits: float = 0.0,
    transient: bool = False,
    title: str = "Collecting fish entropy",
    clock: Clock = time.monotonic,
) -> CollectionResult:
    """:func:`collect` with a live Rich panel on ``console`` (stderr).

    On a terminal the panel animates at ~10 Hz; elsewhere Rich prints only
    the final state (or nothing when ``transient``), which keeps logs short.
    """
    names = _FishNames(engine, clock)

    def view(snapshot: SnapshotLike | None, frames: int, elapsed: float) -> CollectionView:
        return CollectionView(
            source=source.spec,
            simulated=source.simulated,
            target_bits=target_bits,
            fish_target_bits=fish_bits,
            timeout_s=timeout_s,
            elapsed_s=elapsed,
            frames=frames,
            snapshot=snapshot,
            fish=names.labels,
            title=title,
        )

    last = -math.inf
    with Live(
        collection_panel(view(engine.snapshot(), 0, 0.0)),
        console=console,
        auto_refresh=False,
        transient=transient,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as live:

        def on_progress(snapshot: SnapshotLike | None, frames: int, elapsed: float) -> None:
            nonlocal last
            now = clock()
            if now - last >= _REFRESH_S:
                last = now
                names.refresh()
                live.update(collection_panel(view(snapshot, frames, elapsed)), refresh=True)

        result = collect(
            engine,
            target_bits,
            timeout_s=timeout_s,
            reachable_bits=reachable_bits,
            fish_bits=fish_bits,
            on_progress=on_progress,
            clock=clock,
        )
        names.refresh(force=True)
        live.update(
            collection_panel(view(result.snapshot, result.frames, result.elapsed_s)),
            refresh=True,
        )
    if not transient and not console.is_terminal:
        console.line()  # Live ends its last frame with a newline only on terminals
    return result


# --------------------------------------------------------------------------
# Fish keys
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FishKey:
    """A freshly drawn key with everything needed to explain where it came from."""

    key: bytes = field(repr=False)
    receipt: EntropyReceipt
    collection: CollectionResult
    source: SourceInfo
    fish: Mapping[str, FishLabel]
    reasons: tuple[str, ...] = ()

    def receipt_document(self) -> dict[str, Any]:
        """``EntropyReceipt.to_dict()`` plus the ``collection`` block (never secret)."""
        document = self.receipt.to_dict()
        document["collection"] = collection_record(self.collection, self.source, self.reasons)
        return document


def make_fish_key(
    engine: EngineLike,
    *,
    target_bits: float,
    fish_bits: float,
    timeout_s: float,
    reachable_bits: float,
    strict: bool,
    console: Console,
    source_spec: str,
) -> FishKey:
    """Collect camera entropy (with the live panel), then draw a 32-byte key.

    Raises:
        CliError: ``strict`` and a target was not met (exit code
            :attr:`ExitCode.ENTROPY`); no key is drawn in that case.
    """
    result = collect_with_panel(
        engine,
        target_bits=target_bits,
        fish_bits=fish_bits,
        timeout_s=timeout_s,
        reachable_bits=reachable_bits,
        console=console,
        source=describe_source(source_spec, engine),
    )
    source = describe_source(source_spec, engine)  # the frames now say what the source really is
    reasons: tuple[str, ...] = ()
    if not result.met:
        reasons = tuple(explain_shortfall(result))
        if strict:
            raise CliError(
                f"camera entropy target not met in {result.elapsed_s:.1f} s "
                f"(camera {result.camera_bits:.1f} of {target_bits:g} bits, fish "
                f"{result.fish_bits:.1f} of {fish_bits:g}); no key was written",
                ExitCode.ENTROPY,
                details="\n".join(f"- {reason}" for reason in reasons),
                hint="without --strict puffer still writes a key exactly as strong as "
                "os.urandom (and says so in its receipt); or allow a longer --timeout",
            )
        console.print(shortfall(result, reasons))
    key, receipt = engine.rng.generate(KEY_BYTES, "key")
    return FishKey(
        key=key,
        receipt=receipt,
        collection=result,
        source=source,
        fish=fish_labels(engine.fish_cards()),
        reasons=reasons,
    )


def public_random(rng: RngLike, purpose: Literal["salt", "nonce"]) -> Callable[[int], bytes]:
    """A ``random_bytes(n)`` hook for *public* values (salts, nonces).

    Prefers ``rng.salt`` / ``rng.nonce``, which are OS-keyed but consume no
    camera credit (``generate(..., consume=False)``), so a salt never steals
    the fish entropy a key generated right after it should get. Older RNGs
    without those helpers fall back to a regular, credit-consuming draw with
    the same purpose string - equally strong, just more wasteful.
    """
    helper = getattr(rng, purpose, None)
    if callable(helper):
        return cast(Callable[[int], bytes], helper)

    def fallback(n: int) -> bytes:
        return rng.generate(n, purpose)[0]

    return fallback
