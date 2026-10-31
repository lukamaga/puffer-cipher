"""Rich renderables: the live collection panel, entropy receipts, warnings.

Everything here is a pure function from data to a Rich renderable, so the
look of the CLI is unit-testable (render to a recording console, assert on
the text) and independent of where the data comes from.

Two rules keep the output trustworthy:

* **No markup injection.** Fish names come from configuration and re-ID, and
  pulse summaries from remote servers. They are always placed in
  :class:`rich.text.Text` objects (never interpolated into markup strings),
  and colours from outside are used only if they are strict ``#rrggbb``.
* **Honest labels.** Credited bits are presented as what they are - a
  conservative estimate of input min-entropy - and output from the
  simulator always says that it is simulated (DESIGN.md rule 4).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from puffer.cli.protocols import SnapshotLike
from puffer.types import (
    CAMERA_CHANNELS,
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    CHANNEL_SYSTEM,
    HealthState,
    LivenessState,
    LivenessStatus,
)

if TYPE_CHECKING:
    from puffer.entropy.receipt import EntropyReceipt

__all__ = [
    "LIVENESS_TARGET_BITS",
    "SIMULATED_NOTE",
    "CollectionView",
    "FishLabel",
    "attribution",
    "bar",
    "collection_panel",
    "fish_color",
    "health_text",
    "human_bytes",
    "liveness_text",
    "shortfall_panel",
]

SIMULATED_NOTE = (
    "simulated aquarium - fish entropy is simulated; security comes from mixing with os.urandom"
)

LIVENESS_TARGET_BITS = 24.0
"""Confidence the LED verifier requires before VERIFIED (DESIGN.md 5); display only."""

_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
_PALETTE = (
    "#ff7014",
    "#1965e6",
    "#c4b260",
    "#ffd60a",
    "#c81437",
    "#d6d6cd",
    "#00cdff",
    "#a0afa5",
)
"""Fallback fish colours (the simulator's default cast) for fish without a card colour."""

UNIDENTIFIED_PREFIX = "track-"
"""Owner id of fish samples from tracks without a re-ID identity (DESIGN.md 4.1)."""

_EIGHTHS = " ▏▎▍▌▋▊▉"
_DOT = "●"

_HEALTH_STYLE = {
    HealthState.OK: "green",
    HealthState.STARTUP: "cyan",
    HealthState.DEGRADED: "yellow",
    HealthState.FAILED: "bold red",
}
_LIVENESS_STYLE = {
    LivenessState.VERIFIED: "bold green",
    LivenessState.CALIBRATING: "yellow",
    LivenessState.FAILED: "bold red",
    LivenessState.DISABLED: "dim",
}


@dataclass(frozen=True, slots=True)
class FishLabel:
    """How one fish is shown: display name and a safe ``#rrggbb`` colour."""

    name: str
    color: str


def fish_color(fish_id: str, color: object = None) -> str:
    """``color`` if it is a strict ``#rrggbb`` string, else a stable palette colour.

    The fallback is derived from the fish id, so the same fish keeps its
    colour across runs even when the engine supplies none.
    """
    if isinstance(color, str) and _HEX_COLOR.fullmatch(color):
        return color.lower()
    index = hashlib.blake2b(fish_id.encode("utf-8"), digest_size=1).digest()[0]
    return _PALETTE[index % len(_PALETTE)]


def fish_labels(cards: Iterable[Mapping[str, object]]) -> dict[str, FishLabel]:
    """``{fish_id: FishLabel}`` from DESIGN.md 7.1 ``FishCard`` dicts (malformed cards skipped)."""
    labels: dict[str, FishLabel] = {}
    for card in cards:
        fish_id = card.get("fish_id")
        if not isinstance(fish_id, str) or not fish_id:
            continue
        name = card.get("name")
        labels[fish_id] = FishLabel(
            name=name if isinstance(name, str) and name else fish_id,
            color=fish_color(fish_id, card.get("color")),
        )
    return labels


def bar(fraction: float, width: int, style: str) -> Text:
    """A smooth horizontal bar (eighth-block resolution) over a dim track."""
    f = fraction if math.isfinite(fraction) else 0.0
    f = min(1.0, max(0.0, f))
    cells = f * width
    full = int(cells)
    partial = int((cells - full) * 8)
    filled = "█" * full + (_EIGHTHS[partial] if partial and full < width else "")
    text = Text(filled, style=style)
    text.append("·" * (width - len(filled)), style="bright_black")
    return text


def human_bytes(n: int) -> str:
    """``1536`` -> ``"1.5 KiB"`` (binary units, matching ``--bytes 4K``)."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{n} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


def _bits(value: float) -> str:
    return f"{value:,.1f}"


def health_text(state: HealthState | None) -> Text:
    """A health state in its colour (``-`` when unknown)."""
    if state is None:
        return Text("-", style="dim")
    return Text(state.value, style=_HEALTH_STYLE.get(state, ""))


def liveness_text(status: LivenessStatus | None, *, detail: bool = True) -> Text:
    """``● verified  31.5 / 24 confidence bits · match 97% · lag 50 ms``."""
    if status is None:
        return Text("waiting for the first frame", style="dim")
    style = _LIVENESS_STYLE.get(status.state, "")
    text = Text(f"{_DOT} {status.state.value}", style=style)
    if detail and status.state is not LivenessState.DISABLED:
        text.append(
            f"  {status.confidence_bits:.1f} / {LIVENESS_TARGET_BITS:g} confidence bits",
            style="default",
        )
        if status.match_ratio > 0:
            text.append(f" · match {status.match_ratio:.0%}", style="dim")
        if status.state is LivenessState.VERIFIED:
            text.append(f" · lag {status.lag_s * 1000:.0f} ms", style="dim")
    elif status.state is LivenessState.DISABLED:
        text.append("  (no LED challenge for this source)", style="dim")
    return text


# --------------------------------------------------------------------------
# Live collection panel
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CollectionView:
    """Everything the live panel shows at one instant."""

    source: str
    simulated: bool
    target_bits: float
    fish_target_bits: float
    timeout_s: float
    elapsed_s: float
    frames: int
    snapshot: SnapshotLike | None
    fish: Mapping[str, FishLabel]
    title: str = "Collecting fish entropy"


_BAR = 24
_TABLE_BAR = 16


def _credit(snapshot: SnapshotLike | None, channel: str) -> float:
    if snapshot is None:
        return 0.0
    value = snapshot.pool.get(channel, 0.0)
    return value if math.isfinite(value) and value > 0 else 0.0


def _fish_in_view(view: CollectionView) -> Text:
    if view.snapshot is None:
        return Text("-", style="dim")
    seen: list[str] = []
    for track in view.snapshot.tracks:
        if not track.confirmed:
            continue
        key = track.fish_id if track.fish_id is not None else f"track-{track.track_id}"
        if key not in seen:
            seen.append(key)
    if not seen:
        return Text("no fish tracked yet", style="dim")
    text = Text()
    shown = 6
    for fish_id in seen[:shown]:
        label = view.fish.get(fish_id, FishLabel(fish_id, fish_color(fish_id)))
        text.append(f"{_DOT} ", style=label.color)
        text.append(label.name)
        text.append("  ")
    if len(seen) > shown:
        text.append(f"+{len(seen) - shown} more ", style="dim")
    text.append(f"({len(seen)} in view)", style="dim")
    text.overflow = "fold"
    return text


def collection_panel(view: CollectionView) -> Panel:
    """The live panel shown while keygen/encrypt/random wait for camera entropy."""
    snap = view.snapshot
    sensor = _credit(snap, CHANNEL_SENSOR)
    fishbits = _credit(snap, CHANNEL_FISH)
    total = sensor + fishbits
    target = view.target_bits
    met = total >= target
    scale = target if target > 0 else max(total, 1.0)

    # Labels never shrink; the other columns may wrap on narrow terminals
    # (bars have a fixed width, so only the explanatory text ever wraps).
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True, min_width=12)
    grid.add_column(min_width=_BAR)
    grid.add_column()

    frames = Text(f"{view.frames:,} processed")
    if snap is not None:
        frames.append(f" · #{snap.frame_index} · {snap.fps:.1f} fps · {snap.mode.value}", "dim")
    grid.add_row("frames", frames, "")

    remaining = view.timeout_s - view.elapsed_s
    grid.add_row(
        "time",
        bar(view.elapsed_s / view.timeout_s if view.timeout_s > 0 else 1.0, _BAR, "blue"),
        Text(f"{view.elapsed_s:.1f} s of {view.timeout_s:g} s ({max(0.0, remaining):.0f} s left)"),
    )
    policy = getattr(snap, "liveness_policy", None)
    grid.add_row(
        "liveness",
        liveness_text(snap.liveness if snap else None),
        Text(f"policy {policy}", style="dim") if isinstance(policy, str) else "",
    )

    def channel_row(label: str, channel: str, bits: float, goal: float, style: str) -> None:
        state = None
        if snap is not None and channel in snap.channels:
            state = snap.channels[channel].health
        value = Text(f"{_bits(bits)} / {goal:g} bits  " if goal > 0 else f"{_bits(bits)} bits  ")
        value.append_text(health_text(state))
        grid.add_row(label, bar(bits / (goal if goal > 0 else scale), _BAR, style), value)

    fish_goal = view.fish_target_bits
    channel_row("sensor", CHANNEL_SENSOR, sensor, 0.0, "magenta")
    channel_row("fish", CHANNEL_FISH, fishbits, fish_goal, "cyan")
    met = met and fishbits >= fish_goal
    status = Text(f"{_bits(total)} / {target:g} bits")
    if met:
        status.append("  ✓ targets reached", style="bold green")
    grid.add_row(
        "camera", bar(total / scale, _BAR, "green" if total >= target else "white"), status
    )
    grid.add_row("fish in view", _fish_in_view(view), "")

    subtitle = Text(view.source, style="dim")
    if view.simulated:
        subtitle.append("  (simulated aquarium)", style="yellow")
    return Panel(
        grid,
        title=Text(view.title, style="bold"),
        subtitle=subtitle,
        border_style="green" if met else "cyan",
        expand=False,
    )


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------


def _share(value: float | None) -> Text:
    return Text("-", style="dim") if value is None else Text(f"{value:.1%}")


def attribution(
    receipt: EntropyReceipt,
    *,
    heading: str,
    fish: Mapping[str, FishLabel],
    simulated: bool,
) -> RenderableType:
    """Which fish, the sensor and the OS contributed how much to one output.

    Fish shares are of the fish channel; channel shares are of the camera
    total (DESIGN.md 4.5). ``os.urandom`` is listed with its fixed 256 bits:
    it keys every output and is the floor of its strength.
    """
    table = Table(title=Text(heading, style="bold"), title_justify="left", expand=False)
    table.add_column("contributor", overflow="fold")
    table.add_column("bits", justify="right", no_wrap=True)
    table.add_column("", no_wrap=True)
    table.add_column("share", justify="right", no_wrap=True)

    def scope(text: str) -> None:
        table.add_row(Text(text, style="dim italic"), "", "", "")

    scope("fish - share of the fish channel")
    if receipt.fish:
        for credit in receipt.fish:
            label = fish.get(credit.fish_id)
            color = label.color if label is not None else fish_color(credit.fish_id)
            name = Text(f"{_DOT} ", style=color)
            if credit.fish_id.startswith(UNIDENTIFIED_PREFIX) and credit.name == credit.fish_id:
                # FishChannel names tracks re-ID has not recognised (yet) "track-<id>".
                name.append("unidentified", style="italic")
                name.append(f" {credit.fish_id}", style="dim")
            else:
                name.append(credit.name)
                if credit.name != credit.fish_id:
                    name.append(f" {credit.fish_id}", style="dim")
            table.add_row(
                name, _bits(credit.bits), bar(credit.share, _TABLE_BAR, color), _share(credit.share)
            )
    else:
        table.add_row(Text("no fish credited", style="dim"), "", "", "")
    table.add_section()

    scope("channels - share of camera credit")
    channels = receipt.channels
    for channel, title, style in (
        (CHANNEL_FISH, "fish", "cyan"),
        (CHANNEL_SENSOR, "camera sensor", "magenta"),
    ):
        line = channels.get(channel)
        bits = line.credited_bits if line is not None else 0.0
        share = line.share if line is not None else None
        table.add_row(Text(title), _bits(bits), bar(share or 0.0, _TABLE_BAR, style), _share(share))
    table.add_section()
    system = channels.get(CHANNEL_SYSTEM)
    table.add_row(
        Text("os.urandom", style="bold"),
        _bits(system.credited_bits if system else 256.0),
        Text("always mixed in", style="dim"),
        _share(None),
    )

    facts = Text()
    facts.append("liveness ")
    verdict = LivenessStatus(
        receipt.liveness_state, confidence_bits=receipt.liveness_confidence_bits
    )
    facts.append_text(liveness_text(verdict, detail=False))
    if receipt.liveness_state is not LivenessState.DISABLED:
        facts.append(f" ({receipt.liveness_confidence_bits:.1f} confidence bits)")
    facts.append(f" · light {receipt.mode}")
    if receipt.frames is not None:
        facts.append(f" · frames {receipt.frames[0]}-{receipt.frames[1]}")
    health = [
        (channel, line.health)
        for channel, line in channels.items()
        if channel in CAMERA_CHANNELS and line.health is not None
    ]
    if health:
        facts.append(" · health ")
        for i, (channel, state) in enumerate(health):
            facts.append(", " if i else "")
            facts.append(f"{channel} ")
            facts.append_text(health_text(state))

    parts: list[RenderableType] = [table, facts]
    parts.append(
        Text(
            "Credited bits are a conservative estimate of the min-entropy absorbed into the "
            "pool. The output itself is derived from a 256-bit seed keyed with fresh "
            "os.urandom, so it is never weaker than the operating system's RNG.",
            style="dim",
        )
    )
    if simulated:
        parts.append(Text(f"Note: {SIMULATED_NOTE}.", style="yellow"))
    return Group(*parts)


def shortfall_panel(
    *,
    camera_bits: float,
    target_bits: float,
    fish_bits: float,
    fish_target_bits: float,
    elapsed_s: float,
    reasons: Sequence[str],
    what: str = "key",
    recorded_in: str = "the key file's receipt",
) -> Panel:
    """The loud warning shown when a camera entropy target was not reached."""
    body = Text()
    summary = f"After {elapsed_s:.1f} s: camera {_bits(camera_bits)} of {target_bits:g} bits"
    if fish_target_bits > 0:
        summary += f", fish {_bits(fish_bits)} of {fish_target_bits:g} bits"
    body.append(summary + ".\n", style="bold")
    for reason in reasons:
        body.append(f"  • {reason}\n")
    body.append(
        f"\nThe {what} is still exactly as strong as os.urandom - every output is keyed with "
        "256 fresh bits from the OS - but the fish did not contribute the entropy you asked "
        f"for. This is recorded in {recorded_in} (collection.target_met = false). "
        "Use --strict to refuse instead, or give the tank more --timeout."
    )
    return Panel(
        body,
        title=Text("WARNING: camera entropy target not met", style="bold red"),
        border_style="red",
        expand=False,
    )
