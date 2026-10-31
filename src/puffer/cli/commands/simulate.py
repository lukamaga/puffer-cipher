"""``puffer simulate``: render the aquarium simulator to a video file.

The demo video doubles as a liveness lesson: the simulated LED shows a fresh
one-time challenge from :class:`~puffer.liveness.ChallengeSchedule`, so
feeding the recording back into the engine with ``--liveness required``
fails verification - a replayed tank is recognised as stale, by design.

The video is written to a hidden temporary file next to the destination and
renamed into place only when complete, so an interrupted render never leaves
a truncated file that looks finished.
"""

from __future__ import annotations

import enum
import math
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from puffer.cli.commands.common import ForceOption, refuse_existing
from puffer.cli.context import get_state
from puffer.cli.errors import CliError, ExitCode
from puffer.cli.render import human_bytes

__all__ = ["CODECS", "LOSSLESS_CODECS", "SimMode", "simulate"]

CODECS = {
    ".mp4": "mp4v",
    ".m4v": "mp4v",
    ".mov": "mp4v",
    ".avi": "MJPG",
    ".mkv": "FFV1",
}
"""Default FourCC per container (OpenCV's FFmpeg backend)."""

LOSSLESS_CODECS = frozenset({"FFV1", "png "})


class SimMode(enum.StrEnum):
    DAY = "day"
    NIGHT = "night"
    CYCLE = "cycle"


def simulate(
    ctx: typer.Context,
    out: Annotated[
        Path, typer.Option("-o", "--out", dir_okay=False, help="Video file to write.")
    ] = Path("demo.mp4"),
    seconds: Annotated[
        float, typer.Option("--seconds", min=0.1, help="Length of the video, seconds.")
    ] = 20.0,
    fish: Annotated[int, typer.Option("--fish", min=0, max=32, help="Number of fish.")] = 6,
    mode: Annotated[SimMode, typer.Option("--mode", help="Lighting.")] = SimMode.DAY,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Simulator seed for a reproducible tank \\[default: random]."),
    ] = None,
    fps: Annotated[float, typer.Option("--fps", min=0.5, max=240.0)] = 30.0,
    width: Annotated[int, typer.Option("--width", min=160, max=4096)] = 640,
    height: Annotated[int, typer.Option("--height", min=120, max=4096)] = 480,
    codec: Annotated[
        str | None,
        typer.Option(
            "--codec",
            metavar="FOURCC",
            help="Video codec \\[default: by extension: .mp4 mp4v, .avi MJPG, .mkv FFV1].",
        ),
    ] = None,
    led: Annotated[
        bool, typer.Option("--led/--no-led", help="Show the liveness LED challenge.")
    ] = True,
    force: ForceOption = False,
) -> None:
    """Render the aquarium simulator (fish, plants, bubbles, LED) to a video file."""
    from puffer.liveness import ChallengeSchedule
    from puffer.sources.recorder import record
    from puffer.sources.synthetic import SyntheticAquarium, SyntheticConfig

    state = get_state(ctx)
    fourcc = codec if codec is not None else CODECS.get(out.suffix.lower())
    if fourcc is None:
        raise CliError(
            f"no default codec for {out.suffix or 'files without an extension'}",
            ExitCode.USAGE,
            hint=f"use one of {', '.join(CODECS)} or pass --codec FOURCC",
        )
    if len(fourcc) != 4:
        raise CliError(f"--codec must be exactly 4 characters, got {fourcc!r}", ExitCode.USAGE)
    refuse_existing(out, force=force)
    try:
        sim_config = SyntheticConfig(
            width=width,
            height=height,
            fps=fps,
            n_fish=fish,
            seed=seed,
            mode=mode.value,
        )
    except ValueError as exc:
        raise CliError(f"invalid simulator settings: {exc}", ExitCode.USAGE) from exc

    total = max(1, math.ceil(seconds * fps - 1e-9))
    partial = out.with_name(f".{out.stem}.partial{out.suffix}")
    schedule = ChallengeSchedule(symbol_s=0.2) if led else None
    progress = Progress(
        TextColumn("[bold]Rendering the aquarium"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("frames"),
        TimeRemainingColumn(),
        console=state.err,
        transient=True,
    )
    try:
        with SyntheticAquarium(sim_config, led_schedule=schedule) as sim, progress:
            task = progress.add_task("render", total=total)
            frames = record(
                sim,
                partial,
                seconds=seconds,
                fourcc=fourcc,
                on_frame=lambda _frame: progress.advance(task),
            )
            actual_seed = sim.seed
        if force:
            os.replace(partial, out)
        else:
            refuse_existing(out, force=False)  # created while we were rendering?
            os.replace(partial, out)
    finally:
        partial.unlink(missing_ok=True)

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("video", os.fspath(out))
    table.add_row("frames", f"{frames} ({frames / fps:.1f} s at {fps:g} fps)")
    table.add_row("size", f"{width}x{height}, {human_bytes(out.stat().st_size)}, codec {fourcc}")
    table.add_row("tank", f"{fish} fish, {mode.value}, seed {actual_seed}")
    state.out.print(table)
    if fourcc not in LOSSLESS_CODECS:
        state.out.print(
            Text(
                f"{fourcc} is lossy: fine for people, but it smears the pixel noise the sensor "
                "channel measures. Use .mkv (FFV1) for recordings meant for entropy analysis.",
                style="dim",
            )
        )
    if led:
        state.out.print(
            Text(
                "The LED shows a one-time liveness challenge: replaying this video with "
                "--liveness required fails verification - the anti-replay check at work.",
                style="dim",
            )
        )
