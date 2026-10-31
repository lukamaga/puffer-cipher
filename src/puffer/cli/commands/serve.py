"""``puffer serve`` and ``puffer analyze``: thin front ends to the API server and the report.

Both delegate to modules that are imported lazily (:mod:`puffer.cli.wiring`):
the server pulls in FastAPI/uvicorn and the report matplotlib, neither of
which the rest of the CLI should pay for or depend on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from puffer.cli import wiring
from puffer.cli.commands.common import SourceOption
from puffer.cli.context import get_state

__all__ = ["DEFAULT_REPORT_SOURCES", "analyze", "serve"]

DEFAULT_REPORT_SOURCES = ("synthetic:seed=1,mode=day", "synthetic:seed=1,mode=night")


def serve(
    ctx: typer.Context,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            metavar="PATH",
            exists=True,
            dir_okay=False,
            help="TOML configuration (overrides the global --config).",
        ),
    ] = None,
    source: SourceOption = None,
    host: Annotated[str | None, typer.Option("--host", help="Interface to bind.")] = None,
    port: Annotated[int | None, typer.Option("--port", min=1, max=65535, help="TCP port.")] = None,
) -> None:
    """Run the engine, the beacon, the REST/WebSocket API and the dashboard."""
    state = get_state(ctx)
    config = state.config(config_path=config_path, source=source, host=host, port=port)
    state.err.print(
        Text.assemble(
            ("puffer serve ", "bold"),
            (f"http://{config.host}:{config.port}", "bold cyan"),
            (f"  source {config.source} · liveness {config.liveness}", "dim"),
        )
    )
    wiring.serve(config)


def analyze(
    ctx: typer.Context,
    sources: Annotated[
        list[str] | None,
        typer.Option(
            "--source",
            metavar="SPEC",
            help="Source to analyse; repeat for several \\[default: the global --source, else "
            "the simulator by day and by night].",
        ),
    ] = None,
    frames: Annotated[int, typer.Option("--frames", min=1, help="Frames per source.")] = 3000,
    out: Annotated[
        Path, typer.Option("--out", file_okay=False, help="Directory for RESULTS.md.")
    ] = Path("docs/results"),
    media: Annotated[
        Path, typer.Option("--media", file_okay=False, help="Directory for the plots.")
    ] = Path("docs/media"),
    nist_sequences: Annotated[
        int, typer.Option("--nist-sequences", min=1, help="1e6-bit SP 800-22 sequences per stream.")
    ] = 10,
    dieharder: Annotated[
        bool, typer.Option("--dieharder", help="Also run dieharder on PufferRNG output.")
    ] = False,
    seed: Annotated[int | None, typer.Option("--seed", help="Seed for reproducible runs.")] = None,
) -> None:
    """Build the statistics report (SP 800-90B, SP 800-22, ent, dieharder)."""
    state = get_state(ctx)
    chosen = list(sources) if sources else []
    if not chosen:
        chosen = [state.options.source] if state.options.source else list(DEFAULT_REPORT_SOURCES)

    def progress(message: str) -> None:
        state.err.print(Text(f"· {message}", style="dim"))

    path = wiring.run_report(
        sources=chosen,
        frames=frames,
        out_dir=out,
        media_dir=media,
        nist_sequences=nist_sequences,
        dieharder=dieharder,
        seed=seed,
        progress=progress,
    )
    state.out.print(Text.assemble(("Report written to ", ""), (str(path), "bold")))
