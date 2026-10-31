"""``puffer fish``: the leaderboard of the most chaotic swimmers.

Reads ``data_dir/puffer.db`` (written by the engine while it runs). The file
is checked for existence first because opening a missing SQLite database
would silently create an empty one.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table
from rich.text import Text

from puffer.cli import wiring
from puffer.cli.commands.common import JsonOption, emit_json
from puffer.cli.context import get_state
from puffer.cli.render import bar, fish_color

__all__ = ["DB_NAME", "fish"]

DB_NAME = "puffer.db"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def fish(
    ctx: typer.Context,
    days: Annotated[int, typer.Option("--days", min=1, help="Ranking window, UTC days.")] = 7,
    json_output: JsonOption = False,
) -> None:
    """Show the fish leaderboard: mean chaos score and credited bits per fish."""
    state = get_state(ctx)
    config = state.config()
    db_path = Path(config.data_dir) / DB_NAME
    rows: list[dict[str, Any]] = []
    if db_path.is_file():
        stats = wiring.open_fish_stats(db_path)
        try:
            rows = [dict(row) for row in stats.leaderboard(days)]
        finally:
            stats.close()
    if json_output:
        emit_json(rows)
        return
    if not rows:
        state.out.print(
            Text(
                f"No fish statistics in {db_path} yet. Run `puffer serve` (or keygen) "
                "and let the fish swim for a while.",
                style="dim",
            )
        )
        return
    table = Table(title=f"Fish leaderboard - last {days} day(s)", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("fish")
    table.add_column("chaos score")
    table.add_column("", justify="right")
    table.add_column("credited bits", justify="right")
    for row in rows:
        fish_id = str(row.get("fish_id", "?"))
        color = fish_color(fish_id, row.get("color"))
        name = Text("● ", style=color)
        name.append(str(row.get("name") or fish_id))
        chaos = _number(row.get("chaos_score"))
        bits = _number(row.get("bits"))
        table.add_row(
            str(row.get("rank", "")),
            name,
            bar(chaos, 20, color) if chaos is not None else Text("-", style="dim"),
            f"{chaos:.2f}" if chaos is not None else "-",
            f"{bits:,.1f}" if bits is not None else "-",
        )
    state.out.print(table)
