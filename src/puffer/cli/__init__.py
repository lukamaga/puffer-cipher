"""The ``puffer`` command line (DESIGN.md 9.3), built with Typer and Rich.

``puffer keygen`` makes a key the fish helped make, ``encrypt``/``decrypt``
handle ``.puf`` files, ``random`` prints (or streams) PufferRNG output,
``simulate`` renders the aquarium, ``serve`` runs the dashboard, ``beacon``
verifies the public beacon, ``analyze`` builds the statistics report,
``fish`` shows the leaderboard and ``info`` the installation.

Layout: :mod:`.app` (Typer app, error handling), :mod:`.commands` (one
module per command family), :mod:`.collect` (waiting for camera entropy),
:mod:`.render` (Rich output), :mod:`.wiring` (lazy, hookable access to the
engine and friends), :mod:`.protocols` (what the CLI needs from them),
:mod:`.errors` (exit codes).
"""

from __future__ import annotations

from collections.abc import Sequence

from puffer.cli.app import app

__all__ = ["app", "main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point (``puffer = puffer.cli:main``)."""
    app(args=list(argv) if argv is not None else None, prog_name="puffer")
