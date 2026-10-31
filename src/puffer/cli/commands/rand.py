"""``puffer random``: fish-seeded random bytes, or an endless stream for dieharder.

``puffer random --stream | dieharder -g 200 -a`` feeds dieharder's raw-stdin
generator (``-g 200``) with PufferRNG output. The stream is written in 1 MiB
chunks; every chunk is a separate ``PufferRNG.generate`` call, i.e. a fresh
pool draw keyed with fresh ``os.urandom``, while the engine keeps absorbing
frames on its background thread.

dieharder (like ``head -c``) simply closes the pipe when it has read enough.
That is the normal end of a stream, not an error, so ``BrokenPipeError`` is
handled quietly with exit status 0 - following the recipe in the Python
documentation (``signal`` module notes on SIGPIPE): stdout is pointed at
``/dev/null`` so the interpreter's final flush cannot print a traceback.
"""

from __future__ import annotations

import base64
import contextlib
import enum
import os
import re
import sys
from collections.abc import Iterator
from typing import Annotated, BinaryIO

import typer

from puffer.cli.collect import (
    collect_with_panel,
    collection_record,
    describe_source,
    explain_shortfall,
    shortfall,
)
from puffer.cli.commands.common import (
    JsonOption,
    MinBitsOption,
    SourceOption,
    TimeoutOption,
    check_reachable,
    emit_json,
    open_engine,
)
from puffer.cli.context import get_state
from puffer.cli.errors import CliError, ExitCode
from puffer.cli.protocols import RngLike

__all__ = ["CHUNK_BYTES", "MAX_BYTES", "OutputFormat", "parse_size", "random", "write_stream"]

CHUNK_BYTES = 1 << 20
"""Stream chunk: one ``generate`` call (and one pool draw) per MiB."""

MAX_BYTES = 64 << 20
"""Largest non-stream request (held in memory); use --stream for more."""

DEFAULT_COUNT = 32
DEFAULT_MIN_BITS = 0.0
"""``random`` answers at once by default; ``--min-bits`` waits for camera credit first."""
DEFAULT_TIMEOUT_S = 30.0

_SIZE = re.compile(r"(?P<n>\d+)\s*(?P<unit>[kmgt]?)(?:i?b)?", re.IGNORECASE)
_UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


class OutputFormat(enum.StrEnum):
    HEX = "hex"
    BASE64 = "base64"
    RAW = "raw"


def parse_size(text: str) -> int:
    """``"32"``, ``"4K"``, ``"10MiB"``, ``"1g"`` -> bytes (binary units: K = 1024).

    Raises:
        typer.BadParameter: not a positive size.
    """
    match = _SIZE.fullmatch(text.strip())
    if match is None:
        raise typer.BadParameter(f"{text!r} is not a size (examples: 32, 4K, 10M, 1G)")
    value = int(match["n"]) * _UNITS[match["unit"].lower()]
    if value < 1:
        raise typer.BadParameter("the size must be at least 1 byte")
    return value


def write_stream(
    rng: RngLike, out: BinaryIO, *, limit: int | None, chunk: int = CHUNK_BYTES
) -> int:
    """Write ``limit`` bytes (``None``: forever) of PufferRNG output to ``out``.

    Returns the number of bytes written. ``BrokenPipeError`` propagates; the
    command treats it as the reader having finished.
    """
    written = 0
    while limit is None or written < limit:
        n = chunk if limit is None else min(chunk, limit - written)
        data, _ = rng.generate(n, "stream")
        out.write(data)
        written += n
    out.flush()
    return written


def _stdout_is_terminal() -> bool:
    """Whether stdout is a terminal (a seam for tests)."""
    return sys.stdout.isatty()


def _binary_stdout() -> BinaryIO:
    """The binary layer of stdout (a seam for tests)."""
    sys.stdout.flush()
    return sys.stdout.buffer


def _silence_stdout() -> None:
    """After EPIPE, point fd 1 at /dev/null so the final flush at exit cannot fail."""
    with contextlib.suppress(OSError, ValueError):  # no real fd (e.g. under CliRunner)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        finally:
            os.close(devnull)


@contextlib.contextmanager
def _until_reader_closes() -> Iterator[None]:
    """Treat a closed pipe (``| head -c 64``, dieharder finishing) as a normal end."""
    try:
        yield
    except BrokenPipeError:
        _silence_stdout()


def _encode(data: bytes, fmt: OutputFormat) -> str:
    if fmt is OutputFormat.HEX:
        return data.hex()
    return base64.b64encode(data).decode("ascii")


def random(
    ctx: typer.Context,
    count: Annotated[
        str | None,
        typer.Option(
            "-n",
            "--bytes",
            metavar="SIZE",
            help="Number of bytes: 32, 4K, 10M, 1G (binary units). With --stream: total to "
            "write \\[default: 32; endless with --stream].",
        ),
    ] = None,
    fmt: Annotated[
        OutputFormat | None,
        typer.Option("-f", "--format", help="Output encoding \\[default: hex; raw with --stream]."),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Write raw bytes endlessly in 1 MiB chunks, e.g. | dieharder -g 200 -a.",
        ),
    ] = False,
    json_output: JsonOption = False,
    source: SourceOption = None,
    min_bits: MinBitsOption = DEFAULT_MIN_BITS,
    timeout: TimeoutOption = DEFAULT_TIMEOUT_S,
) -> None:
    """Print random bytes from PufferRNG (fish + sensor entropy, keyed with os.urandom).

    By default answers at once with whatever the pool holds. --min-bits N
    first waits for N bits of camera entropy; if the tank cannot provide them
    in time the output is still exactly as strong as os.urandom, and a
    warning says so.
    """
    state = get_state(ctx)
    limit = parse_size(count) if count is not None else None
    if stream:
        if fmt not in (None, OutputFormat.RAW):
            raise CliError("--stream always writes raw bytes", ExitCode.USAGE)
        if json_output:
            raise CliError("--stream cannot be combined with --json", ExitCode.USAGE)
        fmt = OutputFormat.RAW
    else:
        fmt = fmt if fmt is not None else OutputFormat.HEX
        limit = limit if limit is not None else DEFAULT_COUNT
        if limit > MAX_BYTES:
            raise CliError(
                f"at most {MAX_BYTES >> 20} MiB per request", ExitCode.USAGE, hint="use --stream"
            )
        if json_output and fmt is OutputFormat.RAW:
            raise CliError("--json needs --format hex or base64", ExitCode.USAGE)
    if fmt is OutputFormat.RAW and not json_output and _stdout_is_terminal():
        raise CliError(
            "refusing to write raw bytes to a terminal",
            ExitCode.USAGE,
            hint="redirect or pipe the output (e.g. | dieharder -g 200 -a), or use --format hex",
        )

    config = state.config(source=source)
    reachable = check_reachable(min_bits, config)
    with open_engine(state, config) as engine:
        result = collect_with_panel(
            engine,
            target_bits=min_bits,
            timeout_s=timeout,
            reachable_bits=reachable,
            console=state.err,
            source=describe_source(config.source, engine),
            transient=True,
            title="Warming up the fish",
        )
        info = describe_source(config.source, engine)
        reasons = [] if result.met else explain_shortfall(result)
        if not result.met:
            state.err.print(
                shortfall(result, reasons, what="output", recorded_in="the --json receipt")
            )

        if stream:
            engine.start()  # keep absorbing frames while the main thread writes
            try:
                with _until_reader_closes():
                    write_stream(engine.rng, _binary_stdout(), limit=limit)
            finally:
                engine.stop()
            return

        assert limit is not None
        data, receipt = engine.rng.generate(limit, "random")

    with _until_reader_closes():
        if fmt is OutputFormat.RAW:
            out = _binary_stdout()
            out.write(data)
            out.flush()
        elif json_output:
            document = receipt.to_dict()
            document["collection"] = collection_record(result, info, reasons)
            emit_json(
                {
                    "bytes": limit,
                    "format": fmt.value,
                    "data": _encode(data, fmt),
                    "receipt": document,
                }
            )
        else:
            sys.stdout.write(_encode(data, fmt) + "\n")
            sys.stdout.flush()
