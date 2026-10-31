"""``puffer beacon verify`` and ``puffer beacon latest`` (DESIGN.md 3.4).

Verification is offline and needs nothing but the pulses and, ideally, the
beacon's public key: every pulse's Ed25519 signature, summary hash and
output are checked, and consecutive pulses must form an unbroken hash chain
whose precommitments were honoured (:func:`puffer.beacon.verify_chain`).

**Pin the key.** Without ``--public-key`` the key embedded in the first
pulse is trusted on first use. That proves only that the chain is internally
consistent - anyone can produce a valid-looking chain under their own key -
so the command says so, loudly, every time.

Beacon outputs are public randomness (lotteries, audits, games). They must
never be used as secret keys.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from puffer.beacon import (
    ChainReport,
    ErrorCode,
    Pulse,
    VerificationResult,
    verify_chain,
    verify_pulse,
)
from puffer.beacon.pulse import BeaconFormatError
from puffer.cli.commands.common import JsonOption, emit_json
from puffer.cli.context import CliState, get_state
from puffer.cli.errors import CliError, ExitCode
from puffer.cli.http import MAX_RESPONSE_BYTES, HttpStatusError, get_json, join_url, parse_json

__all__ = ["EXPLANATIONS", "beacon_app", "load_pulses", "parse_public_key"]

beacon_app = typer.Typer(
    help="Verify the public randomness beacon (Ed25519 signatures + BLAKE2b hash chain).",
    no_args_is_help=True,
)

EXPLANATIONS: Mapping[ErrorCode, str] = {
    ErrorCode.WRONG_PUBLIC_KEY: "signed by a different key than the trusted one",
    ErrorCode.BAD_SUMMARY_HASH: "the summary does not match its signed hash (edited summary)",
    ErrorCode.BAD_SIGNATURE: "the Ed25519 signature is invalid for the trusted key",
    ErrorCode.BAD_OUTPUT: "output is not BLAKE2b-512(message || signature)",
    ErrorCode.BAD_GENESIS: "pulse 0 must have an all-zero prev_output",
    ErrorCode.BAD_INDEX: "the index does not follow the previous pulse",
    ErrorCode.BAD_TIMESTAMP: "the timestamp is not after the previous pulse's",
    ErrorCode.BROKEN_LINK: "prev_output differs from the previous pulse's output",
    ErrorCode.BAD_PRECOMMITMENT: "local_random does not match the previous precommitment",
}
"""One-line meaning of every verification error code."""

_PUBLIC_KEY = re.compile(r"[0-9a-f]{64}")
_PAGE = 100
_SHOWN = 10
_DEFAULT_URL = "http://127.0.0.1:8000"

PublicKeyOption = Annotated[
    str | None,
    typer.Option(
        "--public-key",
        metavar="HEX",
        help="The beacon's published Ed25519 key (64 hex, GET /api/beacon/public-key). "
        "Without it the chain is only checked for internal consistency.",
    ),
]


def parse_public_key(text: str) -> bytes:
    """A 32-byte Ed25519 public key from 64 hex characters (case-insensitive)."""
    value = text.strip().lower()
    if _PUBLIC_KEY.fullmatch(value) is None:
        raise CliError(
            "--public-key must be 64 hex characters (a 32-byte Ed25519 key)", ExitCode.USAGE
        )
    return bytes.fromhex(value)


def _pulses_from_json(data: object, origin: str) -> list[Pulse]:
    """A list of pulses (``/api/beacon/pulses``) or a single pulse (``/latest``)."""
    items = data if isinstance(data, list) else [data]
    pulses: list[Pulse] = []
    for position, item in enumerate(items):
        try:
            pulses.append(Pulse.from_json(item))
        except BeaconFormatError as exc:
            raise CliError(
                f"{origin}: pulse #{position} is malformed: {exc}", ExitCode.FAILURE
            ) from exc
    return pulses


def _one_pulse(data: object, origin: str) -> Pulse:
    if not isinstance(data, dict):
        raise CliError(f"{origin}: expected a single pulse object", ExitCode.FAILURE)
    return _pulses_from_json(data, origin)[0]


def _is_url(target: str) -> bool:
    return target.lower().startswith(("http://", "https://"))


def _fetch_chain(base: str, max_pulses: int) -> tuple[list[Pulse], bool]:
    """Page through ``/api/beacon/pulses`` from genesis; ``(pulses, truncated)``."""
    pulses: list[Pulse] = []
    start = 0
    while len(pulses) < max_pulses:
        limit = min(_PAGE, max_pulses - len(pulses))
        url = join_url(base, "/api/beacon/pulses", start=start, limit=limit)
        batch = _pulses_from_json(get_json(url), url)
        if not batch:
            return pulses, False
        following = batch[-1].index + 1
        if following <= start:  # a server that ignores ?start= would loop forever
            raise CliError(f"{url}: the server did not advance past index {start}")
        pulses.extend(batch)
        start = following
    return pulses, True


def load_pulses(target: str, *, max_pulses: int) -> tuple[list[Pulse], bool]:
    """Pulses from a JSON file, a pulse URL, or a server base URL (whole chain).

    Returns ``(pulses, truncated)``; ``truncated`` means ``max_pulses`` stopped
    the download before the end of the chain.
    """
    if _is_url(target):
        path = target.split("://", 1)[1].partition("/")[2].split("?", 1)[0]
        if path.strip("/") == "":
            return _fetch_chain(target, max_pulses)
        return _pulses_from_json(get_json(target), target), False
    file = Path(target)
    with file.open("rb") as handle:
        body = handle.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise CliError(f"{target}: larger than {MAX_RESPONSE_BYTES >> 20} MiB", ExitCode.IO)
    return _pulses_from_json(parse_json(body, target), target), False


def _summary(pulse: Pulse) -> dict[str, Any]:
    """The pulse summary as a dict, or ``{}`` (it is untrusted display data)."""
    try:
        value = json.loads(pulse.summary)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _int_field(summary: Mapping[str, Any], name: str) -> str:
    value = summary.get(name)
    return f"{value:,}" if type(value) is int else "-"


def _time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")


def _short(data: bytes) -> str:
    return data.hex()[:16] + "..."


def _top_fish(summary: Mapping[str, Any], limit: int = 3) -> Text:
    text = Text()
    fish = summary.get("top_fish")
    if not isinstance(fish, list):
        return text
    for item in fish[:limit]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            text.append(", " if text.plain else "")
            text.append(item["name"])
            if type(item.get("bits")) is int:
                text.append(f" {item['bits']}", style="dim")
    return text


def _pulse_table(pulses: list[Pulse], title: str) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("index", justify="right", no_wrap=True)
    table.add_column("time (UTC)", no_wrap=True)
    table.add_column("output", no_wrap=True)
    table.add_column("fish", justify="right", no_wrap=True)
    table.add_column("sensor", justify="right", no_wrap=True)
    table.add_column("liveness", no_wrap=True)
    table.add_column("top fish (bits)")
    for pulse in pulses:
        summary = _summary(pulse)
        liveness = summary.get("liveness")
        table.add_row(
            str(pulse.index),
            _time(pulse.timestamp_ms),
            _short(pulse.output),
            _int_field(summary, "fish_bits"),
            _int_field(summary, "sensor_bits"),
            Text(liveness if isinstance(liveness, str) else "-"),
            _top_fish(summary),
        )
    return table


def _warn_unpinned(state: CliState, key: bytes) -> None:
    state.err.print(
        Panel(
            Text(
                "No --public-key pinned: the chain was checked against the key embedded in its "
                f"first pulse ({key.hex()[:16]}...). That proves internal consistency only - "
                "anyone can build a valid-looking chain under their own key. Pin the beacon's "
                "published key (GET /api/beacon/public-key) with --public-key HEX."
            ),
            title=Text("warning: trust on first use", style="bold yellow"),
            border_style="yellow",
            expand=False,
        )
    )


@beacon_app.command("verify")
def verify(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(
            metavar="FILE_OR_URL",
            help="JSON file of pulses (/api/beacon/pulses format), a pulse URL, or a server "
            "URL such as http://host:8000 (downloads and checks the whole chain).",
        ),
    ],
    public_key: PublicKeyOption = None,
    max_pulses: Annotated[
        int, typer.Option("--max-pulses", min=1, help="Download at most this many pulses.")
    ] = 10_000,
    json_output: JsonOption = False,
) -> None:
    """Verify a chain of beacon pulses offline (exit 0 verified, 1 not verified)."""
    state = get_state(ctx)
    pinned = parse_public_key(public_key) if public_key is not None else None
    if not _is_url(target) and not os.path.exists(target):
        raise CliError(f"no such file: {target}", ExitCode.IO)
    pulses, truncated = load_pulses(target, max_pulses=max_pulses)
    if not pulses:
        raise CliError(f"{target} contains no pulses: nothing to verify", ExitCode.FAILURE)
    trusted = pinned if pinned is not None else pulses[0].public_key
    report: ChainReport = verify_chain(pulses, trusted)

    if pinned is None:
        _warn_unpinned(state, trusted)
    if json_output:
        emit_json(
            {
                "ok": report.ok,
                "checked": report.checked,
                "first_index": pulses[0].index,
                "last_index": pulses[-1].index,
                "public_key": trusted.hex(),
                "pinned": pinned is not None,
                "truncated": truncated,
                "errors": [
                    {"position": e.position, "index": e.index, "code": str(e.code)}
                    for e in report.errors
                ],
            }
        )
    elif report.ok:
        span = f"index {pulses[0].index}-{pulses[-1].index}"
        when = f"{_time(pulses[0].timestamp_ms)} - {_time(pulses[-1].timestamp_ms)} UTC"
        state.out.print(
            Text.assemble(
                ("✓ verified ", "bold green"),
                (f"{report.checked} pulse(s), {span}, {when}", ""),
                (f"\n  key {trusted.hex()} ({'pinned' if pinned else 'embedded'})", "dim"),
            )
        )
        shown = pulses[-_SHOWN:]
        title = f"latest {len(shown)} of {len(pulses)} pulses" if len(pulses) > _SHOWN else ""
        state.out.print(_pulse_table(shown, title))
    else:
        state.out.print(
            Text.assemble(
                ("✗ NOT verified: ", "bold red"),
                (f"{len(report.errors)} failed check(s) in {report.checked} pulse(s)", ""),
            )
        )
        table = Table()
        table.add_column("position", justify="right")
        table.add_column("index", justify="right")
        table.add_column("check", style="red")
        table.add_column("meaning")
        for error in report.errors:
            table.add_row(
                str(error.position),
                str(error.index),
                str(error.code),
                EXPLANATIONS.get(error.code, ""),
            )
        state.out.print(table)
    if truncated:
        state.err.print(
            Text(f"note: only the first {len(pulses)} pulses were checked (--max-pulses)")
        )
    if not report.ok:
        raise typer.Exit(ExitCode.FAILURE)


def _default_url(state: CliState) -> str:
    try:
        config = state.config()
    except CliError:
        return _DEFAULT_URL
    return f"http://{config.host}:{config.port}"


@beacon_app.command("latest")
def latest(
    ctx: typer.Context,
    url: Annotated[
        str | None,
        typer.Option("--url", help="Server base URL \\[default: from the configuration]."),
    ] = None,
    public_key: PublicKeyOption = None,
    json_output: JsonOption = False,
) -> None:
    """Fetch the newest pulse, verify it (and its link to the previous one) and show it."""
    state = get_state(ctx)
    pinned = parse_public_key(public_key) if public_key is not None else None
    base = url if url is not None else _default_url(state)
    latest_url = join_url(base, "/api/beacon/latest")
    try:
        data = get_json(latest_url)
    except HttpStatusError as exc:
        if exc.status == 404:
            raise CliError(
                f"the beacon at {base} has not published a pulse yet", ExitCode.FAILURE
            ) from exc
        raise
    pulse = _one_pulse(data, latest_url)
    prev: Pulse | None = None
    prev_note = "genesis pulse" if pulse.index == 0 else ""
    if pulse.index > 0:
        prev_url = join_url(base, f"/api/beacon/pulse/{pulse.index - 1}")
        try:
            prev = _one_pulse(get_json(prev_url), prev_url)
        except CliError as exc:
            prev_note = f"previous pulse unavailable, link not checked ({exc})"
    result: VerificationResult = verify_pulse(pulse, public_key=pinned, prev=prev)
    trusted = pinned if pinned is not None else (prev or pulse).public_key

    if pinned is None:
        _warn_unpinned(state, trusted)
    if json_output:
        emit_json(
            {
                "ok": result.ok,
                "errors": [str(code) for code in result.errors],
                "pinned": pinned is not None,
                "link_checked": prev is not None,
                "pulse": pulse.to_json(),
            }
        )
    else:
        state.out.print(_latest_panel(pulse, result, prev, prev_note))
    if not result.ok:
        raise typer.Exit(ExitCode.FAILURE)


def _latest_panel(
    pulse: Pulse, result: VerificationResult, prev: Pulse | None, prev_note: str
) -> Panel:
    summary = _summary(pulse)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("time", f"{_time(pulse.timestamp_ms)} UTC (period {pulse.period_ms / 1000:g} s)")
    output = pulse.output.hex()
    grid.add_row("output", f"{output[:64]}\n{output[64:]}")
    grid.add_row(
        "entropy",
        f"fish {_int_field(summary, 'fish_bits')} · sensor {_int_field(summary, 'sensor_bits')}"
        f" · system {_int_field(summary, 'system_bits')} bits",
    )
    liveness = summary.get("liveness")
    grid.add_row(
        "liveness",
        f"{liveness if isinstance(liveness, str) else '-'} "
        f"({_int_field(summary, 'liveness_confidence_bits')} confidence bits)",
    )
    grid.add_row("top fish", _top_fish(summary))
    if result.ok:
        checks = "signature, summary hash and output"
        if prev is not None:
            checks += f"; linked to pulse {prev.index} (prev_output, precommitment)"
        verdict = Text.assemble(("✓ ", "bold green"), (checks, ""))
    else:
        verdict = Text("✗ " + ", ".join(str(code) for code in result.errors), style="bold red")
    if prev_note:
        verdict.append(f"\n{prev_note}", style="dim")
    grid.add_row("verified", verdict)
    return Panel(
        grid,
        title=Text(f"Beacon pulse {pulse.index}", style="bold"),
        subtitle=Text("public randomness - never use it as a secret key", style="dim"),
        border_style="green" if result.ok else "red",
        expand=False,
    )
