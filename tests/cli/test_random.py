"""``puffer random``: formats, sizes, JSON, streaming and broken pipes."""

from __future__ import annotations

import base64
import io
import json

import pytest
import typer

from puffer.cli.commands import rand
from puffer.types import LivenessState

from .conftest import Invoke, Tank


class _ClosingPipe(io.BytesIO):
    """A stdout whose reader goes away after ``limit`` bytes (like ``head -c``)."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def write(self, data: object) -> int:
        if self.tell() >= self.limit:
            raise BrokenPipeError(32, "Broken pipe")
        return super().write(data)  # type: ignore[arg-type]


def test_default_is_32_bytes_of_hex(cli: Invoke, tank: Tank) -> None:
    result = cli("random")

    assert result.exit_code == 0, result.output
    text = result.stdout.strip()
    assert len(text) == 64
    bytes.fromhex(text)
    assert tank.engine.rng.calls == [("generate", 32, "random")]
    assert tank.engine.index == -1  # --min-bits defaults to 0: no waiting
    assert tank.engine.closed


def test_base64_and_sizes(cli: Invoke, tank: Tank) -> None:
    result = cli("random", "-n", "4K", "--format", "base64")

    assert result.exit_code == 0, result.output
    assert len(base64.b64decode(result.stdout.strip(), validate=True)) == 4096


def test_raw_writes_exact_bytes(cli: Invoke, tank: Tank) -> None:
    result = cli("random", "-n", "100", "-f", "raw")

    assert result.exit_code == 0, result.output
    assert len(result.stdout_bytes) == 100


def test_outputs_differ(cli: Invoke, tank: Tank) -> None:
    first, second = cli("random").stdout, cli("random").stdout
    assert first != second


def test_json_carries_the_receipt(cli: Invoke, tank: Tank) -> None:
    result = cli("random", "-n", "16", "--json", "--min-bits", "256")

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["bytes"] == 16
    assert document["format"] == "hex"
    assert len(bytes.fromhex(document["data"])) == 16
    receipt = document["receipt"]
    assert receipt["purpose"] == "random"
    assert receipt["channels"]["sensor"]["credited_bits"] >= 256
    assert receipt["collection"]["target_met"] is True


def test_shortfall_is_a_warning_not_an_error(cli: Invoke, tank: Tank) -> None:
    tank.options.update(liveness=LivenessState.FAILED, frame_delay_s=0.001)

    result = cli("random", "--json", "--min-bits", "64", "--timeout", "0.2")

    assert result.exit_code == 0, result.output
    assert "WARNING: camera entropy target not met" in result.stderr
    assert "the --json receipt" in result.stderr
    assert json.loads(result.stdout)["receipt"]["collection"]["target_met"] is False


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--stream", "-f", "hex"), "--stream always writes raw bytes"),
        (("--stream", "--json"), "--stream cannot be combined with --json"),
        (("--json", "-f", "raw"), "--json needs --format hex or base64"),
        (("-n", "65M"), "at most 64 MiB per request"),
        (("-n", "lots"), "is not a size"),
        (("-n", "0"), "at least 1 byte"),
    ],
)
def test_usage_errors(cli: Invoke, tank: Tank, args: tuple[str, ...], message: str) -> None:
    result = cli("random", *args)

    assert result.exit_code == 2
    assert message in result.output
    assert tank.engines == []


def test_raw_bytes_never_go_to_a_terminal(
    cli: Invoke, tank: Tank, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rand, "_stdout_is_terminal", lambda: True)

    result = cli("random", "--stream")

    assert result.exit_code == 2
    assert "refusing to write raw bytes to a terminal" in result.stderr
    assert "dieharder" in result.stderr


def test_stream_with_limit_writes_exactly_that_many_bytes(cli: Invoke, tank: Tank) -> None:
    result = cli("random", "--stream", "-n", "2500K")

    assert result.exit_code == 0, result.output
    assert len(result.stdout_bytes) == 2500 * 1024
    engine = tank.engine
    assert (engine.started, engine.stopped) == (1, 1)  # frames keep flowing while streaming
    sizes = [n for kind, n, purpose in engine.rng.calls if purpose == "stream"]
    assert sizes == [1 << 20, 1 << 20, 2500 * 1024 - 2 * (1 << 20)]


def test_reader_closing_the_pipe_ends_the_stream_quietly(
    cli: Invoke, tank: Tank, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _ClosingPipe(limit=3 << 20)
    monkeypatch.setattr(rand, "_binary_stdout", lambda: pipe)

    result = cli("random", "--stream")  # endless: only the reader can stop it

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert pipe.tell() == 3 << 20
    assert tank.engine.stopped == 1
    assert tank.engine.closed


def test_reader_closing_the_pipe_on_plain_output(
    cli: Invoke, tank: Tank, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rand, "_binary_stdout", lambda: _ClosingPipe(limit=0))

    result = cli("random", "-n", "1M", "-f", "raw")  # e.g. puffer random ... | head -c 0

    assert result.exit_code == 0, result.output
    assert result.stderr == ""


def test_broken_pipe_guard_silences_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    silenced: list[bool] = []
    monkeypatch.setattr(rand, "_silence_stdout", lambda: silenced.append(True))

    with rand._until_reader_closes():
        raise BrokenPipeError(32, "Broken pipe")
    with rand._until_reader_closes():
        pass

    assert silenced == [True]
    with pytest.raises(OSError, match="disk"), rand._until_reader_closes():
        raise OSError("disk full")  # only a closed pipe is a normal end


def test_write_stream_chunks() -> None:
    class Rng:
        def __init__(self) -> None:
            self.sizes: list[int] = []

        def generate(self, n: int, purpose: str = "bytes") -> tuple[bytes, None]:
            self.sizes.append(n)
            return bytes(n), None

    rng, out = Rng(), io.BytesIO()
    assert rand.write_stream(rng, out, limit=10, chunk=4) == 10  # type: ignore[arg-type]
    assert rng.sizes == [4, 4, 2]
    assert out.getvalue() == bytes(10)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("32", 32),
        ("4K", 4096),
        ("4k", 4096),
        ("4KiB", 4096),
        ("4KB", 4096),
        ("10M", 10 << 20),
        ("1G", 1 << 30),
        (" 7 ", 7),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert rand.parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "-1", "1.5M", "4X", "0", "M"])
def test_parse_size_rejects(text: str) -> None:
    with pytest.raises(typer.BadParameter):
        rand.parse_size(text)
