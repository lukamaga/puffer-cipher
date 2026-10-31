"""Unit tests of the CLI building blocks: errors, wiring, collection, rendering."""

from __future__ import annotations

import errno
import io
import logging
import math
import sys
from pathlib import Path

import pytest
from rich.console import Console

from puffer.cli import collect, render, wiring
from puffer.cli.errors import CliError, ExitCode, translate
from puffer.config import PufferConfig
from puffer.crypto import AuthenticationError, FormatError, TruncatedError, WrongKeyError
from puffer.entropy import EntropyPool, PufferRNG
from puffer.hardware import LedHardwareError
from puffer.sources.base import SourceError
from puffer.sources.spec import SourceSpecError
from puffer.types import LivenessState, LivenessStatus

from .fakes import FakeConfig, FakeEngine, RecordingRng

# -------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("exc", "code", "text"),
    [
        (FileExistsError(errno.EEXIST, "exists", "a.key"), ExitCode.EXISTS, "a.key"),
        (WrongKeyError("file is encrypted with key x"), ExitCode.WRONG_KEY, "wrong key"),
        (AuthenticationError("chunk 0"), ExitCode.AUTHENTICATION, "authentication failed"),
        (TruncatedError("chunk 3"), ExitCode.TRUNCATED, "truncated"),
        (FormatError("bad magic"), ExitCode.FORMAT, "not a valid puffer-cipher file"),
        (SourceSpecError("unknown kind"), ExitCode.USAGE, "unknown kind"),
        (SourceError("no camera"), ExitCode.IO, "cannot open the frame source"),
        (FileNotFoundError(errno.ENOENT, "missing", "x"), ExitCode.IO, "no such file: x"),
        (PermissionError(errno.EACCES, "denied", "y"), ExitCode.IO, "permission denied"),
        (OSError(errno.EIO, "boom"), ExitCode.IO, "I/O error"),
    ],
)
def test_translate(exc: BaseException, code: ExitCode, text: str) -> None:
    error = translate(exc)
    assert error is not None
    assert error.code == code
    assert text in error.message


def test_translate_leaves_bugs_alone() -> None:
    assert translate(ZeroDivisionError()) is None
    error = CliError("x", ExitCode.ENTROPY)
    assert translate(error) is error
    assert str(error) == "x"
    assert error.args == ("x",)


def test_spec_errors_carry_the_grammar() -> None:
    error = translate(SourceSpecError("bad"))
    assert error is not None
    assert error.details is not None
    assert "synthetic" in error.details


# -------------------------------------------------------------------- wiring


def test_unavailable_messages() -> None:
    own = wiring._unavailable(
        ModuleNotFoundError("x", name="puffer.analysis.report"), "puffer.analysis.report", "it"
    )
    assert own.code == ExitCode.UNAVAILABLE
    assert "is missing" in own.message
    dep = wiring._unavailable(ModuleNotFoundError("x", name="ultralytics"), "puffer.runtime", "it")
    assert "'ultralytics'" in dep.message
    assert dep.hint == "pip install 'puffer-cipher[yolo]'"
    other = wiring._unavailable(ImportError("cannot import name X"), "puffer.runtime", "it")
    assert "could not be loaded" in other.message


def test_apply_overrides_validates_like_the_config_file() -> None:
    config = PufferConfig()
    assert wiring.apply_overrides(config, source=None) is config
    updated = wiring.apply_overrides(config, port=9001, source="synthetic:seed=2")
    assert (updated.port, updated.source) == (9001, "synthetic:seed=2")
    with pytest.raises(CliError) as info:
        wiring.apply_overrides(config, port=70000)
    assert info.value.code == ExitCode.USAGE


def test_password_variable_never_reaches_the_config_parser(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(wiring, "config_loader", None)
    monkeypatch.setenv("PUFFER_PASSWORD", "hunter2")
    monkeypatch.setenv("PUFFER_PORT", "8123")

    with caplog.at_level(logging.WARNING):
        config = wiring.load_config(None)

    assert config.port == 8123
    assert "PUFFER_PASSWORD" not in caplog.text


def test_real_config_errors_are_usage_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(wiring, "config_loader", None)
    with pytest.raises(CliError) as info:
        wiring.load_config(tmp_path / "missing.toml")
    assert info.value.code == ExitCode.USAGE
    assert "not found" in info.value.message


def test_build_engine_maps_led_and_type_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import puffer.runtime

    def no_led(config: PufferConfig) -> None:
        raise LedHardwareError("gpiozero is not installed")

    monkeypatch.setattr(wiring, "engine_factory", None)
    monkeypatch.setattr(puffer.runtime, "build_engine", no_led)
    with pytest.raises(CliError) as info:
        wiring.build_engine(PufferConfig())
    assert info.value.code == ExitCode.UNAVAILABLE
    assert "LED" in info.value.message
    with pytest.raises(TypeError, match="expected PufferConfig"):
        wiring.build_engine(FakeConfig())


# -------------------------------------------------------------------- collection


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.001
        return self.now


def _collect(engine: FakeEngine, target: float, fish: float = 0.0, timeout: float = 5.0):
    return collect.collect(
        engine, target, timeout_s=timeout, reachable_bits=8192.0, fish_bits=fish, clock=_Clock()
    )


def test_collect_meets_targets_frame_by_frame() -> None:
    engine = FakeEngine(FakeConfig(), sensor_bits=100.0, fish_bits=10.0)
    result = _collect(engine, 250.0)
    assert (result.stop, result.frames, result.met) == ("target", 3, True)
    assert result.camera_bits == pytest.approx(330.0)
    assert all(call[2] == 1 for call in engine.collect_calls)  # one frame per call


def test_collect_keeps_going_for_the_fish() -> None:
    engine = FakeEngine(FakeConfig(), sensor_bits=4096.0, fish_bits=10.0)
    result = _collect(engine, 256.0, fish=35.0)
    assert (result.stop, result.frames) == ("target", 4)
    assert result.fish_bits == pytest.approx(40.0)  # no overshoot past the frame that met it


def test_collect_detects_the_end_of_a_finite_source() -> None:
    engine = FakeEngine(FakeConfig(), sensor_bits=1.0, frames=5)
    result = _collect(engine, 256.0)
    assert (result.stop, result.frames, result.met) == ("source_ended", 5, False)
    assert "the frame source ended after 5 frame(s)" in collect.explain_shortfall(result)


def test_collect_times_out_and_explains_why() -> None:
    engine = FakeEngine(FakeConfig(), liveness=LivenessState.CALIBRATING, frame_delay_s=0.001)
    result = collect.collect(engine, 64.0, timeout_s=0.05, reachable_bits=8192.0)
    assert result.stop == "timeout"
    assert not result.met
    assert result.frames > 0
    reasons = collect.explain_shortfall(result)
    assert any("liveness never verified (state calibrating" in r for r in reasons)
    assert any("sensor channel failed: liveness failed" in r for r in reasons)


def test_collect_zero_target_needs_no_frame() -> None:
    engine = FakeEngine(FakeConfig())
    result = _collect(engine, 0.0)
    assert (result.stop, result.frames, engine.collect_calls) == ("target", 0, [])


def test_shortfall_names_the_fish() -> None:
    result = collect.CollectionResult(256.0, 4100.0, 10, 1.0, "timeout", 64.0, 4.0, None)
    reasons = collect.explain_shortfall(result)
    assert reasons[-1] == "no frame was processed"
    assert not result.met


def test_describe_source_and_redaction() -> None:
    assert collect.describe_source("0").kind == "camera"
    assert collect.describe_source("video:a.mp4").kind == "video"
    assert collect.describe_source("tank.mp4").kind == "unknown"
    info = collect.describe_source("camera:rtsp://user:pw@cam/stream")
    assert info.spec == "camera:rtsp://***@cam/stream"
    assert not info.simulated
    engine = FakeEngine(FakeConfig(), source_id="video:x.mkv")
    engine.collect(1.0, timeout_s=1.0)
    assert collect.describe_source("synthetic", engine).kind == "video"  # the frames win


def test_public_random_prefers_salt() -> None:
    rng = RecordingRng(PufferRNG(EntropyPool()))
    assert len(collect.public_random(rng, "salt")(16)) == 16  # type: ignore[arg-type]
    assert rng.calls == [("salt", 16, "salt")]
    old = RecordingRng(PufferRNG(EntropyPool()), with_salt=False)
    assert len(collect.public_random(old, "salt")(16)) == 16  # type: ignore[arg-type]
    assert old.calls == [("generate", 16, "salt")]


def test_collect_with_panel_on_a_plain_console() -> None:
    engine = FakeEngine(FakeConfig())
    buffer = io.StringIO()
    console = Console(file=buffer, width=120)
    source = collect.SourceInfo("synthetic:seed=7", "synthetic")
    result = collect.collect_with_panel(
        engine, target_bits=200.0, timeout_s=5.0, reachable_bits=8192.0, console=console,
        source=source, fish_bits=24.0,
    )  # fmt: skip
    assert result.met
    text = buffer.getvalue()
    assert text.endswith("\n")
    assert "Collecting fish entropy" in text
    assert "(simulated aquarium)" in text
    assert "● Biscuit" in text
    assert "(3 in view)" in text


# -------------------------------------------------------------------- rendering


def _text(renderable: object, width: int = 400) -> str:
    """Plain text of ``renderable`` on a console wide enough that nothing wraps."""
    console = Console(file=io.StringIO(), width=width, record=True)
    console.print(renderable)
    return " ".join(console.export_text().split())


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.0, "·" * 8), (1.0, "█" * 8), (0.5, "████····"), (2.0, "█" * 8), (-1.0, "·" * 8)],
)
def test_bar(fraction: float, expected: str) -> None:
    assert render.bar(fraction, 8, "red").plain == expected


def test_bar_handles_nan_and_eighths() -> None:
    assert render.bar(math.nan, 4, "red").plain == "····"
    assert render.bar(0.3, 4, "red").plain == "█▏··"


def test_fish_colours_are_validated() -> None:
    assert render.fish_color("f1", "#AABBCC") == "#aabbcc"
    for hostile in ("red", "[bold]#fff", "#12345", None, 7):
        colour = render.fish_color("f1", hostile)
        assert colour.startswith("#")
        assert len(colour) == 7
    assert render.fish_color("f9") == render.fish_color("f9")  # stable fallback


def test_fish_labels_skip_malformed_cards() -> None:
    labels = render.fish_labels(
        [{"fish_id": "f1", "name": "Nemo", "color": "#010203"}, {"name": "no id"}, {"fish_id": ""}]
    )
    assert labels == {"f1": render.FishLabel("Nemo", "#010203")}


def test_human_bytes() -> None:
    assert [render.human_bytes(n) for n in (5, 1536, 3 << 20, 5 << 30)] == [
        "5 B",
        "1.5 KiB",
        "3.0 MiB",
        "5.0 GiB",
    ]


def test_attribution_is_markup_safe_and_honest() -> None:
    pool = EntropyPool()
    engine = FakeEngine(FakeConfig(), fish=(("f1", "[bold red]Evil[/]", "#ff0000"),))
    engine.pool = pool
    engine.rng = RecordingRng(PufferRNG(pool, context=engine._context))
    engine.collect(150.0, timeout_s=1.0)
    _, receipt = engine.rng.generate(32, "key")

    text = _text(render.attribution(receipt, heading="key", fish={}, simulated=True))

    assert "[bold red]Evil[/]" in text
    assert "fish - share of the fish channel" in text
    assert "os.urandom" in text
    assert "256.0" in text
    assert "never weaker than the operating system's RNG" in text
    assert render.SIMULATED_NOTE in text


def test_shortfall_panel_text() -> None:
    panel = render.shortfall_panel(
        camera_bits=1.0, target_bits=256.0, fish_bits=0.0, fish_target_bits=32.0,
        elapsed_s=2.0, reasons=["liveness never verified"], what="output",
        recorded_in="the --json receipt",
    )  # fmt: skip
    text = _text(panel)
    assert "camera 1.0 of 256 bits, fish 0.0 of 32 bits" in text
    assert "The output is still exactly as strong as os.urandom" in text
    assert "the --json receipt" in text


def test_liveness_text() -> None:
    assert "waiting" in render.liveness_text(None).plain
    disabled = render.liveness_text(LivenessStatus(LivenessState.DISABLED))
    assert "no LED challenge" in disabled.plain


def test_render_does_not_need_the_engine_modules() -> None:
    assert "puffer.cli.render" in sys.modules
