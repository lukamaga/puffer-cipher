"""Source spec parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from puffer.sources import camera as camera_module
from puffer.sources.base import SourceError
from puffer.sources.camera import OpenCVCamera
from puffer.sources.recorder import record
from puffer.sources.spec import SPEC_HELP, SourceSpecError, open_source
from puffer.sources.synthetic import SyntheticAquarium, SyntheticConfig
from puffer.sources.video import VideoFileSource

from .conftest import PeriodicLed
from .fakes import FakeCapture


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("spec") / "clip,with comma.mp4"
    record(
        SyntheticAquarium(SyntheticConfig(seed=1, width=160, height=120, led_xy=(150, 8))),
        path,
        max_frames=3,
    )
    return path


def test_default_synthetic() -> None:
    source = open_source("synthetic")
    assert isinstance(source, SyntheticAquarium)
    assert source.config.n_fish == 6
    assert source.config.mode == "day"


def test_synthetic_options() -> None:
    led = PeriodicLed(1.0)
    source = open_source(
        "synthetic:fish=8,seed=42,mode=night,latency=0.08,realtime=1,fps=24,jpeg=80,bubbles=off,led=600:40",
        led_schedule=led,
    )
    assert isinstance(source, SyntheticAquarium)
    c = source.config
    assert (c.n_fish, c.seed, c.mode, c.led_latency_s, c.realtime, c.fps) == (
        8,
        42,
        "night",
        0.08,
        True,
        24.0,
    )
    assert (c.jpeg_quality, c.bubbles, c.led_xy) == (80, False, (600, 40))
    assert source.led_schedule is led


def test_synthetic_misc_values() -> None:
    source = open_source(
        " Synthetic : seed=random , start=now, day=30, night=10, mode=cycle, sway=0 "
    )
    assert isinstance(source, SyntheticAquarium)
    assert source.config.plant_sway == 0.0
    assert source.config.seed is None
    assert source.config.start_time > 0
    assert (source.config.cycle_day_s, source.config.cycle_night_s) == (30.0, 10.0)


def test_video_specs(clip: Path) -> None:
    plain = open_source(f"video:{clip}")
    assert isinstance(plain, VideoFileSource)
    assert not plain.loop
    assert plain.path == clip
    looped = open_source(f"video:{clip},loop=1,start=2.5")
    assert isinstance(looped, VideoFileSource)
    assert looped.loop
    assert looped.start_time == 2.5
    bare = open_source(f"{clip},loop=yes")
    assert isinstance(bare, VideoFileSource)
    assert bare.loop


def test_camera_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[int | str] = []

    def fake_device(index: int | str) -> FakeCapture:
        opened.append(index)
        return FakeCapture()

    monkeypatch.setattr(camera_module, "_open_device", fake_device)
    cam = open_source("camera:0,ir=1,width=1280,height=720")
    assert isinstance(cam, OpenCVCamera)
    assert cam.ir
    assert cam.index == 0
    assert isinstance(open_source("2"), OpenCVCamera)
    assert isinstance(open_source("camera:/dev/video4,fps=15"), OpenCVCamera)
    assert isinstance(open_source("camera"), OpenCVCamera)
    assert opened == [0, 2, "/dev/video4", 0]


def test_picamera_spec_without_picamera2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "picamera2", None)
    with pytest.raises(SourceError, match="picamera2"):
        open_source("picamera:width=1280,height=720,denoise=0")


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("", "empty source spec"),
        ("bogus", "unknown source kind 'bogus'"),
        ("synthetc:fish=2", "did you mean 'synthetic'"),
        ("synthetic:fsh=3", "did you mean 'fish'"),
        ("synthetic:fish", "expected key=value"),
        ("synthetic:fish=lots", "bad value 'lots' for 'fish'.*integer"),
        ("synthetic:realtime=maybe", "boolean"),
        ("synthetic:fps=fast", "expected a number"),
        ("synthetic:mode=dusk", "day, night or cycle"),
        ("synthetic:led=5", "X:Y"),
        ("synthetic:fish=99", "invalid simulator settings"),
        ("video:", "missing video path"),
        ("camera:0,foo=1", "unknown option 'foo'"),
    ],
)
def test_bad_specs(spec: str, message: str) -> None:
    with pytest.raises(SourceSpecError, match=message):
        open_source(spec)


def test_missing_video_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_source(f"video:{tmp_path / 'nope.mp4'}")


def test_help_mentions_every_kind_and_option() -> None:
    for word in ("synthetic", "video", "camera", "picamera", "latency", "loop", "ir", "denoise"):
        assert word in SPEC_HELP
