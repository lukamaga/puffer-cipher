"""Camera sources, driven through fake device backends."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from puffer.sources.base import SourceError
from puffer.sources.camera import OpenCVCamera, Picamera2Camera
from puffer.types import FrameSource, LightMode

from .fakes import FakeCapture, FakePicamera2


def _colour(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (6, 8, 3), dtype=np.uint8)


def test_opencv_camera_configures_device_and_reads_frames() -> None:
    capture = FakeCapture([_colour(1), _colour(2)], fps=25.0)
    ticks = iter([10.0, 10.04])
    camera = OpenCVCamera(
        1,
        width=1280,
        height=720,
        fps=25,
        capture_factory=lambda i: capture,
        clock=lambda: next(ticks),
    )
    assert isinstance(camera, FrameSource)
    assert camera.source_id == "camera:1"
    assert capture.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert capture.props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert camera.fps == 25.0
    first, second = camera.read(), camera.read()
    assert first is not None
    assert second is not None
    assert (first.index, first.timestamp, second.index, second.timestamp) == (0, 10.0, 1, 10.04)
    assert first.mode is LightMode.DAY
    assert first.image.shape == (6, 8, 3)
    camera.close()
    assert capture.released
    assert camera.read() is None
    camera.close()  # idempotent


def test_opencv_camera_falls_back_to_requested_fps() -> None:
    camera = OpenCVCamera(0, fps=15, capture_factory=lambda i: FakeCapture(fps=0.0))
    assert camera.fps == 15.0
    assert OpenCVCamera(0, capture_factory=lambda i: FakeCapture(fps=float("nan"))).fps == 30.0


def test_grayscale_and_ir_frames_are_night() -> None:
    gray = np.full((6, 8), 90, np.uint8)
    camera = OpenCVCamera(capture_factory=lambda i: FakeCapture([gray, _colour()]))
    night, day = camera.read(), camera.read()
    assert night is not None
    assert night.mode is LightMode.NIGHT
    assert night.image.shape == (6, 8, 3)
    assert day is not None
    assert day.mode is LightMode.DAY
    ir = OpenCVCamera(capture_factory=lambda i: FakeCapture([_colour()]), ir=True)
    frame = ir.read()
    assert frame is not None
    assert frame.mode is LightMode.NIGHT


def test_bgra_frames_lose_alpha() -> None:
    bgra = np.dstack([_colour(), np.full((6, 8), 255, np.uint8)])
    frame = OpenCVCamera(capture_factory=lambda i: FakeCapture([bgra])).read()
    assert frame is not None
    assert np.array_equal(frame.image, bgra[..., :3])


def test_transient_read_failures_are_retried() -> None:
    capture = FakeCapture([_colour()], failures=2)
    camera = OpenCVCamera(capture_factory=lambda i: capture, max_read_failures=3)
    assert camera.read() is not None
    dead = OpenCVCamera(capture_factory=lambda i: FakeCapture(failures=10), max_read_failures=3)
    assert dead.read() is None


def test_unopenable_camera_raises_and_releases() -> None:
    capture = FakeCapture(opened=False)
    with pytest.raises(SourceError, match="cannot open camera 3"):
        OpenCVCamera(3, capture_factory=lambda i: capture)
    assert capture.released
    with pytest.raises(ValueError, match="max_read_failures"):
        OpenCVCamera(capture_factory=lambda i: FakeCapture(), max_read_failures=0)


def test_picamera2_configuration_and_capture() -> None:
    fake = FakePicamera2([_colour(3), np.dstack([_colour(4), np.zeros((6, 8), np.uint8)])])
    ticks = iter([1.0, 1.1])
    with Picamera2Camera(
        1280, 720, 24.0, exposure_us=8000, camera_factory=lambda: fake, clock=lambda: next(ticks)
    ) as cam:
        assert fake.started
        assert cam.source_id == "picamera"
        assert cam.fps == 24.0
        assert fake.config is not None
        assert fake.config["main"] == {"size": (1280, 720), "format": "RGB888"}
        controls = fake.config["controls"]
        assert controls["NoiseReductionMode"] == 0  # denoise off by default
        assert controls["AeEnable"] is False
        assert controls["ExposureTime"] == 8000
        assert controls["FrameRate"] == 24.0
        first, second = cam.read(), cam.read()
        assert first is not None
        assert np.array_equal(first.image, _colour(3))
        assert first.timestamp == 1.0
        assert second is not None
        assert second.image.shape == (6, 8, 3)
        assert second.index == 1
    assert fake.stopped
    assert fake.closed
    assert cam.read() is None


def test_picamera2_options() -> None:
    fake = FakePicamera2()
    Picamera2Camera(denoise=True, analogue_gain=2.0, ir=True, camera_factory=lambda: fake)
    assert fake.config is not None
    controls = fake.config["controls"]
    assert "NoiseReductionMode" not in controls
    assert controls["AnalogueGain"] == 2.0
    frame = Picamera2Camera(ir=True, camera_factory=FakePicamera2).read()
    assert frame is not None
    assert frame.mode is LightMode.NIGHT


def test_picamera2_start_failure_releases_camera() -> None:
    fake = FakePicamera2(fail_start=True)
    with pytest.raises(SourceError, match="camera busy"):
        Picamera2Camera(camera_factory=lambda: fake)
    assert fake.closed


def test_missing_picamera2_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "picamera2", None)
    with pytest.raises(SourceError, match="python3-picamera2"):
        Picamera2Camera()
