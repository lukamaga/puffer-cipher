"""``puffer simulate`` renders a real (tiny) simulator video that OpenCV can play back."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from .conftest import Invoke

SMALL = ("--seconds", "1", "--fps", "10", "--width", "320", "--height", "240", "--seed", "7")


def _read_back(path: Path) -> tuple[int, tuple[int, ...] | None]:
    capture = cv2.VideoCapture(str(path))
    frames, shape = 0, None
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                return frames, shape
            frames, shape = frames + 1, image.shape
    finally:
        capture.release()


def test_simulate_writes_a_playable_video(cli: Invoke) -> None:
    result = cli("simulate", "-o", "demo.mp4", *SMALL, "--fish", "3")

    assert result.exit_code == 0, result.output
    assert _read_back(Path("demo.mp4")) == (10, (240, 320, 3))
    out = result.stdout
    assert "10 (1.0 s at 10 fps)" in out
    assert "3 fish, day, seed 7" in out
    assert "mp4v is lossy" in out  # honest about what lossy codecs do to sensor noise
    assert "anti-replay" in out
    assert not list(Path().glob(".demo.partial*"))  # the temporary file was renamed


def test_lossless_container_and_no_led(cli: Invoke) -> None:
    result = cli("simulate", "-o", "night.mkv", *SMALL, "--mode", "night", "--no-led")

    assert result.exit_code == 0, result.output
    assert _read_back(Path("night.mkv"))[0] == 10
    assert "codec FFV1" in result.stdout
    assert "lossy" not in result.stdout
    assert "anti-replay" not in result.stdout


def test_refuses_to_overwrite(cli: Invoke) -> None:
    Path("demo.mp4").write_bytes(b"keep")

    result = cli("simulate", *SMALL)

    assert result.exit_code == 3
    assert Path("demo.mp4").read_bytes() == b"keep"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("-o", "demo.gif"), "no default codec for .gif"),
        (("-o", "demo.avi", "--codec", "MJPEG"), "exactly 4 characters"),
        (("--fish", "40"), "40"),
    ],
)
def test_usage_errors(cli: Invoke, args: tuple[str, ...], message: str) -> None:
    result = cli("simulate", *args)

    assert result.exit_code == 2
    assert message in result.output
