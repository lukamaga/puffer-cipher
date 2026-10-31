"""Recording sources to video files and reading them back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from puffer.sources.base import SourceError, iter_frames
from puffer.sources.recorder import record
from puffer.sources.synthetic import SyntheticAquarium, SyntheticConfig
from puffer.sources.video import VideoFileSource
from puffer.types import Frame, FrameSource, LightMode


@pytest.fixture(scope="module")
def day_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "tank.mp4"
    source = SyntheticAquarium(
        SyntheticConfig(seed=3, width=320, height=240, led_xy=(300, 16), fps=20.0)
    )
    assert record(source, path, max_frames=12) == 12
    return path


def test_record_then_read_back(day_video: Path) -> None:
    with VideoFileSource(day_video) as video:
        assert isinstance(video, FrameSource)
        assert video.fps == pytest.approx(20.0)
        assert (video.width, video.height) == (320, 240)
        assert video.frame_count == 12
        assert video.source_id == "video:tank.mp4"
        frames = list(iter_frames(video))
    assert len(frames) == 12
    assert [f.index for f in frames] == list(range(12))
    assert [f.timestamp for f in frames] == pytest.approx([i / 20.0 for i in range(12)])
    assert all(f.image.shape == (240, 320, 3) and f.image.dtype == np.uint8 for f in frames)
    assert all(f.mode is LightMode.DAY for f in frames)
    assert video.read() is None  # closed


def test_recording_preserves_content(tmp_path: Path) -> None:
    path = tmp_path / "mjpeg.avi"
    source = SyntheticAquarium(SyntheticConfig(seed=4, width=320, height=240, led_xy=(300, 16)))
    reference = SyntheticAquarium(SyntheticConfig(seed=4, width=320, height=240, led_xy=(300, 16)))
    record(source, path, max_frames=3, fourcc="MJPG")
    with VideoFileSource(path) as video:
        for frame in iter_frames(video):
            original = reference.read()
            assert original is not None
            diff = np.abs(frame.image.astype(int) - original.image.astype(int))
            assert diff.mean() < 6.0


def test_looping_keeps_timestamps_increasing(day_video: Path) -> None:
    with VideoFileSource(day_video, loop=True, start_time=5.0) as video:
        frames = list(iter_frames(video, max_frames=30))
        assert video.loops == 2
    assert [f.index for f in frames] == list(range(30))
    times = np.array([f.timestamp for f in frames])
    assert times[0] == 5.0
    assert np.all(np.diff(times) > 0)
    assert np.array_equal(frames[0].image, frames[12].image)


def test_record_seconds_counts_source_time(tmp_path: Path) -> None:
    source = SyntheticAquarium(
        SyntheticConfig(seed=1, width=320, height=240, led_xy=(300, 16), fps=30.0)
    )
    seen: list[int] = []
    count = record(
        source, tmp_path / "half.mp4", seconds=0.5, on_frame=lambda f: seen.append(f.index)
    )
    assert count == 15
    assert seen == list(range(15))


def test_grayscale_recordings_are_night(tmp_path: Path) -> None:
    path = tmp_path / "night.mp4"
    source = SyntheticAquarium(
        SyntheticConfig(seed=2, width=320, height=240, led_xy=(300, 16), mode="night")
    )
    record(source, path, max_frames=4)
    with VideoFileSource(path) as video:
        assert all(f.mode is LightMode.NIGHT for f in iter_frames(video))
    with VideoFileSource(path, detect_night=False) as video:
        frame = video.read()
        assert frame is not None
        assert frame.mode is LightMode.DAY


def test_realtime_video_is_paced(day_video: Path) -> None:
    import time

    with VideoFileSource(day_video, realtime=True) as video:
        start = time.perf_counter()
        list(iter_frames(video, max_frames=5))
        assert time.perf_counter() - start >= 4 / 20.0 - 0.01


def test_video_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VideoFileSource(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(SourceError, match="cannot decode"):
        VideoFileSource(junk)


def test_empty_loop_does_not_spin(tmp_path: Path) -> None:
    class _Empty:
        def __init__(self) -> None:
            self.opened = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, None]:
            return False, None

        def get(self, prop_id: int) -> float:
            return 0.0

        def set(self, prop_id: int, value: float) -> bool:
            return True

        def release(self) -> None:
            pass

    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    video = VideoFileSource(path, loop=True, capture_factory=lambda _: _Empty())
    assert video.fps == 30.0
    assert video.frame_count is None
    assert video.read() is None


def test_record_rejects_a_changing_frame_size(tmp_path: Path) -> None:
    class _Resizing:
        source_id, fps = "resizing", 10.0

        def __init__(self) -> None:
            self.i = 0

        def read(self) -> Frame | None:
            self.i += 1
            size = 32 if self.i < 3 else 48
            return Frame(self.i, self.i / 10.0, 0.0, np.zeros((size, size, 3), np.uint8))

        def close(self) -> None:
            pass

    with pytest.raises(SourceError, match="frame size changed"):
        record(_Resizing(), tmp_path / "resize.avi", fourcc="MJPG", max_frames=5)


def test_record_argument_validation(tmp_path: Path) -> None:
    source: FrameSource = SyntheticAquarium(
        SyntheticConfig(seed=1, width=320, height=240, led_xy=(300, 16))
    )
    with pytest.raises(ValueError, match="fourcc"):
        record(source, tmp_path / "x.mp4", fourcc="h264x")
    with pytest.raises(ValueError, match="seconds"):
        record(source, tmp_path / "x.mp4", seconds=0)
    with pytest.raises(ValueError, match="max_frames"):
        record(source, tmp_path / "x.mp4", max_frames=-2)
    with pytest.raises(SourceError, match="cannot write"):
        record(source, tmp_path / "no" / "such" / "dir" / "x.mp4", max_frames=1)
