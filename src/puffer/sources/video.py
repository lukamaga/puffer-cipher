"""Recorded video files as a frame source.

Recordings make experiments repeatable (the same fish, again) and are the
MVP source before a camera is installed. They are also exactly what a replay
attack looks like, which is why liveness defaults to ``off`` for video files
in the engine and a recording can never pass the LED challenge.

Timestamps come from the frame index (``start_time + index / fps``), not
from the container's presentation timestamps: they are monotonic by
construction, keep increasing across loops, and do not depend on the codec.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import cv2

from puffer.sources.base import (
    CaptureLike,
    FrameSourceBase,
    RealtimePacer,
    SourceError,
    is_monochrome,
    to_bgr,
)
from puffer.types import Frame, LightMode

__all__ = ["DEFAULT_FPS", "VideoFileSource"]

DEFAULT_FPS = 30.0
"""Used when a container does not report a usable frame rate."""


def _open_file(path: str) -> CaptureLike:
    return cv2.VideoCapture(path)


class VideoFileSource(FrameSourceBase):
    """Frames from a video file (anything OpenCV/FFmpeg can decode).

    ``loop=True`` restarts at the end (by reopening the file, which works
    for every container, unlike seeking). ``realtime=True`` paces reads to
    the file's frame rate, for the live dashboard. Frames whose three
    channels are (nearly) equal are tagged ``LightMode.NIGHT`` when
    ``detect_night`` is set: a grayscale recording is an IR recording.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        loop: bool = False,
        realtime: bool = False,
        start_time: float = 0.0,
        detect_night: bool = True,
        capture_factory: Callable[[str], CaptureLike] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        self.loop = loop
        self.start_time = start_time
        self.detect_night = detect_night
        self._factory = capture_factory or _open_file
        self._cap: CaptureLike | None = self._open()
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if math.isfinite(fps) and fps > 0 else DEFAULT_FPS
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count: int | None = int(count) if math.isfinite(count) and count > 0 else None
        self.source_id = f"video:{self.path.name}"
        self.loops = 0
        self._index = 0
        self._pass_frames = 0
        self._pacer = RealtimePacer(self.fps) if realtime else None

    def _open(self) -> CaptureLike:
        cap = self._factory(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"cannot decode video file {self.path} (unsupported codec or corrupt file)"
            )
        return cap

    def read(self) -> Frame | None:
        """Next frame; ``None`` at the end of a non-looping file or after close."""
        if self._cap is None:
            return None
        ok, image = self._cap.read()
        if not ok or image is None:
            # An empty file must not loop forever.
            if not self.loop or self._pass_frames == 0:
                return None
            self._cap.release()
            self._cap = self._open()
            self.loops += 1
            self._pass_frames = 0
            ok, image = self._cap.read()
            if not ok or image is None:
                return None
        self._pass_frames += 1
        if self._pacer is not None:
            self._pacer.wait()
        bgr = to_bgr(image)
        night = self.detect_night and is_monochrome(bgr)
        frame = Frame(
            index=self._index,
            timestamp=self.start_time + self._index / self.fps,
            wall_time=time.time(),
            image=bgr,
            mode=LightMode.NIGHT if night else LightMode.DAY,
            source_id=self.source_id,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        """Release the decoder (idempotent)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
