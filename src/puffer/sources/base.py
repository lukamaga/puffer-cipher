"""Building blocks shared by every frame source.

* :class:`FrameSourceBase` - context-manager support for the
  :class:`puffer.types.FrameSource` protocol.
* :func:`iter_frames` - the canonical "read until exhausted" loop.
* :class:`RealtimePacer` - sleeps so that a source that can produce frames
  faster than real time (simulator, video file) is delivered at its nominal
  frame rate, e.g. for the live dashboard.
* :func:`is_monochrome` and :func:`to_bgr` - normalisation helpers that make
  every source honour the :class:`puffer.types.Frame` image contract
  (HxWx3 uint8 BGR, grayscale replicated into three channels).
"""

from __future__ import annotations

import abc
import time
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, Protocol, Self, cast

import cv2
import numpy as np
import numpy.typing as npt

from puffer.types import Frame, FrameSource

__all__ = [
    "CaptureLike",
    "FrameSourceBase",
    "RealtimePacer",
    "SourceError",
    "is_monochrome",
    "iter_frames",
    "to_bgr",
]


_U8 = npt.NDArray[np.uint8]


class SourceError(RuntimeError):
    """A frame source could not be opened or its backend is unavailable."""


class CaptureLike(Protocol):
    """The subset of ``cv2.VideoCapture`` the file and camera sources use.

    Typing against this protocol (instead of ``cv2.VideoCapture``) lets tests
    substitute a fake capture object, so camera logic is testable without a
    camera.
    """

    def isOpened(self) -> bool: ...  # noqa: N802 - OpenCV's method name

    def read(self) -> tuple[bool, Any]: ...

    def get(self, prop_id: int) -> float: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def release(self) -> None: ...


class FrameSourceBase(abc.ABC):
    """Abstract base giving concrete sources ``with`` support.

    Subclasses set ``source_id`` and ``fps`` and implement :meth:`read` and
    :meth:`close`; leaving a ``with`` block always closes the device, which
    matters for cameras that otherwise stay locked by the process.
    """

    source_id: str
    fps: float

    @abc.abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the source is exhausted."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the underlying device or file (idempotent)."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def iter_frames(source: FrameSource, max_frames: int | None = None) -> Iterator[Frame]:
    """Yield frames until the source is exhausted or ``max_frames`` were read.

    The source is *not* closed: whoever opened it owns it (a camera may be
    iterated several times, e.g. calibration then capture).
    """
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be >= 0")
    count = 0
    while max_frames is None or count < max_frames:
        frame = source.read()
        if frame is None:
            return
        count += 1
        yield frame


class RealtimePacer:
    """Sleep so that successive :meth:`wait` calls return at ``fps``.

    Deadlines are absolute (``t0 + k / fps``), so per-frame jitter does not
    accumulate into drift. If the consumer falls more than ``max_lag_s``
    behind (process suspended, slow machine) the schedule is re-anchored
    instead of delivering a burst of frames to "catch up" - for a live view
    a skipped beat is better than a fast-forward.

    ``clock`` and ``sleep`` are injectable for deterministic tests.
    """

    def __init__(
        self,
        fps: float,
        *,
        max_lag_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not fps > 0:
            raise ValueError("fps must be > 0")
        self.period = 1.0 / fps
        self.max_lag_s = max_lag_s
        self._clock = clock
        self._sleep = sleep
        self._next: float | None = None

    def reset(self) -> None:
        """Forget the schedule; the next :meth:`wait` returns immediately."""
        self._next = None

    def wait(self) -> float:
        """Block until the next frame is due; return the current clock value."""
        now = self._clock()
        if self._next is None or now - self._next > self.max_lag_s:
            self._next = now + self.period
            return now
        delay = self._next - now
        if delay > 0:
            self._sleep(delay)
            now = self._clock()
        self._next += self.period
        return now


def is_monochrome(
    image: npt.NDArray[np.uint8], *, tolerance: float = 3.0, max_tint: float = 12.0, step: int = 7
) -> bool:
    """Heuristically decide whether a BGR image is really grayscale.

    IR cameras and grayscale video files decode to three (nearly) equal
    channels. Video codecs store colour as YUV, and the conversion back can
    leave a *constant* offset between channels (a faint tint) plus a little
    chroma noise. So the test is that the channel differences barely *vary*
    across the image (standard deviation within ``tolerance``) and that any
    constant tint stays below ``max_tint``. A strided subsample keeps this
    O(pixels / step^2).
    """
    if image.ndim == 2 or image.shape[2] == 1:
        return True
    sub = image[::step, ::step, :3].astype(np.int16)
    diffs = (sub[..., 0] - sub[..., 1], sub[..., 1] - sub[..., 2])
    spread = max(float(d.std()) for d in diffs)
    tint = max(abs(float(d.mean())) for d in diffs)
    return spread <= tolerance and tint <= max_tint


def to_bgr(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Convert a grayscale / BGRA / BGR uint8 image to contiguous HxWx3 BGR."""
    if image.dtype != np.uint8:
        raise TypeError(f"expected a uint8 image, got {image.dtype}")
    if image.ndim == 2:
        return cast(_U8, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    if image.ndim == 3 and image.shape[2] == 1:
        return cast(_U8, cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR))
    if image.ndim == 3 and image.shape[2] == 4:
        return cast(_U8, cv2.cvtColor(image, cv2.COLOR_BGRA2BGR))
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image)
    raise ValueError(f"unsupported image shape {image.shape}")
