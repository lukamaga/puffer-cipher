"""Live cameras: any OpenCV-supported device, and the Raspberry Pi camera.

Live frames are timestamped with ``time.monotonic()`` immediately after
capture. That is the clock the LED controller stamps its commands with, so
the liveness verifier can treat the whole capture pipeline delay as one
constant lag and estimate it.

Frames are tagged ``LightMode.NIGHT`` when the camera is declared to be in
IR mode (``ir=True``) or when the frame is monochrome (a NoIR camera under
an 850 nm illuminator, or a camera that switched to its night mode).

Neither class can be exercised without hardware, so both take a factory
for the underlying device: tests pass a fake capture object and drive all
of the logic (configuration, retries, colour handling, timestamps).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from puffer.sources.base import CaptureLike, FrameSourceBase, SourceError, is_monochrome, to_bgr
from puffer.types import Frame, LightMode

__all__ = ["OpenCVCamera", "Picamera2Camera"]


def _open_device(index: int | str) -> CaptureLike:
    return cv2.VideoCapture(index)


class OpenCVCamera(FrameSourceBase):
    """A webcam / V4L2 / UVC device (or a stream URL) through ``cv2.VideoCapture``.

    Requested ``width``/``height``/``fps`` are hints: drivers pick the
    nearest supported mode, so the actual frame rate is read back. A few
    consecutive read failures (``max_read_failures``) are retried before the
    camera is reported as exhausted (``read() -> None``), which the engine's
    ``source_timeout`` health check then turns into a FAILED channel.
    """

    def __init__(
        self,
        index: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        ir: bool = False,
        *,
        max_read_failures: int = 5,
        capture_factory: Callable[[int | str], CaptureLike] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_read_failures < 1:
            raise ValueError("max_read_failures must be >= 1")
        self.index = index
        self.ir = ir
        self.max_read_failures = max_read_failures
        self._clock = clock
        cap = (capture_factory or _open_device)(index)
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"cannot open camera {index!r} (is it connected and not used by another process?)"
            )
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        actual = float(cap.get(cv2.CAP_PROP_FPS))
        self.fps = actual if math.isfinite(actual) and actual > 0 else float(fps or 30.0)
        self.source_id = f"camera:{index}"
        self._cap: CaptureLike | None = cap
        self._index = 0

    def read(self) -> Frame | None:
        """Grab the next frame; ``None`` after repeated failures or close."""
        if self._cap is None:
            return None
        for _ in range(self.max_read_failures):
            ok, image = self._cap.read()
            t = self._clock()
            if ok and image is not None:
                return self._frame(image, t)
        return None

    def _frame(self, image: Any, t: float) -> Frame:
        bgr = to_bgr(np.asarray(image, dtype=np.uint8))
        night = self.ir or is_monochrome(bgr)
        frame = Frame(
            index=self._index,
            timestamp=t,
            wall_time=time.time(),
            image=bgr,
            mode=LightMode.NIGHT if night else LightMode.DAY,
            source_id=self.source_id,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        """Release the device (idempotent)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class Picamera2Camera(FrameSourceBase):
    """Raspberry Pi camera through ``picamera2`` (imported lazily).

    Defaults follow ``docs/hardware.md``: denoising off (it removes exactly
    the temporal noise the sensor channel harvests) and, when given, fixed
    exposure and gain (auto-exposure makes noise statistics drift and can
    trip the adaptive-proportion health test). The ``RGB888`` format is
    stored in B, G, R byte order by libcamera, i.e. it is already OpenCV's
    BGR.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        ir: bool = False,
        *,
        denoise: bool = False,
        exposure_us: int | None = None,
        analogue_gain: float | None = None,
        camera_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if camera_factory is None:
            try:
                from picamera2 import Picamera2  # optional, Pi only
            except ImportError as exc:
                raise SourceError(
                    "picamera2 is not installed. On Raspberry Pi OS run "
                    "'sudo apt install python3-picamera2' and create the virtualenv with "
                    "--system-site-packages (see docs/hardware.md); elsewhere use 'camera:0'"
                ) from exc
            camera_factory = Picamera2
        controls: dict[str, Any] = {"FrameRate": float(fps)}
        if not denoise:
            controls["NoiseReductionMode"] = 0  # libcamera draft NoiseReductionModeEnum.Off
        if exposure_us is not None or analogue_gain is not None:
            controls["AeEnable"] = False
            if exposure_us is not None:
                controls["ExposureTime"] = int(exposure_us)
            if analogue_gain is not None:
                controls["AnalogueGain"] = float(analogue_gain)
        camera = camera_factory()
        try:
            config = camera.create_video_configuration(
                main={"size": (int(width), int(height)), "format": "RGB888"}, controls=controls
            )
            camera.configure(config)
            camera.start()
        except Exception as exc:
            camera.close()
            raise SourceError(f"cannot start the Raspberry Pi camera: {exc}") from exc
        self.fps = float(fps)
        self.ir = ir
        self.controls = controls
        self.source_id = "picamera"
        self._camera: Any = camera
        self._clock = clock
        self._index = 0

    def read(self) -> Frame | None:
        """Capture the next frame; ``None`` after close."""
        if self._camera is None:
            return None
        array = self._camera.capture_array("main")
        t = self._clock()
        bgr = to_bgr(np.asarray(array, dtype=np.uint8))
        night = self.ir or is_monochrome(bgr)
        frame = Frame(
            index=self._index,
            timestamp=t,
            wall_time=time.time(),
            image=bgr,
            mode=LightMode.NIGHT if night else LightMode.DAY,
            source_id=self.source_id,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        """Stop and release the camera (idempotent)."""
        if self._camera is not None:
            camera, self._camera = self._camera, None
            try:
                camera.stop()
            finally:
                camera.close()
