"""Fake camera backends so device logic is testable without hardware."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class FakeCapture:
    """Stands in for ``cv2.VideoCapture`` (see ``puffer.sources.base.CaptureLike``)."""

    def __init__(
        self,
        frames: list[Any] | None = None,
        *,
        opened: bool = True,
        fps: float = 30.0,
        failures: int = 0,
    ) -> None:
        self.frames = list(frames) if frames is not None else []
        self.opened = opened
        self.props: dict[int, float] = {cv2.CAP_PROP_FPS: fps}
        self.failures = failures
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        if self.failures > 0:
            self.failures -= 1
            return False, None
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop_id: int) -> float:
        return self.props.get(prop_id, 0.0)

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = value
        return True

    def release(self) -> None:
        self.released = True


class FakePicamera2:
    """Stands in for ``picamera2.Picamera2``."""

    def __init__(self, frames: list[Any] | None = None, *, fail_start: bool = False) -> None:
        self.frames = list(frames) if frames is not None else []
        self.fail_start = fail_start
        self.config: dict[str, Any] | None = None
        self.started = False
        self.stopped = False
        self.closed = False

    def create_video_configuration(
        self, main: dict[str, Any], controls: dict[str, Any]
    ) -> dict[str, Any]:
        return {"main": main, "controls": controls}

    def configure(self, config: dict[str, Any]) -> None:
        self.config = config

    def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("camera busy")
        self.started = True

    def capture_array(self, stream: str) -> Any:
        assert stream == "main"
        return self.frames.pop(0) if self.frames else np.zeros((4, 4, 3), np.uint8)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True
