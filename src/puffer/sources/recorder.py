"""Record any frame source to a video file.

Used by ``puffer simulate`` (render the simulator to a demo video) and to
capture real aquarium footage for repeatable experiments. Note that lossy
codecs (the default ``mp4v``) smear exactly the low-order pixel noise the
sensor entropy channel measures; record with a lossless codec (``FFV1`` in
``.mkv``, or ``png `` in ``.avi``) when the recording is meant for entropy
analysis rather than for people.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import cv2

from puffer.sources.base import SourceError, iter_frames
from puffer.types import Frame, FrameSource

__all__ = ["record"]


def record(
    source: FrameSource,
    path: str | os.PathLike[str],
    *,
    max_frames: int | None = None,
    seconds: float | None = None,
    fourcc: str = "mp4v",
    fps: float | None = None,
    on_frame: Callable[[Frame], None] | None = None,
) -> int:
    """Write frames from ``source`` to ``path``; return the number written.

    Stops at ``max_frames``, after ``seconds`` of *source time* (frame
    timestamps, so a simulator records exactly ``seconds * fps`` frames no
    matter how fast it renders), or when the source is exhausted, whichever
    comes first. The output frame rate defaults to ``source.fps``. The
    source is not closed. ``on_frame`` is called after each written frame
    (progress bars).
    """
    if len(fourcc) != 4:
        raise ValueError(f"fourcc must be exactly 4 characters, got {fourcc!r}")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be >= 0")
    if seconds is not None and not seconds > 0:
        raise ValueError("seconds must be > 0")
    out_fps = float(fps if fps is not None else source.fps)
    if not out_fps > 0:
        raise ValueError("output fps must be > 0")

    writer: cv2.VideoWriter | None = None
    size: tuple[int, int] | None = None
    first_t: float | None = None
    count = 0
    try:
        for frame in iter_frames(source, max_frames):
            if seconds is not None:
                if first_t is None:
                    first_t = frame.timestamp
                elif frame.timestamp - first_t >= seconds - 1e-9:
                    break
            if writer is None:
                size = (frame.width, frame.height)
                writer = cv2.VideoWriter(
                    os.fspath(path), cv2.VideoWriter.fourcc(*fourcc), out_fps, size
                )
                if not writer.isOpened():
                    raise SourceError(f"cannot write {os.fspath(path)!r} with codec {fourcc!r}")
            elif (frame.width, frame.height) != size:
                raise SourceError(
                    f"frame size changed during recording: {size} -> {(frame.width, frame.height)}"
                )
            writer.write(frame.image)
            count += 1
            if on_frame is not None:
                on_frame(frame)
    finally:
        if writer is not None:
            writer.release()
    return count
