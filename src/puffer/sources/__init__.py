"""Frame sources: the aquarium simulator, video files and live cameras.

Every source implements :class:`puffer.types.FrameSource`. Most callers only
need :func:`open_source` with a spec string (``"synthetic"``,
``"video:tank.mp4,loop=1"``, ``"camera:0"``, ``"picamera"``).
"""

from puffer.sources.base import FrameSourceBase, RealtimePacer, SourceError, iter_frames
from puffer.sources.camera import OpenCVCamera, Picamera2Camera
from puffer.sources.recorder import record
from puffer.sources.spec import SPEC_HELP, SourceSpecError, open_source
from puffer.sources.synthetic import (
    DEFAULT_CAST,
    FishSpec,
    FishTruth,
    SyntheticAquarium,
    SyntheticConfig,
)
from puffer.sources.video import VideoFileSource

__all__ = [
    "DEFAULT_CAST",
    "SPEC_HELP",
    "FishSpec",
    "FishTruth",
    "FrameSourceBase",
    "OpenCVCamera",
    "Picamera2Camera",
    "RealtimePacer",
    "SourceError",
    "SourceSpecError",
    "SyntheticAquarium",
    "SyntheticConfig",
    "VideoFileSource",
    "iter_frames",
    "open_source",
    "record",
]
