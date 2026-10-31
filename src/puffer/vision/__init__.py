"""Computer vision for puffer-cipher: find, follow and recognise the fish.

The pipeline (:class:`FishTracker`) is detector -> :class:`ByteTracker`
(constant-velocity Kalman + two-stage IoU association) -> :class:`FishReID`
(persistent identities) -> :class:`MotionFeatures` (velocity, heading,
acceleration, turn rate, Kalman innovation). The innovations are the raw
material of the fish entropy channel; everything else serves tracking
quality, attribution ("which fish made this key") and the dashboard.

No cryptography lives here - only statistics and image processing.
"""

from puffer.vision.bytetrack import ByteTracker, Track, TrackState
from puffer.vision.detector import MotionDetector, is_grayscale
from puffer.vision.features import FishCharacter, MotionFeatures
from puffer.vision.geometry import bbox_to_xyah, iou_matrix, linear_assignment, xyah_to_bbox
from puffer.vision.kalman import CHI2INV95, KalmanFilter, KalmanUpdate
from puffer.vision.pipeline import FishTracker
from puffer.vision.reid import FISH_NAMES, FishIdentity, FishReID
from puffer.vision.yolo import YoloDetector

__all__ = [
    "CHI2INV95",
    "FISH_NAMES",
    "ByteTracker",
    "FishCharacter",
    "FishIdentity",
    "FishReID",
    "FishTracker",
    "KalmanFilter",
    "KalmanUpdate",
    "MotionDetector",
    "MotionFeatures",
    "Track",
    "TrackState",
    "YoloDetector",
    "bbox_to_xyah",
    "iou_matrix",
    "is_grayscale",
    "linear_assignment",
    "xyah_to_bbox",
]
