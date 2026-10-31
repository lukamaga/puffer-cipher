"""Optional neural detector backed by Ultralytics YOLO.

The motion detector needs no training data and is the default; a YOLO
model trained on the actual tank is more robust to plants swaying and
bubbles, and - when trained with **one class per individual fish** -
doubles as a re-identification oracle: each :class:`Detection` then carries
the fish's name as its ``label`` and :class:`~puffer.vision.reid.FishReID`
(``trust_labels=True``) uses it directly.

``ultralytics`` (and with it PyTorch) is a heavy optional dependency, so it
is imported lazily, only when a weights path is given. Anything that
quacks like an Ultralytics model (``predict(...)`` returning results with
``boxes.xyxy / conf / cls`` and ``names``) can be passed instead, which is
how the tests run without PyTorch.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from puffer.types import Detection, Frame

_INSTALL_HINT = (
    "YoloDetector needs the optional 'ultralytics' package: pip install 'puffer-cipher[yolo]'"
)


@runtime_checkable
class YoloModel(Protocol):
    """The subset of ``ultralytics.YOLO`` used here."""

    def predict(self, source: Any, **kwargs: Any) -> Sequence[Any]: ...


def _to_numpy(values: Any) -> npt.NDArray[np.float64]:
    """Convert a torch tensor (any device) or array-like to a float numpy array."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float64)


def _load_ultralytics(weights: str | os.PathLike[str]) -> YoloModel:
    try:
        ultralytics = importlib.import_module("ultralytics")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    model: YoloModel = ultralytics.YOLO(os.fspath(weights))
    return model


class YoloDetector:
    """Ultralytics YOLO wrapper implementing :class:`puffer.types.Detector`.

    Parameters
    ----------
    weights:
        Path to ``.pt`` / ``.onnx`` weights (loads ``ultralytics`` lazily) or a
        ready model object exposing ``predict``.
    conf, iou:
        Confidence threshold and NMS IoU passed to ``predict``. Keep ``conf``
        low (default 0.25): ByteTrack wants the low-score boxes too.
    device:
        Torch device string (``"cpu"``, ``"cuda:0"``, ...), ``None`` = auto.
    class_names:
        Optional override of the model's class names (list indexed by class
        id, or mapping). With per-individual training these are fish names.
    """

    def __init__(
        self,
        weights: str | os.PathLike[str] | YoloModel,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str | None = None,
        class_names: Sequence[str] | Mapping[int, str] | None = None,
    ) -> None:
        if isinstance(weights, (str, os.PathLike)):
            self.model: YoloModel = _load_ultralytics(weights)
        elif isinstance(weights, YoloModel):
            self.model = weights
        else:
            raise TypeError("weights must be a path or an object with a predict() method")
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        if class_names is None:
            self._names: dict[int, str] = {}
        elif isinstance(class_names, Mapping):
            self._names = {int(k): str(v) for k, v in class_names.items()}
        else:
            self._names = dict(enumerate(class_names))

    def label_for(self, class_id: int, result_names: Any = None) -> str:
        """Resolve a class id: explicit ``class_names`` > model names > ``"fish"``."""
        if class_id in self._names:
            return self._names[class_id]
        for names in (result_names, getattr(self.model, "names", None)):
            if isinstance(names, Mapping) and class_id in names:
                return str(names[class_id])
            if (
                isinstance(names, Sequence)
                and not isinstance(names, str)
                and 0 <= class_id < len(names)
            ):
                return str(names[class_id])
        return "fish"

    def detect(self, frame: Frame) -> list[Detection]:
        """Run the model on the BGR frame and convert boxes to detections.

        Ultralytics interprets numpy input as BGR (OpenCV order), which is
        exactly :attr:`Frame.image`. Boxes are clipped to the frame and
        degenerate boxes are dropped so downstream Kalman maths never sees
        a zero height.
        """
        kwargs: dict[str, Any] = {"conf": self.conf, "iou": self.iou, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(frame.image, **kwargs)
        width, height = float(frame.width), float(frame.height)
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
            scores = _to_numpy(boxes.conf).reshape(-1)
            classes = _to_numpy(boxes.cls).reshape(-1).astype(np.int64)
            names = getattr(result, "names", None)
            for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes, strict=True):
                bx1, bx2 = float(np.clip(x1, 0.0, width)), float(np.clip(x2, 0.0, width))
                by1, by2 = float(np.clip(y1, 0.0, height)), float(np.clip(y2, 0.0, height))
                if bx2 - bx1 < 1.0 or by2 - by1 < 1.0:
                    continue
                class_id = int(cls)
                detections.append(
                    Detection(
                        (bx1, by1, bx2, by2),
                        float(np.clip(score, 0.0, 1.0)),
                        class_id,
                        self.label_for(class_id, names),
                    )
                )
        return detections
