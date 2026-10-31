"""YoloDetector against fake Ultralytics models (no PyTorch needed)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from puffer.types import Frame
from puffer.vision.yolo import YoloDetector


class FakeTensor:
    """Mimics a CUDA torch tensor: must go through detach().cpu().numpy()."""

    def __init__(self, values: list[Any]) -> None:
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


@dataclass
class FakeBoxes:
    xyxy: Any
    conf: Any
    cls: Any


@dataclass
class FakeResult:
    boxes: FakeBoxes | None
    names: dict[int, str] | None = None


@dataclass
class FakeModel:
    results: list[FakeResult]
    names: dict[int, str] = field(default_factory=lambda: {0: "fish"})
    calls: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)

    def predict(self, source: Any, **kwargs: Any) -> list[FakeResult]:
        self.calls.append((source, kwargs))
        return self.results


def frame(width: int = 320, height: int = 240) -> Frame:
    return Frame(0, 0.0, 0.0, np.zeros((height, width, 3), dtype=np.uint8))


def test_maps_boxes_to_detections() -> None:
    boxes = FakeBoxes(
        xyxy=FakeTensor([[10, 20, 50, 40], [100, 100, 130, 120]]),
        conf=FakeTensor([0.9, 0.3]),
        cls=FakeTensor([0, 0]),
    )
    model = FakeModel([FakeResult(boxes)])
    detector = YoloDetector(model, conf=0.2, iou=0.4, device="cpu")
    f = frame()
    dets = detector.detect(f)
    assert [d.bbox for d in dets] == [(10.0, 20.0, 50.0, 40.0), (100.0, 100.0, 130.0, 120.0)]
    assert [d.score for d in dets] == pytest.approx([0.9, 0.3])
    assert {d.label for d in dets} == {"fish"}
    source, kwargs = model.calls[0]
    assert source is f.image
    assert kwargs == {"conf": 0.2, "iou": 0.4, "verbose": False, "device": "cpu"}


def test_device_omitted_when_auto() -> None:
    model = FakeModel([])
    YoloDetector(model).detect(frame())
    assert "device" not in model.calls[0][1]


def test_per_individual_classes_carry_fish_names() -> None:
    boxes = FakeBoxes(np.array([[0, 0, 20, 10], [50, 50, 80, 60]]), [0.8, 0.7], [1, 0])
    names = {0: "Nemo", 1: "Dory"}
    dets = YoloDetector(FakeModel([FakeResult(boxes, names=names)], names=names)).detect(frame())
    assert [(d.class_id, d.label) for d in dets] == [(1, "Dory"), (0, "Nemo")]


def test_class_name_override_and_fallbacks() -> None:
    boxes = FakeBoxes(np.array([[0, 0, 20, 10]] * 3), [0.8] * 3, [0, 1, 7])
    model = FakeModel([FakeResult(boxes)], names={0: "a", 1: "b"})
    assert [d.label for d in YoloDetector(model).detect(frame())] == ["a", "b", "fish"]
    overridden = YoloDetector(model, class_names=["Bubbles", "Gill"]).detect(frame())
    assert [d.label for d in overridden] == ["Bubbles", "Gill", "fish"]
    mapped = YoloDetector(model, class_names={7: "Wanda"}).detect(frame())
    assert [d.label for d in mapped] == ["a", "b", "Wanda"]


def test_list_style_model_names() -> None:
    detector = YoloDetector(FakeModel([], names={}))
    assert detector.label_for(1, ["x", "y"]) == "y"
    assert detector.label_for(5, ["x", "y"]) == "fish"


def test_boxes_clipped_and_degenerate_dropped() -> None:
    boxes = FakeBoxes(
        np.array([[-10, -5, 30, 20], [300, 200, 400, 300], [319.5, 10, 330, 20], [5, 5, 5, 9]]),
        [1.2, 0.5, 0.9, 0.9],
        [0, 0, 0, 0],
    )
    dets = YoloDetector(FakeModel([FakeResult(boxes)])).detect(frame())
    assert [d.bbox for d in dets] == [(0.0, 0.0, 30.0, 20.0), (300.0, 200.0, 320.0, 240.0)]
    assert dets[0].score == 1.0


def test_results_without_boxes() -> None:
    assert YoloDetector(FakeModel([FakeResult(None)])).detect(frame()) == []


def test_missing_ultralytics_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # makes the import fail
    with pytest.raises(ImportError, match=r"pip install 'puffer-cipher\[yolo\]'"):
        YoloDetector("fish.pt")


def test_weights_path_loads_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []

    class FakeYOLO(FakeModel):
        def __init__(self, path: str) -> None:
            super().__init__([])
            loaded.append(path)

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    detector = YoloDetector("weights/tank.pt")
    assert loaded == ["weights/tank.pt"]
    assert isinstance(detector.model, FakeYOLO)


def test_rejects_non_model() -> None:
    with pytest.raises(TypeError, match="predict"):
        YoloDetector(42)  # type: ignore[arg-type]
