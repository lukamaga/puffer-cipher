"""YOLO dataset export (simulator ground truth, tracker pseudo-labels) and the training script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from puffer.sources import SyntheticConfig
from puffer.types import Frame
from puffer.vision.dataset import (
    GENERIC_CLASS,
    LabelledBox,
    YoloDatasetWriter,
    export_pseudo_labels,
    export_synthetic,
    read_yolo_labels,
    yolo_line,
)

from .scenes import BLUE, ORANGE, FishSpec, Scene, linear, wander

ROOT = Path(__file__).resolve().parents[2]


def label_files(root: Path, split: str) -> list[Path]:
    return sorted((root / "labels" / split).glob("*.txt"))


def test_yolo_line_normalises_and_clips() -> None:
    assert (
        yolo_line(2, (10.0, 20.0, 30.0, 60.0), 100, 200) == "2 0.200000 0.200000 0.200000 0.200000"
    )
    # clipped at the right edge: (90..100) x (0..10)
    assert (
        yolo_line(0, (90.0, -5.0, 120.0, 10.0), 100, 100) == "0 0.950000 0.050000 0.100000 0.100000"
    )
    assert yolo_line(0, (99.5, 10.0, 130.0, 20.0), 100, 100) is None  # a sliver left in view


def test_writer_layout_split_and_yaml(tmp_path: Path) -> None:
    image = np.full((48, 64, 3), 90, np.uint8)
    with YoloDatasetWriter(
        tmp_path, val_fraction=0.25, block=2, provenance={"labels": "test"}
    ) as w:
        for i in range(16):
            boxes = [LabelledBox((4.0, 4.0, 20.0, 12.0), "Nemo")] if i % 2 else []
            w.add(image, boxes)
        w.add(image, [LabelledBox((30.0, 20.0, 40.0, 30.0), "Dory")])
    # blocks of 2 images, every 4th block to val: images 6, 7, 14, 15
    assert sorted(p.stem for p in label_files(tmp_path, "val")) == [
        "000006",
        "000007",
        "000014",
        "000015",
    ]
    assert len(label_files(tmp_path, "train")) == 13
    assert len(list((tmp_path / "images" / "train").glob("*.jpg"))) == 13
    assert (tmp_path / "labels" / "train" / "000000.txt").read_text() == ""  # negative example
    assert read_yolo_labels(tmp_path / "labels" / "train" / "000001.txt") == [
        (0, 0.1875, 0.166667, 0.25, 0.166667)
    ]
    assert read_yolo_labels(tmp_path / "labels" / "train" / "000016.txt")[0][0] == 1
    yaml = (tmp_path / "dataset.yaml").read_text()
    assert f'path: "{tmp_path.resolve()}"' in yaml
    assert "train: images/train\nval: images/val\n" in yaml
    assert '  0: "Nemo"\n  1: "Dory"' in yaml
    assert (tmp_path / "classes.txt").read_text() == "Nemo\nDory\n"
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["labels"] == "test"
    assert provenance["images"] == {"train": 13, "val": 4}


def test_writer_with_fixed_classes_rejects_unknown_labels(tmp_path: Path) -> None:
    writer = YoloDatasetWriter(tmp_path, ["fish"])
    with pytest.raises(ValueError, match="unknown class"):
        writer.add(np.zeros((8, 8, 3), np.uint8), [LabelledBox((0, 0, 4, 4), "plant")])
    summary = writer.close()
    assert summary.classes == ("fish",)
    with pytest.raises(RuntimeError, match="closed"):
        writer.add(np.zeros((8, 8, 3), np.uint8), [])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"val_fraction": 1.0}, "val_fraction"),
        ({"block": 0}, "block"),
        ({"image_ext": ".bmp"}, "image_ext"),
        ({"classes": ["a", "a"]}, "unique"),
    ],
)
def test_writer_validation(tmp_path: Path, kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        YoloDatasetWriter(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_export_synthetic_has_perfect_per_fish_labels(tmp_path: Path) -> None:
    config = SyntheticConfig(seed=11, width=320, height=240, n_fish=3)
    summary = export_synthetic(tmp_path, config, 40, every=4, block=2)
    assert summary.classes == ("Biscuit", "Captain Fin", "Professor Puff")
    assert summary.total_images == 10
    assert summary.images["val"] > 0
    assert summary.images["train"] > 0
    assert summary.total_boxes == 30  # three fish in every frame
    for split in ("train", "val"):
        for path in label_files(tmp_path, split):
            labels = read_yolo_labels(path)
            assert sorted(cls for cls, *_ in labels) == [0, 1, 2]
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["labels"].startswith("simulator ground truth")
    assert provenance["seed"] == 11
    # reproducible: same seed, same labels
    again = export_synthetic(tmp_path / "again", config, 40, every=4, block=2)
    assert again.total_boxes == summary.total_boxes
    first = label_files(tmp_path, "train")[0]
    assert first.read_text() == (tmp_path / "again" / "labels" / "train" / first.name).read_text()


def test_export_synthetic_single_class(tmp_path: Path) -> None:
    config = SyntheticConfig(seed=3, width=320, height=240, n_fish=2)
    summary = export_synthetic(tmp_path, config, 10, every=5, per_fish=False)
    assert summary.classes == (GENERIC_CLASS,)
    assert summary.total_boxes == 4


class SceneSource:
    """A FrameSource over the test scenes (no LED, no ground truth needed)."""

    source_id = "test-scene"
    fps = 30.0

    def __init__(self, scene: Scene, n: int) -> None:
        self.scene = scene
        self.n = n

    def read(self) -> Frame | None:
        if self.scene.index >= self.n:
            return None
        frame, _ = self.scene.render()
        return frame

    def close(self) -> None:
        pass


def test_pseudo_labels_follow_the_tracked_fish(tmp_path: Path, rng: np.random.Generator) -> None:
    fish = [
        FishSpec("a", ORANGE, 30, 12, linear((20, 60), (45, 10))),
        FishSpec("b", BLUE, 28, 11, wander(rng, (200, 150), (60, 40))),
    ]
    scene = Scene(fish, seed=4)
    summary = export_pseudo_labels(SceneSource(scene, 150), tmp_path, every=10, warmup=60)
    assert summary.classes == (GENERIC_CLASS,)
    assert summary.total_images == 9  # frames 60, 70, ..., 140
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert "pseudo-labels" in provenance["labels"]
    assert provenance["source"] == "test-scene"
    # by the end both fish are classified and labelled at their true positions
    truth = {f.name: f.path(140 / 30.0) for f in fish}
    last = next(
        p for split in ("train", "val") for p in label_files(tmp_path, split) if "0000140" in p.name
    )
    labels = read_yolo_labels(last)
    assert len(labels) == 2
    for _, cx, cy, _, _ in labels:
        x, y = cx * 320, cy * 240
        assert min(np.hypot(x - tx, y - ty) for tx, ty in truth.values()) < 5.0


def test_pseudo_labels_per_fish_use_identity_names(tmp_path: Path) -> None:
    fish = [FishSpec("a", ORANGE, 30, 12, linear((20, 60), (45, 10)))]
    summary = export_pseudo_labels(
        SceneSource(Scene(fish, seed=5), 150), tmp_path, every=10, warmup=60, per_fish=True
    )
    assert "Nemo" in summary.classes  # the first auto-enrolled name


def test_pseudo_labels_need_frames_after_warmup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="warm-up"):
        export_pseudo_labels(SceneSource(Scene([], seed=6), 20), tmp_path, warmup=50)


def _load_script(name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None, reason="ultralytics installed"
)
def test_train_script_explains_missing_ultralytics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script("train_yolo")
    data = tmp_path / "dataset.yaml"
    data.write_text("names: {0: fish}\n")
    assert script.main(["train", "--data", str(data)]) == 2  # type: ignore[attr-defined]
    err = capsys.readouterr().err
    assert "pip install 'puffer-cipher[yolo]'" in err
    assert "ultralytics" not in sys.modules


def test_train_script_simulate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script("train_yolo")
    out = tmp_path / "sim"
    code = script.main(  # type: ignore[attr-defined]
        ["simulate", "--out", str(out), "--frames", "10", "--every", "5", "--fish", "2"]
    )
    assert code == 0
    assert (out / "dataset.yaml").is_file()
    assert "2 images" in capsys.readouterr().out
