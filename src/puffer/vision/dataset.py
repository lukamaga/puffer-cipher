"""Export fish detection datasets in the Ultralytics YOLO format.

The motion detector needs no training, but a YOLO model trained on *your*
tank is more robust to its particular plants, bubbler and reflections, and
trained with one class per individual fish it also names them
(``YoloDetector`` + ``FishReID(trust_labels=True)``). This module produces
the training data two ways:

* :func:`export_synthetic` - from the aquarium simulator, with **perfect**
  ground-truth boxes and one class per fish (or a single ``fish`` class).
  Useful to test the training pipeline end to end and to pre-train; it is
  not a substitute for real footage (DESIGN.md rule 4).
* :func:`export_pseudo_labels` - from a recorded video (or camera) of the
  real tank, labelled by :class:`~puffer.vision.FishTracker`: only tracks
  classified as fish and *measured* in that frame are labelled, with the
  detector's box (not the Kalman prediction). These are **pseudo-labels**:
  load them into CVAT or Label Studio, correct them (missed fish, a plant
  that slipped through, boxes clipped by occlusion, identities), and only
  then train. Frames without any fish are kept as negative examples, which
  is exactly what teaches a detector that plants are not fish.

Layout (what ``ultralytics`` expects, and what CVAT imports as
"Ultralytics YOLO Detection")::

    root/
      dataset.yaml          path, train/val image dirs, class names
      classes.txt           one class name per line (class id = line number)
      provenance.json       how the labels were made (simulator / pseudo-labels)
      images/train/*.jpg    images/val/*.jpg
      labels/train/*.txt    labels/val/*.txt   "class cx cy w h", normalised to [0, 1]

**Train/validation split by time blocks.** Consecutive video frames are
near-duplicates; a per-frame random split would put almost the same image
in both sets and report a validation score the model has not earned. Images
are therefore grouped into blocks of ``block`` consecutive exported frames
and every ``1 / val_fraction``-th block (offset by ``seed``) goes to
validation, so validation footage is interleaved over the whole recording
but never adjacent to a training frame of the same block.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import numpy.typing as npt

from puffer.types import BBox, Frame, FrameSource
from puffer.vision.pipeline import FishTracker

if TYPE_CHECKING:
    from puffer.sources.synthetic import SyntheticConfig

__all__ = [
    "GENERIC_CLASS",
    "ExportSummary",
    "LabelledBox",
    "YoloDatasetWriter",
    "export_pseudo_labels",
    "export_synthetic",
    "read_yolo_labels",
    "yolo_line",
]

GENERIC_CLASS = "fish"
"""Class name of a single-class (any fish) dataset; also ``FishReID``'s generic label."""

Split = Literal["train", "val"]
SPLITS: tuple[Split, ...] = ("train", "val")


@dataclass(frozen=True, slots=True)
class LabelledBox:
    """One object to label: a pixel box and its class name."""

    bbox: BBox
    label: str = GENERIC_CLASS


@dataclass(frozen=True, slots=True)
class ExportSummary:
    """What an export wrote."""

    root: Path
    yaml_path: Path
    classes: tuple[str, ...]
    images: Mapping[str, int]  # split -> number of images
    boxes: Mapping[str, int]  # split -> number of labelled boxes

    @property
    def total_images(self) -> int:
        return sum(self.images.values())

    @property
    def total_boxes(self) -> int:
        return sum(self.boxes.values())


def yolo_line(
    class_id: int, bbox: BBox, width: int, height: int, min_size: float = 2.0
) -> str | None:
    """``"class cx cy w h"`` with coordinates normalised to the image, or ``None``.

    The box is clipped to the image first; a box that is (or becomes) smaller
    than ``min_size`` pixels on a side - a fish leaving the frame - is not
    worth a label and returns ``None``.
    """
    x1 = min(max(bbox[0], 0.0), float(width))
    y1 = min(max(bbox[1], 0.0), float(height))
    x2 = min(max(bbox[2], 0.0), float(width))
    y2 = min(max(bbox[3], 0.0), float(height))
    w, h = x2 - x1, y2 - y1
    if w < min_size or h < min_size:
        return None
    cx, cy = (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def _yaml_string(text: str) -> str:
    """A YAML scalar for ``text``: JSON strings are valid YAML and quote everything."""
    return json.dumps(text, ensure_ascii=False)


class YoloDatasetWriter:
    """Write images and YOLO label files; :meth:`close` writes ``dataset.yaml``.

    Parameters
    ----------
    root:
        Output directory (created). Existing files with the same names are
        overwritten; other files are left alone.
    classes:
        Class names in class-id order. ``None`` grows the list as new labels
        appear (in order of first appearance).
    val_fraction, block, seed:
        Time-block split (module docstring). ``val_fraction=0`` puts
        everything in ``train``.
    image_ext, jpeg_quality:
        ``".jpg"`` (default; what cameras deliver anyway) or ``".png"``.
    provenance:
        Free-form facts recorded in ``provenance.json`` (source, labeller,
        seed...). A dataset should say where its labels came from.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        classes: Sequence[str] | None = None,
        *,
        val_fraction: float = 0.2,
        block: int = 10,
        seed: int = 0,
        image_ext: str = ".jpg",
        jpeg_quality: int = 95,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if block < 1:
            raise ValueError("block must be >= 1")
        if image_ext not in (".jpg", ".png"):
            raise ValueError("image_ext must be '.jpg' or '.png'")
        if classes is not None and len(set(classes)) != len(classes):
            raise ValueError("class names must be unique")
        self.root = Path(root)
        self.val_fraction = float(val_fraction)
        self.block = int(block)
        self.seed = int(seed)
        self.image_ext = image_ext
        self.jpeg_quality = int(jpeg_quality)
        self.provenance = dict(provenance or {})
        self._fixed = classes is not None
        self._classes: list[str] = list(classes) if classes is not None else []
        self._count = 0
        self._images = dict.fromkeys(SPLITS, 0)
        self._boxes = dict.fromkeys(SPLITS, 0)
        self._closed = False
        for kind in ("images", "labels"):
            for split in SPLITS:
                (self.root / kind / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ split

    def split_for(self, index: int) -> Split:
        """Split of the ``index``-th exported image (blocks of consecutive images)."""
        if self.val_fraction <= 0.0:
            return "train"
        period = max(2, round(1.0 / self.val_fraction))
        block = index // self.block
        return "val" if (block + self.seed) % period == period - 1 else "train"

    def class_id(self, label: str) -> int:
        """Class id of ``label``; new labels are appended unless the classes are fixed."""
        try:
            return self._classes.index(label)
        except ValueError:
            if self._fixed:
                raise ValueError(f"unknown class {label!r}; known: {self._classes}") from None
            self._classes.append(label)
            return len(self._classes) - 1

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self._classes)

    # ------------------------------------------------------------------ write

    def add(
        self, image: npt.NDArray[np.uint8], boxes: Iterable[LabelledBox], name: str | None = None
    ) -> Path:
        """Write one BGR image and its labels; return the image path.

        No boxes makes a negative example (background only), which is how a
        detector learns that plants and bubbles are not fish.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        if image.ndim not in (2, 3):
            raise ValueError("image must be HxW or HxWxC")
        height, width = image.shape[:2]
        split = self.split_for(self._count)
        stem = name if name is not None else f"{self._count:06d}"
        lines = [
            line
            for box in boxes
            if (line := yolo_line(self.class_id(box.label), box.bbox, width, height)) is not None
        ]
        image_path = self.root / "images" / split / f"{stem}{self.image_ext}"
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality] if self.image_ext == ".jpg" else []
        if not cv2.imwrite(str(image_path), image, params):
            raise OSError(f"could not write {image_path}")
        label_path = self.root / "labels" / split / f"{stem}.txt"
        label_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        self._count += 1
        self._images[split] += 1
        self._boxes[split] += len(lines)
        return image_path

    def close(self) -> ExportSummary:
        """Write ``dataset.yaml``, ``classes.txt`` and ``provenance.json``; return a summary."""
        if not self._classes:
            self._classes.append(GENERIC_CLASS)  # a dataset of negatives still needs a class
        root = self.root.resolve()
        names = "\n".join(f"  {i}: {_yaml_string(n)}" for i, n in enumerate(self._classes))
        yaml_path = self.root / "dataset.yaml"
        yaml_path.write_text(
            "# Ultralytics YOLO dataset written by puffer.vision.dataset\n"
            f"path: {_yaml_string(str(root))}\n"
            "train: images/train\n"
            "val: images/val\n"
            f"names:\n{names}\n",
            encoding="utf-8",
        )
        (self.root / "classes.txt").write_text(
            "".join(f"{n}\n" for n in self._classes), encoding="utf-8"
        )
        provenance = {
            **self.provenance,
            "classes": self._classes,
            "images": self._images,
            "boxes": self._boxes,
            "split": {"val_fraction": self.val_fraction, "block": self.block, "seed": self.seed},
        }
        (self.root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._closed = True
        return ExportSummary(
            root=self.root,
            yaml_path=yaml_path,
            classes=tuple(self._classes),
            images={str(k): v for k, v in self._images.items()},
            boxes={str(k): v for k, v in self._boxes.items()},
        )

    def __enter__(self) -> YoloDatasetWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._closed and exc_type is None:
            self.close()


# ---------------------------------------------------------------------------
# exporters
# ---------------------------------------------------------------------------


def _check_sampling(n_frames: int, every: int, skip: int) -> None:
    if n_frames < 1 or every < 1 or skip < 0:
        raise ValueError("require n_frames >= 1, every >= 1 and skip >= 0")


def export_synthetic(
    root: str | os.PathLike[str],
    config: SyntheticConfig | None = None,
    n_frames: int = 3000,
    *,
    every: int = 5,
    skip: int = 0,
    per_fish: bool = True,
    val_fraction: float = 0.2,
    block: int = 10,
    image_ext: str = ".jpg",
) -> ExportSummary:
    """Render the simulator and export every ``every``-th frame with perfect labels.

    ``config`` is a :class:`puffer.sources.SyntheticConfig` (default: the
    default tank with a fresh random seed - pass ``seed=`` for a
    reproducible dataset). With ``per_fish`` every fish is its own class,
    named like the fish (classes are fixed in cast order, so class ids are
    stable across exports of the same cast); otherwise one ``fish`` class.

    The simulator's boxes are *amodal*: a fish hidden behind a front plant
    or another fish is still labelled where it is. Real annotations usually
    label only what is visible; mind the difference when mixing the two.
    """
    from puffer.sources.synthetic import SyntheticAquarium, SyntheticConfig

    _check_sampling(n_frames, every, skip)
    cfg = config if config is not None else SyntheticConfig()
    source = SyntheticAquarium(cfg)
    names = {spec.fish_id: spec.name for spec in source.fish_specs}
    classes = list(names.values()) if per_fish else [GENERIC_CLASS]
    provenance = {
        "labels": "simulator ground truth (exact, amodal)",
        "source": "puffer.sources.SyntheticAquarium",
        "seed": source.seed,
        "mode": cfg.mode,
        "size": [cfg.width, cfg.height],
        "frames": n_frames,
        "every": every,
        "per_fish": per_fish,
    }
    writer = YoloDatasetWriter(
        root,
        classes,
        val_fraction=val_fraction,
        block=block,
        seed=source.seed % 1_000_003,
        image_ext=image_ext,
        provenance=provenance,
    )
    try:
        for i in range(n_frames):
            frame = source.read()
            if frame is None:  # pragma: no cover - the simulator never ends
                break
            if i < skip or (i - skip) % every:
                continue
            boxes = [
                LabelledBox(truth.bbox, names[truth.fish_id] if per_fish else GENERIC_CLASS)
                for truth in source.ground_truth()
            ]
            writer.add(frame.image, boxes, name=f"sim{source.seed % 100_000:05d}_{frame.index:06d}")
    finally:
        source.close()
    return writer.close()


def export_pseudo_labels(
    source: FrameSource,
    root: str | os.PathLike[str],
    n_frames: int | None = None,
    *,
    tracker: FishTracker | None = None,
    every: int = 10,
    warmup: int = 150,
    per_fish: bool = False,
    val_fraction: float = 0.2,
    block: int = 10,
    image_ext: str = ".jpg",
    source_name: str | None = None,
) -> ExportSummary:
    """Track ``source`` with :class:`FishTracker` and export pseudo-labelled frames.

    Every frame is tracked (tracking needs consecutive frames); every
    ``every``-th frame after ``warmup`` is exported with one box per track
    that is classified as fish *and* was measured in that frame (the
    detection box, not a prediction). Candidates, coasting tracks and
    clutter are not labelled; frames with no fish are exported as negative
    examples. With ``per_fish`` the label is the re-identified fish's name
    (``FishReID``), and fish not (yet) identified get the generic ``fish``
    label - fix them in the annotation tool.

    These labels are a starting point for manual correction, never ground
    truth: anything the tracker misses or mistakes is in them.
    """
    if n_frames is not None:
        _check_sampling(n_frames, every, warmup)
    elif every < 1 or warmup < 0:
        raise ValueError("require every >= 1 and warmup >= 0")
    ft = tracker if tracker is not None else FishTracker()
    provenance = {
        "labels": "pseudo-labels from puffer FishTracker - review and correct before training",
        "source": source_name if source_name is not None else getattr(source, "source_id", "?"),
        "every": every,
        "warmup": warmup,
        "per_fish": per_fish,
    }
    writer = YoloDatasetWriter(
        root,
        None if per_fish else [GENERIC_CLASS],
        val_fraction=val_fraction,
        block=block,
        image_ext=image_ext,
        provenance=provenance,
    )
    index = 0
    while n_frames is None or index < n_frames:
        frame = source.read()
        if frame is None:
            break
        boxes = _pseudo_boxes(ft, frame, per_fish)
        if index >= warmup and (index - warmup) % every == 0:
            writer.add(frame.image, boxes, name=f"frame_{frame.index:07d}")
        index += 1
    summary = writer.close()
    if summary.total_images == 0:
        raise ValueError(f"no frame exported: the source ended within the {warmup}-frame warm-up")
    return summary


def _pseudo_boxes(ft: FishTracker, frame: Frame, per_fish: bool) -> list[LabelledBox]:
    result = ft.process(frame)
    measured = {o.track_id for o in result.observations}
    by_id = {tr.track_id: tr for tr in ft.tracker.tracked_tracks}
    identities = ft.identities if per_fish else {}
    boxes: list[LabelledBox] = []
    for snap in result.tracks:
        track = by_id.get(snap.track_id)
        if not snap.confirmed or snap.track_id not in measured or track is None:
            continue
        identity = identities.get(snap.fish_id) if snap.fish_id is not None else None
        label = identity.name if identity is not None else GENERIC_CLASS
        boxes.append(LabelledBox(track.detection.bbox, label))
    return boxes


def read_yolo_labels(path: str | os.PathLike[str]) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into ``(class, cx, cy, w, h)`` tuples (for checks and tests)."""
    out: list[tuple[int, float, float, float, float]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}: malformed label line {raw!r}")
        cls, *coords = parts
        cx, cy, w, h = (float(v) for v in coords)
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            raise ValueError(f"{path}: coordinates outside [0, 1] in {raw!r}")
        out.append((int(cls), cx, cy, w, h))
    return out
