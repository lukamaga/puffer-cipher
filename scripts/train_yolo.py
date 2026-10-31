#!/usr/bin/env python3
"""Build a fish dataset and fine-tune an Ultralytics YOLO detector on it.

Three steps, one subcommand each (see docs/yolo.md for the whole workflow):

    # 1a. perfect labels from the simulator (one class per fish), or ...
    python scripts/train_yolo.py simulate --out data/yolo-sim --frames 6000 --seed 7

    # 1b. ... pseudo-labels from a recording of YOUR tank, to correct in CVAT/Label Studio
    python scripts/train_yolo.py pseudo-label --source tank.mp4 --out data/yolo-tank

    # 2. fine-tune (needs: pip install 'puffer-cipher[yolo]')
    python scripts/train_yolo.py train --data data/yolo-tank/dataset.yaml --epochs 80

    # 3. run with it:  PUFFER_DETECTOR=yolo:runs/puffer/fish/weights/best.pt puffer serve

``ultralytics`` (and PyTorch) is imported only by ``train``, so the export
steps run in the lightweight base install.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INSTALL_HINT = (
    "training needs the optional 'ultralytics' package (it pulls in PyTorch):\n"
    "    pip install 'puffer-cipher[yolo]'\n"
    "or train on a machine with a GPU and copy best.pt back."
)


class MissingDependencyError(RuntimeError):
    """An optional dependency needed by this command is not installed."""


def load_ultralytics() -> Any:
    """Import ``ultralytics`` lazily with an actionable error."""
    try:
        return importlib.import_module("ultralytics")
    except ImportError as exc:
        raise MissingDependencyError(INSTALL_HINT) from exc


def cmd_simulate(args: argparse.Namespace) -> int:
    from puffer.sources import SyntheticConfig
    from puffer.vision.dataset import export_synthetic

    config = SyntheticConfig(seed=args.seed, mode=args.mode, n_fish=args.fish)
    summary = export_synthetic(
        args.out,
        config,
        args.frames,
        every=args.every,
        per_fish=not args.generic,
        val_fraction=args.val,
    )
    print(
        f"wrote {summary.total_images} images ({summary.images['train']} train, "
        f"{summary.images['val']} val), {summary.total_boxes} boxes, "
        f"{len(summary.classes)} classes -> {summary.yaml_path}"
    )
    return 0


def cmd_pseudo_label(args: argparse.Namespace) -> int:
    from puffer.sources import open_source
    from puffer.vision.dataset import export_pseudo_labels

    source = open_source(args.source)
    try:
        summary = export_pseudo_labels(
            source,
            args.out,
            args.frames,
            every=args.every,
            warmup=args.warmup,
            per_fish=args.per_fish,
            val_fraction=args.val,
            source_name=args.source,
        )
    finally:
        source.close()
    print(
        f"wrote {summary.total_images} images, {summary.total_boxes} pseudo-labelled boxes "
        f"-> {summary.root}\n"
        "These are machine labels: import the folder into CVAT ('Ultralytics YOLO Detection') "
        "or Label Studio, correct them, export again in the same format, then train."
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    data = Path(args.data)
    if not data.is_file():
        raise FileNotFoundError(f"{data}: no such dataset.yaml (run 'simulate' or 'pseudo-label')")
    ultralytics = load_ultralytics()
    model = ultralytics.YOLO(args.model)
    kwargs: dict[str, Any] = {
        "data": str(data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": True,
        # Aquarium footage: colour identifies individual fish, so keep hue
        # augmentation mild; flips are fine (fish swim both ways).
        "hsv_h": 0.005,
        "fliplr": 0.5,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    results = model.train(**kwargs)
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    print(
        f"best weights: {best}\n"
        f"run with:     PUFFER_DETECTOR=yolo:{best} puffer serve\n"
        "per-fish classes name the fish (FishReID(trust_labels=True) is enabled for yolo:)."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="export simulator frames with perfect labels")
    sim.add_argument("--out", type=Path, required=True)
    sim.add_argument("--frames", type=int, default=6000, help="frames to render (30 fps)")
    sim.add_argument("--every", type=int, default=5, help="export every n-th frame")
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--mode", choices=("day", "night", "cycle"), default="day")
    sim.add_argument("--fish", type=int, default=6)
    sim.add_argument(
        "--generic", action="store_true", help="one 'fish' class instead of one per fish"
    )
    sim.add_argument("--val", type=float, default=0.2, help="validation fraction")
    sim.set_defaults(func=cmd_simulate)

    pseudo = sub.add_parser(
        "pseudo-label", help="export a recording with FishTracker pseudo-labels"
    )
    pseudo.add_argument("--source", required=True, help="source spec, e.g. tank.mp4 or camera:0")
    pseudo.add_argument("--out", type=Path, required=True)
    pseudo.add_argument("--frames", type=int, default=None, help="frames to read (default: all)")
    pseudo.add_argument("--every", type=int, default=10)
    pseudo.add_argument("--warmup", type=int, default=150, help="frames before the first export")
    pseudo.add_argument("--per-fish", action="store_true", help="label boxes with re-ID names")
    pseudo.add_argument("--val", type=float, default=0.2)
    pseudo.set_defaults(func=cmd_pseudo_label)

    train = sub.add_parser("train", help="fine-tune Ultralytics YOLO on a dataset.yaml")
    train.add_argument("--data", required=True, help="dataset.yaml from simulate/pseudo-label")
    train.add_argument("--model", default="yolo11n.pt", help="pretrained weights to start from")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--device", default=None, help="'cpu', '0', 'mps', ... (default: auto)")
    train.add_argument("--project", default="runs/puffer")
    train.add_argument("--name", default="fish")
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=cmd_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result: int = args.func(args)
    except MissingDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
