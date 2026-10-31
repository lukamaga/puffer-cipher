#!/usr/bin/env python3
"""Measure fish-tracking quality on the seeded aquarium simulator; print Markdown.

The numbers go into the README, so the script is deliberately boring:
every scenario is a fixed simulator configuration, every run a fixed seed,
and the same command reproduces the same table (fps aside, which depends
on the machine). Ratios are *pooled* over seeds (counts summed, then
divided), and the worst seed is shown next to the pooled contamination and
recall so a good average cannot hide a bad run.

    python scripts/eval_tracking.py                  # default: 8 seeds x 30 s per scenario
    python scripts/eval_tracking.py --seeds 1-3 --jobs 4
    python scripts/eval_tracking.py --scenarios day,day-clean --json out.json

Scenarios (640x480, 6 fish, 30 fps):

* ``day`` - the default tank: swaying plants, caustics on the sand, a
  bubble column; ``day-clean`` - the same tank with still plants, no
  caustics and no bubbles (what the detector can do without clutter);
* ``day-no-classifier`` - ablation: every confirmed track is credited as a
  fish (``TrackClassifier.permissive``), i.e. without track classification;
* ``night`` / ``night-clean`` - IR camera, sleeping fish (no colour, so no
  appearance re-identification);
* ``cycle`` - a compressed day/night cycle (30 s day incl. 8 s dusk, 20 s
  night) covering both transitions.

All data is synthetic (DESIGN.md rule 4): the simulator's fish, plants and
light are models, so these numbers say how the tracker copes with *these*
models of clutter - a real tank needs its own measurement
(``puffer.vision.dataset`` + manual labels, see docs/yolo.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puffer.sources import SyntheticAquarium, SyntheticConfig  # noqa: E402
from puffer.vision import FishTracker  # noqa: E402
from puffer.vision.classify import TrackClassifier  # noqa: E402
from puffer.vision.evaluate import TrackingMetrics, evaluate_tracking  # noqa: E402

CLEAN = {"plant_sway": 0.0, "caustics": False, "bubbles": False}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    config: dict[str, Any] = field(default_factory=dict)
    frames: int | None = None  # None: --frames
    permissive: bool = False


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("day", "day, plants + caustics + bubbles", {"mode": "day"}),
        Scenario("day-clean", "day, no clutter", {"mode": "day", **CLEAN}),
        Scenario(
            "day-no-classifier",
            "day, clutter, track classification off (ablation)",
            {"mode": "day"},
            permissive=True,
        ),
        Scenario("night", "night (IR), plants + bubbles", {"mode": "night"}),
        Scenario("night-clean", "night (IR), no clutter", {"mode": "night", **CLEAN}),
        Scenario(
            "cycle",
            "day/night cycle (30 s day, 20 s night)",
            {"mode": "cycle", "cycle_day_s": 30.0, "cycle_night_s": 20.0},
            frames=1800,
        ),
    )
}
DEFAULT_SCENARIOS = ("day", "day-clean", "day-no-classifier", "night", "night-clean", "cycle")


def parse_seeds(text: str) -> list[int]:
    """``"1-5"``, ``"1,3,7"`` or a mix like ``"1-3,9"``."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("no seeds given")
    return seeds


def run_one(job: tuple[str, int, int, int]) -> tuple[str, int, TrackingMetrics]:
    """One scenario x seed (runs in a worker process)."""
    name, seed, frames, warmup = job
    scenario = SCENARIOS[name]
    source = SyntheticAquarium(SyntheticConfig(seed=seed, **scenario.config))
    tracker = (
        FishTracker(classifier=TrackClassifier.permissive())
        if scenario.permissive
        else FishTracker()
    )
    return name, seed, evaluate_tracking(source, tracker, frames, warmup=warmup)


def throughput(seed: int, frames: int = 300, repeats: int = 3) -> float:
    """Median frames/s of ``FishTracker.process`` on the default 640x480 tank.

    Frames are rendered first, so the simulator's own cost is excluded; the
    tracker is warmed up on 150 frames before timing. Single process.
    """
    source = SyntheticAquarium(SyntheticConfig(seed=seed))
    clip = [source.read() for _ in range(150 + frames)]
    rates: list[float] = []
    for _ in range(repeats):
        tracker = FishTracker()
        for frame in clip[:150]:
            if frame is not None:
                tracker.process(frame)
        start = time.perf_counter()
        for frame in clip[150:]:
            if frame is not None:
                tracker.process(frame)
        rates.append(frames / (time.perf_counter() - start))
    return statistics.median(rates)


def fmt(value: float, spec: str = ".3f") -> str:
    return "n/a" if math.isnan(value) else format(value, spec)


def pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100.0 * value:.1f} %"


def table(results: dict[str, list[tuple[int, TrackingMetrics]]]) -> str:
    head = (
        "| scenario | det/frame (fish) | det. precision | det. recall | fish recall "
        "(worst) | observed recall | contamination (worst) | track / fish-ID switches per min | "
        "identities / fish | ID purity | ID coverage | centre error |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, runs in results.items():
        pooled = TrackingMetrics.pool([m for _, m in runs])
        worst_recall = min(m.fish_recall for _, m in runs)
        worst_cont = max(
            (m.contamination for _, m in runs if not math.isnan(m.contamination)),
            default=math.nan,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    SCENARIOS[name].description,
                    f"{fmt(pooled.detections_per_frame, '.1f')} "
                    f"({fmt(pooled.fish_per_frame, '.1f')})",
                    pct(pooled.detection_precision),
                    pct(pooled.detection_recall),
                    f"{pct(pooled.fish_recall)} ({pct(worst_recall)})",
                    pct(pooled.observed_recall),
                    f"{pct(pooled.contamination)} ({pct(worst_cont)})",
                    f"{fmt(pooled.id_switches_per_minute, '.1f')} / "
                    f"{fmt(pooled.fish_id_switches_per_minute, '.1f')}",
                    f"{pooled.identities} / {pooled.gt_fish}",
                    pct(pooled.identity_purity),
                    pct(pooled.identity_coverage),
                    f"{fmt(pooled.mean_center_error_px, '.1f')} px",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def provenance() -> str:
    import cv2
    import numpy as np

    try:
        commit = subprocess.run(  # read-only: which code produced the table
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", "src/puffer/vision"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit += " (vision modified)" if dirty else ""
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return (
        f"code {commit}; Python {platform.python_version()}, OpenCV {cv2.__version__}, "
        f"NumPy {np.__version__}; {platform.machine()}, {os.cpu_count()} CPUs"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1-8"))
    parser.add_argument("--frames", type=int, default=900, help="frames per run (30 fps)")
    parser.add_argument("--warmup", type=int, default=150, help="unscored frames per run")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help=f"comma-separated subset of: {', '.join(SCENARIOS)}",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--no-throughput", action="store_true")
    parser.add_argument("--json", type=Path, help="also write every run's metrics here")
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(unknown)}")
    jobs = [
        (name, seed, SCENARIOS[name].frames or args.frames, args.warmup)
        for name in names
        for seed in args.seeds
    ]
    results: dict[str, list[tuple[int, TrackingMetrics]]] = {name: [] for name in names}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for name, seed, metrics in pool.map(run_one, jobs):
            results[name].append((seed, metrics))
            print(f"  {name} seed {seed}: done", file=sys.stderr, flush=True)

    seeds = ", ".join(str(s) for s in args.seeds)
    print("### Fish tracking on the aquarium simulator\n")
    print(
        f"Synthetic data (seeded simulator, 640x480, 6 fish, 30 fps). Seeds {seeds}; "
        f"{args.frames} frames per run ({args.frames / 30:.0f} s; cycle: 1800 frames), "
        f"the first {args.warmup} not scored. Matching: Hungarian on IoU >= 0.3. "
        "Ratios pooled over seeds, worst seed in parentheses.\n"
    )
    print(table(results))
    print(
        "\n*Fish recall*: ground-truth fish-frames covered by a track classified as fish "
        "(including up to 2/3 s of coasting at the predicted box while a fish is hidden). "
        "*Observed recall*: the subset that produced an entropy observation. "
        "*Contamination*: observations whose box lies mostly (>= 50 %) off every fish. "
        "*ID purity / coverage*: re-identification (`fish_id`) against the simulator's "
        "fish; at night IR frames carry no colour, so no identities are assigned.\n"
    )
    if not args.no_throughput:
        fps = throughput(args.seeds[0])
        print(
            f"Throughput: FishTracker {fps:.0f} frames/s at 640x480 (detection, tracking, "
            "classification, re-ID; frames pre-rendered, one process, median of 3).\n"
        )
    print(f"_{provenance()}_")

    if args.json is not None:
        payload = {
            name: {str(seed): m.to_dict() for seed, m in runs} for name, runs in results.items()
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
