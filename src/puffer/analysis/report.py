"""The statistics report (DESIGN.md section 10): what the tank's samples are worth, measured.

:func:`run_report` drives the *real* engine (:func:`puffer.runtime.build_engine`)
over one or more frame sources, captures every channel output through
:meth:`~puffer.engine.EntropyEngine.add_output_listener`, and turns what it
saw into ``RESULTS.md``, a machine-readable ``results.json`` and the charts
of :mod:`puffer.analysis.plots`.

What is measured, and why in this order
---------------------------------------
1. **Raw min-entropy** (SP 800-90B section 3.1.3, non-IID track) of the
   uint8 samples each channel credits: the only figure in the report that
   says anything about entropy.
2. **Credit, day versus night**: credited bits per second and per sample, as
   the engine computed them, plus a day -> night -> day run of the simulator
   (``mode=cycle``) as a time series.
3. **Conditioned streams without OS randomness**
   (:class:`~puffer.entropy.ConditionedStream`: sensor only, fish only,
   combined) and the final :class:`~puffer.entropy.PufferRNG` output (keyed
   with ``os.urandom``), through SP 800-22 (section 4.2 multi-sequence
   analysis on 10^6-bit sequences where the data allows, the tests that fit
   otherwise) and the ``ent`` equivalent. ``ent`` also runs on the *raw*
   channel bytes, which are expected to fail: hashed output passes these
   tests whatever went in (DESIGN.md rule 4), raw samples show what went in.
4. **Liveness**: the LED anti-replay check on a live simulator run versus a
   lossless (FFV1) recording of that very run replayed against a fresh
   challenge, plus a control replay against the recorded challenge.
5. **Tracking quality** (:func:`puffer.vision.evaluate.evaluate_tracking`)
   by day and by night: every fish observation is credited as entropy.
6. **dieharder** on PufferRNG output fed through a pipe (``-g 200``), on a
   time budget.

Design decisions
----------------
* **Everything goes through the engine.** Samples are what the engine's
  channels produced (LED region excluded, tracker downscaled, health
  states applied), credits are what it credited. Analysis runs use
  ``liveness=off`` and no LED (nothing to verify, nothing blinking), the
  beacon is never built, and each run gets a throw-away ``data_dir``.
* **Reproducible except where production is not.** A seed given in the
  config (or drawn once and printed) is written into every simulator spec
  that has none, and seeds the recorded LED challenge. Two things stay
  random on purpose, as in production: the sensor sample sites (the engine
  draws them from ``os.urandom``) and PufferRNG output (keyed with
  ``os.urandom``), so a re-run reproduces the figures closely, not exactly.
* **Conditioned streams use the assessed entropy, conservatively.** A block
  is compressed once ``n * min(H, H_cap) * safety_factor`` reaches 512 bits,
  with ``H`` the SP 800-90B assessment of that very run and the engine's
  cap and safety factor. Each 256-bit output block is therefore backed by
  four times its size in assessed min-entropy. The combined stream
  interleaves both channels frame by frame at the sample-weighted credit
  rate.
* **Honest sample counts.** SP 800-90B asks for 1,000,000 samples; the
  report states how many each assessment used, and what fraction of the
  standard's requirement that is.
* **Parallel where it is free.** Collection runs are independent engines and
  the heavy statistics are pure functions, so with ``jobs > 1`` they run in
  worker processes (``spawn`` start method: OpenCV and fork do not mix).
  dieharder runs last, next to an engine thread that keeps absorbing frames,
  exactly like ``puffer random --stream | dieharder -g 200``.

Nothing here touches secrets: PufferRNG output is generated for testing and
discarded, conditioned streams are unkeyed by design and analysis only.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import enum
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.stats import binom, poisson

from puffer.analysis.bits import bytes_to_bits, split_sequences
from puffer.analysis.ent import ent_stats
from puffer.analysis.external import (
    ExternalToolError,
    dieharder_available,
    ent_available,
    parse_dieharder_output,
    run_ent,
)
from puffer.analysis.sp800_22 import (
    DEFAULT_ALPHA,
    MIN_SEQUENCES_FOR_UNIFORMITY,
    SUITE_TESTS,
    UNIFORMITY_THRESHOLD,
    MultiSuiteResult,
    SuiteResult,
    run_multi,
    run_suite,
)
from puffer.config import PufferConfig
from puffer.entropy import DEFAULT_H_CAP, ConditionedStream
from puffer.entropy.estimators import BITSTRING_LIMIT, InsufficientDataError, assess_non_iid
from puffer.liveness import ChallengeSchedule
from puffer.runtime import build_engine
from puffer.sources import SyntheticAquarium, SyntheticConfig, VideoFileSource, open_source
from puffer.types import (
    CAMERA_CHANNELS,
    CHANNEL_FISH,
    CHANNEL_SENSOR,
    CHANNEL_SYSTEM,
    ChannelOutput,
    HealthState,
    LightMode,
    LivenessState,
)

__all__ = [
    "DEFAULT_SOURCES",
    "DIEHARDER_SELECTION",
    "REPORT_VERSION",
    "SP800_90B_SAMPLES",
    "ReportConfig",
    "render_markdown",
    "run_report",
    "write_results",
]

REPORT_VERSION: Final = 1
DEFAULT_SOURCES: Final = ("synthetic:seed=1,mode=day", "synthetic:seed=1,mode=night")
SP800_90B_SAMPLES: Final = 1_000_000
"""Samples SP 800-90B section 3.1 asks for before an entropy estimate is made."""

CHUNK_BYTES: Final = 1 << 20
"""PufferRNG output is requested 1 MiB at a time (one pool draw each, like ``random --stream``)."""

BLOCK_ENTROPY_BITS: Final = 512
OUTPUT_BLOCK_BYTES: Final = 32
LIVENESS_THRESHOLD_BITS: Final = 24.0
"""``LivenessVerifier`` default ``min_confidence_bits`` (DESIGN.md section 5)."""
LIVENESS_MIN_MATCH: Final = 0.9

SERIES_PERIOD_S: Final = 0.5
"""Spacing of the time series written to ``results.json`` (the dashboard's 2 Hz)."""

DIEHARDER_SELECTION: Final = (
    *(str(n) for n in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17)),
    "100",
    "101",
    "102",
    *(f"200:{n}" for n in range(1, 13)),
    *(f"201:{n}" for n in range(2, 6)),
    *(f"202:{n}" for n in range(2, 6)),
    "204",
    "205",
    "206",
    "207",
    "208",
    "209",
    *(f"203:{n}" for n in (0, 1, 2, 3, 4, 8, 16, 32)),
)
"""dieharder tests run by default (``-d N`` or ``-d N -n ntuple``), about 13 minutes.

Every test and ntuple of ``dieharder -a`` except two cuts. ``diehard_sums``
(14) is left out because dieharder itself rates it "Do Not Use". The RGB
lagged sum (203) runs lags 0-4, 8, 16 and 32 instead of all 33: lag ``n``
reads ``n + 1`` times the data, so the full sweep alone takes about half an
hour, while these eight cover short, medium and long lags. The lagged sums
run last, so a run that hits its budget loses only the longest lags.
Multi-ntuple tests get an explicit ``-n``: ``-d 200`` alone runs nothing and
``-d 201`` alone runs the ntuple-0 variant, which fails for any input."""

STATES: Final = tuple(state.value for state in HealthState)
_DEFAULT_SYNTHETIC = SyntheticConfig()
_REFERENCE_LED = (608, 32, 640, 480)  # SyntheticConfig's LED position for a 640x480 frame
_SPEC_DROP_FOR_DERIVED = frozenset({"mode", "day", "night", "realtime", "start"})

_ALPHA: Final = DEFAULT_ALPHA
Progress = Callable[[str], None]
_T = TypeVar("_T")
_R = TypeVar("_R")


# ======================================================================== configuration


@dataclass
class ReportConfig:
    """What to measure (DESIGN.md section 10; fields after ``seed`` extend it).

    ``sources`` are source specs; each is collected for ``frames`` frames.
    The derived sections (day/night cycle, liveness, tracking) run on the
    first ``synthetic`` source, re-configured as each section needs, and are
    skipped (with a note) when there is none.

    Extensions, all with defaults that keep the CLI's behaviour:

    * ``nist_bits``: length of one SP 800-22 sequence (10^6 in SP 800-22).
    * ``output_sequences``: sequences of PufferRNG output (at least 55 lets
      SP 800-22 assess p-value uniformity per sub-test).
    * ``max_90b_samples``: samples per channel given to SP 800-90B.
    * ``cycle_day_s``/``cycle_night_s``/``cycle_frames``: the day -> night ->
      day run (``None`` frames: day + night + day; 0 skips it).
    * ``liveness_s``: length of the live, replay and control runs (0 skips).
    * ``tracking_frames``/``tracking_warmup``: tracking evaluation (0 skips).
    * ``dieharder_budget_s``/``dieharder_tests``: wall-clock budget and the
      tests, as ``"N"`` (``-d N``) or ``"N:n"`` (``-d N -n n``) entries;
      ``None`` runs :data:`DIEHARDER_SELECTION`, ``("all",)`` the full
      battery (``-a``, about 45 minutes on a 2020s CPU core).
    * ``jobs``: worker processes (``None``: up to 4, one per CPU).
    * ``plots``: render the charts (needs matplotlib).
    """

    sources: list[str]
    frames: int = 3000
    out_dir: Path = Path("docs/results")
    media_dir: Path = Path("docs/media")
    nist_sequences: int = 10
    dieharder: bool = False
    seed: int | None = None
    nist_bits: int = 1_000_000
    output_sequences: int = 100
    max_90b_samples: int = SP800_90B_SAMPLES
    cycle_day_s: float = 40.0
    cycle_night_s: float = 120.0
    cycle_frames: int | None = None
    liveness_s: float = 40.0
    tracking_frames: int = 1800
    tracking_warmup: int = 150
    dieharder_budget_s: float = 1200.0
    dieharder_tests: tuple[str, ...] | None = None
    jobs: int | None = None
    plots: bool = True

    def __post_init__(self) -> None:
        self.sources = [str(s).strip() for s in self.sources]
        self.out_dir = Path(self.out_dir)
        self.media_dir = Path(self.media_dir)
        if not self.sources or not all(self.sources):
            raise ValueError("at least one non-empty source spec is required")
        _positive("frames", self.frames, minimum=2)
        _positive("nist_sequences", self.nist_sequences)
        _positive("nist_bits", self.nist_bits, minimum=100)
        _positive("output_sequences", self.output_sequences)
        _positive("max_90b_samples", self.max_90b_samples, minimum=2)
        _positive("tracking_frames", self.tracking_frames, minimum=0)
        _positive("tracking_warmup", self.tracking_warmup, minimum=0)
        if self.tracking_frames and self.tracking_frames <= self.tracking_warmup:
            raise ValueError("tracking_frames must exceed tracking_warmup (or be 0)")
        if self.cycle_frames is not None:
            _positive("cycle_frames", self.cycle_frames, minimum=0)
        for name in ("cycle_day_s", "cycle_night_s", "dieharder_budget_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if not (math.isfinite(self.liveness_s) and self.liveness_s >= 0):
            raise ValueError("liveness_s must be >= 0")
        if self.jobs is not None:
            _positive("jobs", self.jobs)
        if self.dieharder_tests is not None:
            self.dieharder_tests = tuple(str(t).strip() for t in self.dieharder_tests)
            if not self.dieharder_tests or not all(map(_valid_selection, self.dieharder_tests)):
                raise ValueError(
                    "dieharder_tests entries are 'N', 'N:n' (test number, ntuple) or 'all'"
                )

    @property
    def workers(self) -> int:
        if self.jobs is not None:
            return self.jobs
        return max(1, min(4, os.cpu_count() or 1))

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["out_dir"] = os.fspath(self.out_dir)
        out["media_dir"] = os.fspath(self.media_dir)
        out["dieharder_tests"] = list(self.dieharder_tests) if self.dieharder_tests else None
        return out


def _valid_selection(entry: str) -> bool:
    if entry == "all":
        return True
    test, _, ntup = entry.partition(":")
    return test.isdigit() and (not ntup or ntup.isdigit())


def _dieharder_args(entry: str) -> list[str]:
    """``"203:8"`` -> ``["-d", "203", "-n", "8"]``; ``"all"`` -> ``["-a"]``."""
    if entry == "all":
        return ["-a"]
    test, _, ntup = entry.partition(":")
    return ["-d", test, *(["-n", ntup] if ntup else [])]


def _positive(name: str, value: object, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


# ======================================================================== source specs


def _synthetic_items(spec: str) -> list[tuple[str, str]] | None:
    """``[(key, value), ...]`` of a ``synthetic`` spec; ``None`` for any other kind."""
    kind, _, rest = spec.strip().partition(":")
    if kind.strip().lower() != "synthetic":
        return None
    items: list[tuple[str, str]] = []
    for part in rest.split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep:  # malformed: let open_source report it with its own message
            return None
        items.append((key.strip().lower(), value.strip()))
    return items


def _join_spec(items: Sequence[tuple[str, str]]) -> str:
    return "synthetic:" + ",".join(f"{k}={v}" for k, v in items) if items else "synthetic"


def _set_options(items: Sequence[tuple[str, str]], **updates: object) -> list[tuple[str, str]]:
    """``items`` with ``updates`` replacing existing keys in place or appended."""
    out = [(k, str(updates[k]) if k in updates else v) for k, v in items]
    present = {k for k, _ in items}
    out += [(k, str(v)) for k, v in updates.items() if k not in present]
    return out


def _option(items: Sequence[tuple[str, str]], key: str) -> str | None:
    for k, v in items:
        if k == key:
            return v
    return None


def _prepare_synthetic(items: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    """Seed, not real-time, and an LED inside a frame that is not 640x480."""
    updates: dict[str, object] = {"realtime": 0}
    if _option(items, "seed") in (None, "random", "none"):
        updates["seed"] = seed
    width, height = _option(items, "width"), _option(items, "height")
    if _option(items, "led") is None and (width or height):
        x, y, ref_w, ref_h = _REFERENCE_LED
        w = int(width) if width else _DEFAULT_SYNTHETIC.width
        h = int(height) if height else _DEFAULT_SYNTHETIC.height
        updates["led"] = f"{round(x * w / ref_w)}:{round(y * h / ref_h)}"
    return _set_options(items, **updates)


def _synthetic_fps(items: Sequence[tuple[str, str]]) -> float:
    fps = _option(items, "fps")
    return float(fps) if fps else float(_DEFAULT_SYNTHETIC.fps)


@dataclass(frozen=True)
class _Plan:
    seed: int
    seed_source: str  # "config" or "drawn"
    runs: list[_RunJob]
    base: list[tuple[str, str]] | None  # first synthetic source, prepared
    cycle: _RunJob | None
    liveness: _LivenessJob | None
    tracking: list[_TrackingJob]
    skipped: dict[str, str]


def _plan(config: ReportConfig) -> _Plan:
    seed_source = "config"
    seed = config.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
        seed_source = "drawn"
    output_bytes = max(config.nist_sequences, config.output_sequences) * config.nist_bits // 8
    runs: list[_RunJob] = []
    base: list[tuple[str, str]] | None = None
    labels: set[str] = set()
    for spec in config.sources:
        items = _synthetic_items(spec)
        if items is not None:
            items = _prepare_synthetic(items, seed)
            if base is None:
                base = items
            spec = _join_spec(items)
            label = _option(items, "mode") or "day"
        else:
            label = spec.partition(":")[2].rsplit("/", 1)[-1] or spec
        unique, n = label, 2
        while unique in labels:
            unique, n = f"{label}-{n}", n + 1
        labels.add(unique)
        runs.append(_RunJob(unique, spec, config.frames))
    runs[0] = dataclasses.replace(runs[0], output_bytes=output_bytes)

    skipped: dict[str, str] = {}
    cycle = liveness = None
    tracking: list[_TrackingJob] = []
    if base is None:
        reason = "needs a synthetic source (the simulator knows the ground truth and the LED)"
        skipped.update(cycle=reason, liveness=reason, tracking=reason)
    else:
        derived = [(k, v) for k, v in base if k not in _SPEC_DROP_FOR_DERIVED]
        fps = _synthetic_fps(base)
        day_s, night_s = config.cycle_day_s, config.cycle_night_s
        n_cycle = config.cycle_frames
        if n_cycle is None:
            n_cycle = round((2 * day_s + night_s) * fps)
        if n_cycle >= 2:
            items = [*derived, ("mode", "cycle"), ("day", f"{day_s:g}"), ("night", f"{night_s:g}")]
            cycle = _RunJob("cycle", _join_spec(items), n_cycle)
        else:
            skipped["cycle"] = "disabled (cycle_frames = 0)"
        n_live = round(config.liveness_s * fps)
        if n_live >= 2:
            day_spec = _join_spec([*derived, ("mode", "day")])
            liveness = _LivenessJob(day_spec, n_live, schedule_seed=seed)
        else:
            skipped["liveness"] = "disabled (liveness_s = 0)"
        if config.tracking_frames:
            for mode in ("day", "night"):
                spec = _join_spec([*derived, ("mode", mode)])
                tracking.append(
                    _TrackingJob(mode, spec, config.tracking_frames, config.tracking_warmup)
                )
        else:
            skipped["tracking"] = "disabled (tracking_frames = 0)"
    return _Plan(seed, seed_source, runs, base, cycle, liveness, tracking, skipped)


# ======================================================================== collection


@dataclass(frozen=True)
class _RunJob:
    label: str
    spec: str
    frames: int
    output_bytes: int = 0
    keep_samples: bool = True


@dataclass(frozen=True)
class _LivenessJob:
    spec: str
    frames: int
    schedule_seed: int
    control: bool = True


@dataclass(frozen=True)
class _TrackingJob:
    label: str
    spec: str
    frames: int
    warmup: int


@dataclass
class RunData:
    """Everything one engine run produced, per frame and per channel."""

    label: str
    spec: str
    source: dict[str, Any]
    fps: float
    wall_s: float
    t: npt.NDArray[np.float64]
    night: npt.NDArray[np.bool_]
    credit: dict[str, npt.NDArray[np.float64]]
    n_samples: dict[str, npt.NDArray[np.int64]]
    state: dict[str, list[str]]
    bits_per_s: dict[str, npt.NDArray[np.float64]]
    h_est: dict[str, npt.NDArray[np.float64]]
    tracks: npt.NDArray[np.int64]
    liveness_state: list[str]
    confidence: npt.NDArray[np.float64]
    match_ratio: npt.NDArray[np.float64]
    lag_s: npt.NDArray[np.float64]
    samples: dict[str, npt.NDArray[np.uint8]]
    output: bytes = b""
    draws: list[dict[str, float]] = field(default_factory=list)
    errors: int = 0
    last_error: str | None = None

    @property
    def frames(self) -> int:
        return int(self.t.size)

    @property
    def seconds(self) -> float:
        """Frame-clock time covered, counting the last frame's period."""
        return self.frames / self.fps if self.fps > 0 else 0.0


class _OutputRecorder:
    """Engine output listener: per-channel credit, health, sample counts and samples."""

    def __init__(self, keep_samples: bool) -> None:
        self.keep_samples = keep_samples
        self.credit: dict[str, list[float]] = {name: [] for name in CAMERA_CHANNELS}
        self.n: dict[str, list[int]] = {name: [] for name in CAMERA_CHANNELS}
        self.state: dict[str, list[str]] = {name: [] for name in CAMERA_CHANNELS}
        self.samples: dict[str, list[npt.NDArray[np.uint8]]] = {n: [] for n in CAMERA_CHANNELS}

    def __call__(self, out: ChannelOutput, credit: float, state: HealthState) -> None:
        name = out.channel
        if name not in self.credit:
            return
        self.credit[name].append(float(credit))
        self.n[name].append(int(out.samples.size))
        self.state[name].append(state.value)
        if self.keep_samples and out.samples.size:
            self.samples[name].append(np.array(out.samples, dtype=np.uint8, copy=True).ravel())


def _engine_config(spec: str, data_dir: Path, **overrides: Any) -> PufferConfig:
    values: dict[str, Any] = {
        "source": spec,
        "liveness": "off",
        "led": "none",
        "beacon_enabled": False,
        "data_dir": data_dir,
    }
    values.update(overrides)
    return PufferConfig(**values)


def _describe_source(source: object) -> dict[str, Any]:
    if isinstance(source, SyntheticAquarium):
        cfg = source.config
        info: dict[str, Any] = {
            "kind": "synthetic",
            "name": "simulated aquarium",
            "seed": int(source.seed),
            "width": cfg.width,
            "height": cfg.height,
            "fps": cfg.fps,
            "fish": len(cfg.cast),
            "mode": cfg.mode,
            "read_noise_day": cfg.noise_sigma_day,
            "read_noise_night": cfg.noise_sigma_night,
            "shot_noise": cfg.shot_noise,
            "jpeg_quality": cfg.jpeg_quality,
        }
        if cfg.mode == "cycle":
            info.update(
                cycle_day_s=cfg.cycle_day_s,
                cycle_night_s=cfg.cycle_night_s,
                transition_s=cfg.transition_s,
            )
        return info
    if isinstance(source, VideoFileSource):
        return {
            "kind": "video",
            "name": f"recorded video {source.path.name}",
            "path": source.path.name,
            "width": source.width,
            "height": source.height,
            "fps": source.fps,
            "file_frames": source.frame_count,
        }
    return {"kind": "camera", "name": str(getattr(source, "source_id", "camera"))}


def _drive(
    engine: Any,
    job: _RunJob,
    *,
    on_frame: Callable[[Any], None] | None = None,
) -> RunData:
    """Step ``engine`` for ``job.frames`` frames and record everything."""
    recorder = _OutputRecorder(job.keep_samples)
    unsubscribe = engine.add_output_listener(recorder)
    rows: list[tuple[Any, ...]] = []
    output = bytearray()
    draws: list[dict[str, float]] = []
    n_calls = math.ceil(job.output_bytes / CHUNK_BYTES) if job.output_bytes else 0
    due = [round((k + 1) * job.frames / (n_calls + 1)) for k in range(n_calls)]

    def generate() -> None:
        n = min(CHUNK_BYTES, job.output_bytes - len(output))
        data, receipt = engine.rng.generate(n, "report")
        output.extend(data)
        channels = receipt.to_dict()["channels"]
        draws.append({name: float(c["credited_bits"]) for name, c in channels.items()})

    started = time.perf_counter()
    try:
        for i in range(job.frames):
            snap = engine.step()
            if snap is None:
                break
            channels = snap.channels
            live = snap.liveness
            rows.append(
                (
                    snap.t,
                    snap.mode is LightMode.NIGHT,
                    *(channels[name].bits_per_s for name in (*CAMERA_CHANNELS, CHANNEL_SYSTEM)),
                    *(
                        math.nan
                        if channels[name].h_per_sample is None
                        else channels[name].h_per_sample
                        for name in CAMERA_CHANNELS
                    ),
                    sum(1 for track in snap.tracks if track.confirmed),
                    live.state.value,
                    live.confidence_bits,
                    live.match_ratio,
                    live.lag_s,
                )
            )
            if on_frame is not None:
                on_frame(engine)
            while due and i + 1 >= due[0]:
                due.pop(0)
                generate()
        while len(output) < job.output_bytes:
            generate()
    finally:
        unsubscribe()
    wall = time.perf_counter() - started
    n = len(rows)
    cols = list(zip(*rows, strict=True)) if rows else [()] * 13
    source = engine.source
    fps = float(getattr(source, "fps", 0.0) or 0.0)
    return RunData(
        label=job.label,
        spec=job.spec,
        source=_describe_source(source),
        fps=fps,
        wall_s=wall,
        t=np.asarray(cols[0], dtype=np.float64),
        night=np.asarray(cols[1], dtype=np.bool_),
        credit={k: np.asarray(v[:n], dtype=np.float64) for k, v in recorder.credit.items()},
        n_samples={k: np.asarray(v[:n], dtype=np.int64) for k, v in recorder.n.items()},
        state={k: list(v[:n]) for k, v in recorder.state.items()},
        bits_per_s={
            name: np.asarray(cols[2 + i], dtype=np.float64)
            for i, name in enumerate((*CAMERA_CHANNELS, CHANNEL_SYSTEM))
        },
        h_est={
            name: np.asarray(cols[5 + i], dtype=np.float64)
            for i, name in enumerate(CAMERA_CHANNELS)
        },
        tracks=np.asarray(cols[7], dtype=np.int64),
        liveness_state=[str(s) for s in cols[8]],
        confidence=np.asarray(cols[9], dtype=np.float64),
        match_ratio=np.asarray(cols[10], dtype=np.float64),
        lag_s=np.asarray(cols[11], dtype=np.float64),
        samples={
            k: np.concatenate(v) if v else np.empty(0, dtype=np.uint8)
            for k, v in recorder.samples.items()
        },
        output=bytes(output),
        draws=draws,
        errors=int(engine.errors),
        last_error=engine.last_error,
    )


def _collect_run(job: _RunJob) -> RunData:
    """One analysis run: the source of ``job`` through a liveness-free engine."""
    with tempfile.TemporaryDirectory(prefix="puffer-report-") as tmp:
        engine = build_engine(_engine_config(job.spec, Path(tmp)))
        try:
            return _drive(engine, job)
        finally:
            engine.close()


def _seeded_schedule(seed: int) -> ChallengeSchedule:
    """Challenge A: a reproducible LED pattern (production draws from ``os.urandom``)."""
    generator = np.random.default_rng(seed)
    return ChallengeSchedule(symbol_s=PufferConfig().symbol_s, random_bytes=generator.bytes)


@dataclass
class LivenessData:
    live: RunData
    replay: RunData
    control: RunData | None
    led_roi: tuple[int, int, int, int] | None
    video_bytes: int
    video_frames: int
    codec: str = "FFV1"


def _collect_liveness(job: _LivenessJob) -> LivenessData:
    """Live run (challenge A, recorded losslessly), replay (fresh challenge), control (A)."""
    import cv2

    with tempfile.TemporaryDirectory(prefix="puffer-report-live-") as tmp:
        tmpdir = Path(tmp)
        video = tmpdir / "tank.mkv"
        writer: Any = None

        def record(engine: Any) -> None:
            nonlocal writer
            frame = engine.latest_frame()
            if frame is None:
                return
            if writer is None:
                fourcc = cv2.VideoWriter.fourcc(*"FFV1")
                writer = cv2.VideoWriter(
                    os.fspath(video), fourcc, float(engine.source.fps), (frame.width, frame.height)
                )
                if not writer.isOpened():
                    raise RuntimeError("cannot write an FFV1 .mkv (OpenCV without FFmpeg?)")
            writer.write(frame.image)

        live_cfg = _engine_config(job.spec, tmpdir / "live", liveness="required", led="auto")
        engine = build_engine(live_cfg, schedule=_seeded_schedule(job.schedule_seed))
        try:
            led_roi = engine.led_roi
            live = _drive(
                engine, _RunJob("live", job.spec, job.frames, keep_samples=False), on_frame=record
            )
        finally:
            engine.close()
            if writer is not None:
                writer.release()

        video_spec = f"video:{video}"
        roi = {"led_roi": led_roi} if led_roi is not None else {}
        replay_cfg = _engine_config(video_spec, tmpdir / "replay", liveness="required", **roi)
        engine = build_engine(replay_cfg)  # a fresh challenge: os.urandom bits drawn now
        try:
            replay = _drive(
                engine, _RunJob("replay", video_spec, job.frames + 1, keep_samples=False)
            )
        finally:
            engine.close()

        control = None
        if job.control:
            control_cfg = _engine_config(video_spec, tmpdir / "control", liveness="required", **roi)
            engine = build_engine(control_cfg, schedule=_seeded_schedule(job.schedule_seed))
            try:
                control = _drive(
                    engine, _RunJob("control", video_spec, job.frames + 1, keep_samples=False)
                )
            finally:
                engine.close()
        size = video.stat().st_size if video.exists() else 0
    return LivenessData(live, replay, control, led_roi, size, replay.frames)


def _collect_tracking(job: _TrackingJob) -> dict[str, Any]:
    """Tracking quality on the simulator's ground truth (``puffer.vision.evaluate``)."""
    from puffer.vision import FishTracker
    from puffer.vision.evaluate import evaluate_tracking

    source = open_source(job.spec)
    try:
        if not isinstance(source, SyntheticAquarium):  # pragma: no cover - plan guarantees it
            raise TypeError("tracking evaluation needs the simulator")
        started = time.perf_counter()
        metrics = evaluate_tracking(source, FishTracker(), job.frames, warmup=job.warmup)
        wall = time.perf_counter() - started
        return {
            "label": job.label,
            "spec": job.spec,
            "source": _describe_source(source),
            "frames": job.frames,
            "warmup": job.warmup,
            "wall_s": wall,
            "metrics": metrics.to_dict(),
        }
    finally:
        source.close()


def _worker_init() -> None:
    """One OpenCV thread per worker: the workers already use every core between them."""
    import cv2

    cv2.setNumThreads(1)


class _Pool:
    """``submit`` that runs in worker processes when ``workers > 1``, inline otherwise."""

    def __init__(self, workers: int) -> None:
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        if workers > 1:
            context = multiprocessing.get_context("spawn")
            self._executor = concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=context, initializer=_worker_init
            )

    def submit(self, fn: Callable[..., _R], *args: Any) -> concurrent.futures.Future[_R]:
        if self._executor is not None:
            return self._executor.submit(fn, *args)
        future: concurrent.futures.Future[_R] = concurrent.futures.Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> _Pool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _wait(
    futures: Mapping[str, concurrent.futures.Future[Any]], say: Progress, verb: str
) -> dict[str, Any]:
    """Results by key, reporting each one as it completes."""
    keys = {id(f): k for k, f in futures.items()}
    out: dict[str, Any] = {}
    for future in concurrent.futures.as_completed(list(futures.values())):
        key = keys[id(future)]
        out[key] = future.result()
        say(f"{verb} {key}")
    return {k: out[k] for k in futures}


# ======================================================================== analysis: SP 800-90B


def _assess(samples: npt.NDArray[np.uint8], max_samples: int) -> dict[str, Any]:
    """SP 800-90B non-IID assessment of the first ``max_samples`` samples (8 bits each)."""
    used = samples[:max_samples]
    base: dict[str, Any] = {
        "n_available": int(samples.size),
        "n_used": int(used.size),
        "standard_samples": SP800_90B_SAMPLES,
        "fraction_of_standard": used.size / SP800_90B_SAMPLES,
    }
    started = time.perf_counter()
    try:
        assessment = assess_non_iid(used, 8)
    except InsufficientDataError as exc:
        return {**base, "min_entropy": None, "error": str(exc)}
    out = {**base, **assessment.to_dict(), "seconds": time.perf_counter() - started}
    out["bitstring_limit"] = BITSTRING_LIMIT
    return out


def credit_rate(h_assessed: float | None, channel: str, safety_factor: float) -> float:
    """Credited bits per sample the engine's rule gives an assessed ``H``: ``min(H, cap) * sf``."""
    if h_assessed is None or not math.isfinite(h_assessed) or h_assessed <= 0:
        return 0.0
    return min(h_assessed, DEFAULT_H_CAP[channel]) * safety_factor


# ======================================================================== analysis: credit


def _phase_stats(run: RunData, mask: npt.NDArray[np.bool_], seconds: float) -> dict[str, Any]:
    out: dict[str, Any] = {"frames": int(mask.sum()), "seconds": seconds}
    for name in CAMERA_CHANNELS:
        credit = run.credit[name][mask]
        n = run.n_samples[name][mask]
        states = [s for s, m in zip(run.state[name], mask, strict=True) if m]
        counts = {state: states.count(state) for state in STATES}
        total = max(1, len(states))
        out[name] = {
            "credited_bits": float(credit.sum()),
            "bits_per_s": float(credit.sum() / seconds) if seconds > 0 else 0.0,
            "samples": int(n.sum()),
            "samples_per_frame": float(n.mean()) if n.size else 0.0,
            # per sample while credited: start-up and FAILED outputs earn 0 by design
            "credited_per_sample": (
                float(credit.sum() / n[credit > 0].sum()) if n[credit > 0].sum() else 0.0
            ),
            "health": {state: counts[state] / total for state in STATES},
        }
    out["tracks_per_frame"] = float(run.tracks[mask].mean()) if mask.any() else 0.0
    return out


def _run_summary(run: RunData, assessed: Mapping[str, Any] | None) -> dict[str, Any]:
    """Credit and estimates of one run, as the engine produced them."""
    everything = np.ones(run.frames, dtype=np.bool_)
    out: dict[str, Any] = {
        "label": run.label,
        "spec": run.spec,
        "source": run.source,
        "frames": run.frames,
        "seconds": run.seconds,
        "night_fraction": float(run.night.mean()) if run.frames else 0.0,
        "wall_s": run.wall_s,
        "engine_fps": run.frames / run.wall_s if run.wall_s > 0 else 0.0,
        "errors": run.errors,
        "last_error": run.last_error,
        "output_bytes": len(run.output),
        "draws": len(run.draws),
        "overall": _phase_stats(run, everything, run.seconds),
    }
    for name in CAMERA_CHANNELS:
        stats = out["overall"][name]
        credited = np.flatnonzero(run.credit[name] > 0)
        if credited.size:
            first = int(credited[0])
            span = run.seconds - first / run.fps
            stats["first_credit_s"] = first / run.fps
            stats["steady_bits_per_s"] = float(run.credit[name][first:].sum() / span)
        else:
            stats["first_credit_s"] = None
            stats["steady_bits_per_s"] = 0.0
        h = run.h_est[name]
        ready = h[np.isfinite(h) & (h > 0)]
        stats["h_engine_mean"] = float(ready.mean()) if ready.size else None
        stats["h_engine_final"] = float(h[-1]) if h.size and np.isfinite(h[-1]) else None
        stats["h_cap"] = DEFAULT_H_CAP[name]
        stats["h_assessed"] = (
            None if assessed is None else assessed.get(name, {}).get("min_entropy")
        )
    out["system"] = {
        "bits_per_draw": 256.0,
        "draws": len(run.draws),
        "credited_bits": 256.0 * len(run.draws),
    }
    return out


def _series(run: RunData, *, period_s: float = SERIES_PERIOD_S) -> dict[str, Any]:
    """The run sampled every ``period_s`` of frame time (engine EMA rates, estimates)."""
    if not run.frames:
        return {"t": []}
    step = max(1, round(period_s * run.fps))
    idx = np.arange(0, run.frames, step)

    def pick(values: npt.NDArray[Any], digits: int) -> list[float | None]:
        return [None if not math.isfinite(v) else round(float(v), digits) for v in values[idx]]

    return {
        "period_s": step / run.fps,
        "t": pick(run.t, 3),
        "night": [bool(v) for v in run.night[idx]],
        "bits_per_s": {name: pick(run.bits_per_s[name], 2) for name in CAMERA_CHANNELS},
        "h_est": {name: pick(run.h_est[name], 4) for name in CAMERA_CHANNELS},
        "tracks": [int(v) for v in run.tracks[idx]],
        "health": {name: [run.state[name][i] for i in idx] for name in CAMERA_CHANNELS},
    }


def _credit_bins(run: RunData, bin_s: float = 1.0) -> dict[str, Any]:
    """Exact credited bits per second of frame time, in ``bin_s`` bins (no smoothing)."""
    if not run.frames:
        return {"bin_s": bin_s, "t": []}
    idx = np.floor((run.t - run.t[0]) / bin_s + 1e-9).astype(np.int64)
    n_bins = int(idx[-1]) + 1
    frames = np.bincount(idx, minlength=n_bins).astype(np.float64)
    seconds = np.where(frames > 0, frames / run.fps, 1.0)
    night = np.bincount(idx, weights=run.night.astype(np.float64), minlength=n_bins)
    return {
        "bin_s": bin_s,
        "t": [round(float(run.t[0]) + i * bin_s, 3) for i in range(n_bins)],
        "night": [bool(n > f / 2) for n, f in zip(night, frames, strict=True)],
        "bits_per_s": {
            name: [
                round(float(v), 2)
                for v in np.bincount(idx, weights=run.credit[name], minlength=n_bins) / seconds
            ]
            for name in CAMERA_CHANNELS
        },
        "tracks": [
            round(float(v), 2)
            for v in np.bincount(idx, weights=run.tracks.astype(np.float64), minlength=n_bins)
            / np.maximum(frames, 1)
        ],
    }


def _cycle_summary(run: RunData) -> dict[str, Any]:
    src = run.source
    day_s = float(src.get("cycle_day_s", 0.0))
    night_s = float(src.get("cycle_night_s", 0.0))
    t = run.t
    phases = []
    bounds = [
        ("day", 0.0, day_s),
        ("night", day_s, day_s + night_s),
        ("night, second half", day_s + night_s / 2, day_s + night_s),
        ("day (again)", day_s + night_s, float(t[-1]) + 1.0 / run.fps),
    ]
    for name, start, end in bounds:
        mask = (t >= start) & (t < end)
        if mask.any():
            phases.append(
                {
                    "phase": name,
                    "start_s": start,
                    "end_s": end,
                    **_phase_stats(run, mask, end - start),
                }
            )
    return {
        "spec": run.spec,
        "source": src,
        "frames": run.frames,
        "seconds": run.seconds,
        "phases": phases,
        "bins": _credit_bins(run),
        "series": _series(run),
    }


# ======================================================================== analysis: streams


def _per_frame_split(run: RunData, name: str) -> list[npt.NDArray[np.uint8]]:
    counts = run.n_samples[name]
    return np.split(run.samples[name], np.cumsum(counts)[:-1]) if counts.size else []


def _conditioned_streams(
    run: RunData, assessed: Mapping[str, Any], safety_factor: float
) -> dict[str, dict[str, Any]]:
    """Sensor only, fish only and combined, compressed without OS randomness."""
    rates = {
        name: credit_rate(assessed.get(name, {}).get("min_entropy"), name, safety_factor)
        for name in CAMERA_CHANNELS
    }
    inputs: dict[str, tuple[npt.NDArray[np.uint8], float]] = {
        name: (run.samples[name], rates[name]) for name in CAMERA_CHANNELS
    }
    n_total = sum(int(run.samples[name].size) for name in CAMERA_CHANNELS)
    if n_total:
        combined_rate = sum(rates[n] * run.samples[n].size for n in CAMERA_CHANNELS) / n_total
        frames = zip(*(_per_frame_split(run, n) for n in CAMERA_CHANNELS), strict=True)
        parts = [chunk for frame in frames for chunk in frame if chunk.size]
        combined = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint8)
        inputs["combined"] = (combined, combined_rate)
    out: dict[str, dict[str, Any]] = {}
    for name, (samples, rate) in inputs.items():
        info: dict[str, Any] = {
            "samples_in": int(samples.size),
            "h_per_sample": rate,
            "block_entropy_bits": BLOCK_ENTROPY_BITS,
            "out_bytes_per_block": OUTPUT_BLOCK_BYTES,
        }
        if rate <= 0 or not samples.size:
            out[name] = {**info, "data": b"", "error": "no credited min-entropy to condition"}
            continue
        stream = ConditionedStream(min(rate, 8.0), BLOCK_ENTROPY_BITS, OUTPUT_BLOCK_BYTES)
        data = stream.feed(samples)
        out[name] = {
            **info,
            "block_samples": stream.block_samples,
            "blocks": stream.blocks_out,
            "data": data,
        }
    return out


def _nist(data: bytes, n_bits: int, max_sequences: int) -> dict[str, Any]:
    """SP 800-22: section 4.2 over 10^6-bit sequences, or the tests that fit one short one."""
    bits = bytes_to_bits(data)
    m = min(max_sequences, bits.size // n_bits)
    started = time.perf_counter()
    if m >= 1:
        multi = run_multi(split_sequences(bits[: m * n_bits], n_bits))
        out = _multi_summary(multi)
    elif bits.size >= 100:
        out = _single_summary(run_suite(bits))
    else:
        return {
            "mode": "none",
            "n_bits": int(bits.size),
            "sequences": 0,
            "error": "fewer than 100 bits",
        }
    out["available_bits"] = int(bits.size)
    out["seconds"] = time.perf_counter() - started
    return out


def _multi_summary(multi: MultiSuiteResult) -> dict[str, Any]:
    p_values: dict[str, list[float]] = {}
    for suite in multi.suites:
        for result in suite.applicable:
            for name, p in result.subtests():
                p_values.setdefault(name, []).append(round(float(p), 6))
    applicable = {s.test for s in multi.subtests}
    skipped = {r.name: r.skipped or "" for r in multi.suites[0].results if r.name not in applicable}
    per_test: dict[str, Any] = {}
    for test in SUITE_TESTS:
        subs = [s for s in multi.subtests if s.test == test]
        if not subs:
            continue
        proportions = [s.proportion for s in subs]
        per_test[test] = {
            "subtests": len(subs),
            "sequences": max(s.sequences for s in subs),
            "min_proportion": min(proportions),
            "mean_proportion": float(np.mean(proportions)),
            "proportion_interval": list(subs[0].proportion_interval),
            "flagged": sum(not s.passed for s in subs),
            "proportion_failures": sum(not s.proportion_ok for s in subs),
            "uniformity_failures": sum(s.uniform is False for s in subs),
            "min_uniformity_p": min(s.uniformity_p for s in subs),
        }
    histogram = [sum(s.histogram[i] for s in multi.subtests) for i in range(10)]
    expected_flagged = sum(_chance_flag(s.sequences, s.proportion_interval) for s in multi.subtests)
    return {
        "mode": "multi",
        "n_bits": multi.n_bits,
        "sequences": multi.n_sequences,
        "alpha": multi.alpha,
        "uniformity_assessed": multi.n_sequences >= MIN_SEQUENCES_FOR_UNIFORMITY,
        "passed": multi.passed,
        "flagged": [s.name for s in multi.flagged],
        "expected_flagged": expected_flagged,
        "subtest_count": len(multi.subtests),
        "per_test": per_test,
        "skipped": skipped,
        "pooled_histogram": [int(v) for v in histogram],
        "subtests": [s.to_dict() for s in multi.subtests],
        "p_values": p_values,
    }


def _chance_tail(observed: int, expected: float) -> float:
    """``P[Poisson(expected) >= observed]``: how often chance alone puts this many outside."""
    return float(poisson.sf(observed - 1, expected)) if observed > 0 else 1.0


def _band_edge(res: Mapping[str, Any]) -> float:
    """Lower acceptance edge for sub-tests that ran on every sequence of a multi result."""
    full = [
        s["proportion_interval"][0] for s in res["subtests"] if s["sequences"] == res["sequences"]
    ]
    return min(full) if full else min(s["proportion_interval"][0] for s in res["subtests"])


def _chance_flag(m: int, interval: tuple[float, float]) -> float:
    """P[a sub-test of an ideal generator is flagged] with ``m`` sequences (section 4.2).

    Flagged means the pass proportion leaves the band (binomial, p = 0.99 per
    sequence) or, from 55 sequences on, the 10-bin uniformity check fails,
    which by construction happens with probability 0.0001.
    """
    lo, hi = interval
    k = np.arange(m + 1)
    outside = (k / m < lo - 1e-12) | (k / m > hi + 1e-12)
    pmf = binom.pmf(k, m, 1.0 - _ALPHA)
    p_flag = float(pmf[outside].sum())
    if m >= MIN_SEQUENCES_FOR_UNIFORMITY:
        p_flag = 1.0 - (1.0 - p_flag) * (1.0 - UNIFORMITY_THRESHOLD)
    return p_flag


def _single_summary(suite: SuiteResult) -> dict[str, Any]:
    tests = []
    for r in suite.results:
        tests.append(
            {
                "name": r.name,
                "skipped": r.skipped,
                "passed": r.passed if r.applicable else None,
                "p_values": [round(float(p), 6) for p in r.p_values],
                "min_p": min(r.p_values) if r.p_values else None,
            }
        )
    return {
        "mode": "single",
        "n_bits": suite.n_bits,
        "sequences": 1,
        "alpha": suite.alpha,
        "passed": suite.passed,
        "applicable": len(suite.applicable),
        "p_value_count": suite.p_value_count,
        "failed_count": suite.failed_count,
        "expected_false_failures": suite.expected_false_failures,
        "failures_tail_probability": suite.failures_tail_probability(),
        "tests": tests,
        "skipped": {r.name: r.skipped or "" for r in suite.skipped},
    }


def _ent(data: bytes) -> dict[str, Any] | None:
    return ent_stats(data).to_dict() if data else None


def _ent_binary(streams: Mapping[str, bytes]) -> dict[str, Any]:
    """The real ``ent`` on a few streams, as a cross-check of the in-process figures."""
    if not ent_available():
        return {"available": False}
    out: dict[str, Any] = {"available": True, "results": {}}
    with tempfile.TemporaryDirectory(prefix="puffer-report-ent-") as tmp:
        for name, data in streams.items():
            if not data:
                continue
            path = Path(tmp) / "stream.bin"
            path.write_bytes(data)
            try:
                out["results"][name] = run_ent(path).to_dict()
            except ExternalToolError as exc:
                out["results"][name] = {"error": str(exc)}
    return out


# ======================================================================== analysis: liveness


def _liveness_run(run: RunData) -> dict[str, Any]:
    verified = [i for i, s in enumerate(run.liveness_state) if s == LivenessState.VERIFIED.value]
    camera = sum(float(run.credit[name].sum()) for name in CAMERA_CHANNELS)
    first_credit = np.flatnonzero(run.credit[CHANNEL_SENSOR] > 0)
    after = run.liveness_state[verified[0] :] if verified else []
    step = max(1, round(0.1 * run.fps)) if run.fps else 1
    idx = np.arange(0, run.frames, step)
    judged_states = {LivenessState.VERIFIED.value, LivenessState.FAILED.value}
    judged = np.asarray([s in judged_states for s in run.liveness_state], dtype=np.bool_)
    return {
        "frames": run.frames,
        "seconds": run.seconds,
        "time_to_verified_s": float(run.t[verified[0]]) if verified else None,
        "time_to_first_credit_s": float(run.t[first_credit[0]]) if first_credit.size else None,
        "verified_frames": len(verified),
        "verified_fraction_after_first": (
            sum(s == LivenessState.VERIFIED.value for s in after) / len(after) if after else 0.0
        ),
        "max_confidence_bits": float(np.max(run.confidence)) if run.frames else 0.0,
        "final_confidence_bits": float(run.confidence[-1]) if run.frames else 0.0,
        # judged windows only: while CALIBRATING the ratio covers a partial window
        "max_match_ratio": float(np.max(run.match_ratio[judged])) if judged.any() else None,
        "judged_frames": int(judged.sum()),
        "final_match_ratio": float(run.match_ratio[-1]) if run.frames else 0.0,
        "final_lag_s": float(run.lag_s[-1]) if run.frames else 0.0,
        "final_state": run.liveness_state[-1] if run.frames else None,
        "camera_credited_bits": camera,
        "series": {
            "t": [round(float(v), 3) for v in run.t[idx]],
            "confidence_bits": [round(float(v), 2) for v in run.confidence[idx]],
            "match_ratio": [round(float(v), 4) for v in run.match_ratio[idx]],
            "state": [run.liveness_state[i] for i in idx],
        },
    }


def _liveness_summary(data: LivenessData, spec: str, seed: int) -> dict[str, Any]:
    return {
        "spec": spec,
        "source": data.live.source,
        "schedule_seed": seed,
        "symbol_s": PufferConfig().symbol_s,
        "window_bits": PufferConfig().liveness_window_bits,
        "threshold_bits": LIVENESS_THRESHOLD_BITS,
        "min_match": LIVENESS_MIN_MATCH,
        "led_roi": list(data.led_roi) if data.led_roi else None,
        "recording": {"codec": data.codec, "bytes": data.video_bytes, "frames": data.video_frames},
        "live": _liveness_run(data.live),
        "replay": _liveness_run(data.replay),
        "control": _liveness_run(data.control) if data.control is not None else None,
    }


# ======================================================================== dieharder


def _dieharder(spec: str, config: ReportConfig, say: Progress) -> dict[str, Any]:
    """PufferRNG output through ``dieharder -g 200`` (raw stdin) while an engine runs."""
    exe = shutil.which("dieharder")
    out: dict[str, Any] = {"available": exe is not None, "ran": False}
    if exe is None:
        out["reason"] = "dieharder is not installed"
        return out
    entries = config.dieharder_tests or DIEHARDER_SELECTION
    selections = [_dieharder_args(e) for e in entries]
    out.update(
        spec=spec,
        budget_s=config.dieharder_budget_s,
        selection=list(entries),
        commands=[" ".join(["dieharder", "-g", "200", *s]) for s in selections],
    )
    rows: list[dict[str, Any]] = []
    fed = calls = 0
    timed_out = False
    header = ""
    started = time.monotonic()
    deadline = started + config.dieharder_budget_s
    with tempfile.TemporaryDirectory(prefix="puffer-report-dh-") as tmp:
        engine = build_engine(_engine_config(spec, Path(tmp)))
        engine.start()  # keeps absorbing frames, like `puffer random --stream`
        try:
            for selection in selections:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                say(f"dieharder {' '.join(selection)} (budget left {remaining:.0f} s)")
                text, n_bytes, n_calls, expired = _dieharder_once(
                    [exe, "-g", "200", *selection], engine.rng, remaining
                )
                fed += n_bytes
                calls += n_calls
                header = header or _dieharder_version(text)
                rows += [r.to_dict() for r in parse_dieharder_output(text)]
                if expired:
                    timed_out = True
                    break
        finally:
            snap = engine.snapshot()
            frames = snap.frame_index + 1 if snap is not None else 0
            engine.close()
    counts = {k: sum(r["assessment"] == k for r in rows) for k in ("PASSED", "WEAK", "FAILED")}
    out.update(
        ran=True,
        version=header,
        elapsed_s=time.monotonic() - started,
        timed_out=timed_out,
        bytes_fed=fed,
        generate_calls=calls,
        engine_frames=frames,
        results=rows,
        counts=counts,
        tests=sorted({r["test_name"] for r in rows}),
    )
    return out


def _dieharder_version(text: str) -> str:
    match = re.search(r"dieharder version (\S+)", text)
    return f"dieharder {match.group(1)}" if match else ""


def _dieharder_once(cmd: list[str], rng: Any, timeout_s: float) -> tuple[str, int, int, bool]:
    """Run one dieharder process fed by ``rng``; (output, bytes fed, generate calls, timed out)."""
    proc = subprocess.Popen(  # noqa: S603 - fixed argument list, no shell
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    stdin, stdout = proc.stdin, proc.stdout
    stop = threading.Event()
    counters = [0, 0]
    lines: list[bytes] = []

    def feed() -> None:
        try:
            while not stop.is_set():
                data, _ = rng.generate(CHUNK_BYTES, "dieharder")
                stdin.write(data)
                counters[0] += len(data)
                counters[1] += 1
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass  # dieharder closed its end: it has read enough
        finally:
            with contextlib.suppress(OSError, ValueError):
                stdin.close()

    def read() -> None:
        lines.extend(stdout)

    feeder = threading.Thread(target=feed, name="dieharder-feed", daemon=True)
    reader = threading.Thread(target=read, name="dieharder-read", daemon=True)
    feeder.start()
    reader.start()
    expired = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        expired = True
        proc.kill()
        proc.wait()
    finally:
        stop.set()
        feeder.join(10.0)
        reader.join(10.0)
    if not expired and proc.returncode != 0:
        tail = b"".join(lines)[-500:].decode("utf-8", "replace")
        raise ExternalToolError(f"dieharder exited with {proc.returncode}: {tail}")
    return b"".join(lines).decode("utf-8", "replace"), counters[0], counters[1], expired


# ======================================================================== orchestration


def run_report(config: ReportConfig, *, progress: Progress | None = None) -> Path:
    """Collect, analyse and write the report; return the path of ``RESULTS.md``.

    Writes ``RESULTS.md`` and ``results.json`` into ``config.out_dir`` and
    the charts (``<name>-light.png`` / ``<name>-dark.png``) into
    ``config.media_dir``.
    """
    say: Progress = progress if progress is not None else (lambda _message: None)
    started = time.monotonic()
    plan = _plan(config)
    safety = PufferConfig().safety_factor
    timings: dict[str, float] = {}
    say(f"seed {plan.seed} ({plan.seed_source}); {config.workers} worker(s)")

    with _Pool(config.workers) as pool:
        # 1. collection: independent engine runs
        mark = time.monotonic()
        futures: dict[str, concurrent.futures.Future[Any]] = {}
        for job in plan.runs:
            say(f"collecting {job.label}: {job.spec}, {job.frames} frames")
            futures[f"run:{job.label}"] = pool.submit(_collect_run, job)
        if plan.cycle is not None:
            say(f"collecting cycle: {plan.cycle.spec}, {plan.cycle.frames} frames")
            futures["cycle"] = pool.submit(
                _collect_run, dataclasses.replace(plan.cycle, keep_samples=False)
            )
        if plan.liveness is not None:
            say(
                f"collecting liveness: live, replay and control, {plan.liveness.frames} frames each"
            )
            futures["liveness"] = pool.submit(_collect_liveness, plan.liveness)
        for tjob in plan.tracking:
            futures[f"tracking:{tjob.label}"] = pool.submit(_collect_tracking, tjob)
        collected = _wait(futures, say, "collected")
        timings["collection_s"] = time.monotonic() - mark
        runs: dict[str, RunData] = {j.label: collected[f"run:{j.label}"] for j in plan.runs}

        # 2. SP 800-90B per run and channel
        mark = time.monotonic()
        assess_futures = {
            f"{label}/{name}": pool.submit(_assess, run.samples[name], config.max_90b_samples)
            for label, run in runs.items()
            for name in CAMERA_CHANNELS
        }
        assessed_flat = _wait(assess_futures, say, "assessed")
        assessed = {
            label: {name: assessed_flat[f"{label}/{name}"] for name in CAMERA_CHANNELS}
            for label in runs
        }
        timings["sp800_90b_s"] = time.monotonic() - mark

        # 3. conditioned streams, final output: SP 800-22 and ent
        mark = time.monotonic()
        say("conditioning streams")
        streams: dict[str, dict[str, Any]] = {}
        stream_meta: dict[str, dict[str, Any]] = {}
        for label, run in runs.items():
            for name, info in _conditioned_streams(run, assessed[label], safety).items():
                key = f"{label}/{name}"
                streams[key] = info
        first = plan.runs[0].label
        final_key = f"{first}/pufferrng"
        streams[final_key] = {"data": runs[first].output, "keyed_with_os_urandom": True}
        nist_futures = {}
        for key, info in streams.items():
            limit = config.nist_sequences
            if key == final_key:
                limit = max(config.nist_sequences, config.output_sequences)
            nist_futures[key] = pool.submit(_nist, info["data"], config.nist_bits, limit)
        nist = _wait(nist_futures, say, "SP 800-22 on")
        for key, info in streams.items():
            meta = {k: v for k, v in info.items() if k != "data"}
            meta["bytes"] = len(info["data"])
            meta["bits"] = 8 * len(info["data"])
            stream_meta[key] = meta
        ent_raw = {
            f"{label}/{name}": _ent(run.samples[name].tobytes())
            for label, run in runs.items()
            for name in CAMERA_CHANNELS
        }
        ent_streams = {key: _ent(info["data"]) for key, info in streams.items()}
        binary = _ent_binary(
            {
                f"{first}/sensor (raw)": runs[first].samples[CHANNEL_SENSOR].tobytes(),
                final_key: runs[first].output,
            }
        )
        timings["streams_s"] = time.monotonic() - mark

    # 4. dieharder (needs the cores the pool just released)
    mark = time.monotonic()
    if config.dieharder:
        if dieharder_available():
            dieharder = _dieharder(plan.runs[0].spec, config, say)
        else:
            dieharder = {"available": False, "ran": False, "reason": "dieharder is not installed"}
    else:
        dieharder = {"available": dieharder_available(), "ran": False, "reason": "not requested"}
    timings["dieharder_s"] = time.monotonic() - mark

    results: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": _provenance(),
        "config": config.to_dict(),
        "seed": plan.seed,
        "seed_source": plan.seed_source,
        "workers": config.workers,
        "engine": _engine_settings(),
        "skipped_sections": plan.skipped,
        "runs": {label: _run_summary(run, assessed[label]) for label, run in runs.items()},
        "sp800_90b": assessed,
        "streams": stream_meta,
        "sp800_22": nist,
        "ent": {"raw": ent_raw, "streams": ent_streams, "binary": binary},
        "final_output": {
            "run": first,
            "bytes": len(runs[first].output),
            "generate_calls": len(runs[first].draws),
            "chunk_bytes": CHUNK_BYTES,
            "draws": runs[first].draws[:16],
        },
        "cycle": _cycle_summary(collected["cycle"]) if "cycle" in collected else None,
        "liveness": (
            _liveness_summary(collected["liveness"], plan.liveness.spec, plan.seed)
            if plan.liveness is not None
            else None
        ),
        "tracking": {j.label: collected[f"tracking:{j.label}"] for j in plan.tracking} or None,
        "dieharder": dieharder,
    }
    timings["total_s"] = time.monotonic() - started
    results["timings"] = timings
    return write_results(results, config, progress=say)


def write_results(
    results: Mapping[str, Any], config: ReportConfig, *, progress: Progress | None = None
) -> Path:
    """Write ``results.json``, the charts and ``RESULTS.md``; return the Markdown path.

    Separate from :func:`run_report` so the documents can be re-rendered
    from a saved ``results.json`` without collecting anything again.
    """
    say: Progress = progress if progress is not None else (lambda _message: None)
    out_dir, media_dir = config.out_dir, config.media_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = _jsonable(results)
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(clean, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    say(f"wrote {json_path}")
    charts: dict[str, str] = {}
    if config.plots:
        from puffer.analysis import plots

        media_dir.mkdir(parents=True, exist_ok=True)
        charts = plots.render_all(clean, media_dir)
        say(f"wrote {2 * len(charts)} chart files to {media_dir}")
    md_path = out_dir / "RESULTS.md"
    media_rel = Path(os.path.relpath(media_dir.resolve(), out_dir.resolve()))
    md_path.write_text(render_markdown(clean, charts=charts, media_rel=media_rel), encoding="utf-8")
    say(f"wrote {md_path}")
    return md_path


def _engine_settings() -> dict[str, Any]:
    cfg = PufferConfig()
    return {
        "sensor_sites": cfg.sensor_sites,
        "safety_factor": cfg.safety_factor,
        "h_cap": dict(DEFAULT_H_CAP),
        "pool_capacity_bits": cfg.pool_capacity_bits,
        "sensor_h_assessed": cfg.sensor_h_assessed,
        "fish_h_assessed": cfg.fish_h_assessed,
        "vision_max_width": cfg.vision_max_width,
        "liveness_window_bits": cfg.liveness_window_bits,
        "symbol_s": cfg.symbol_s,
    }


def _provenance() -> dict[str, Any]:
    import cv2
    import nacl
    import scipy

    root = Path(__file__).resolve().parents[3]
    commit = "unknown"
    dirty: list[str] = []
    git = shutil.which("git")
    if git is not None and (root / ".git").exists():
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            commit = subprocess.run(  # noqa: S603 - read-only: which code produced the numbers
                [git, "-C", str(root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
            status = subprocess.run(  # noqa: S603
                [git, "-C", str(root), "status", "--porcelain", "--", "src/puffer"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            dirty = sorted(
                {line[3:].split("/")[2] for line in status.splitlines() if line[3:].count("/") >= 2}
            )
    return {
        "commit": commit,
        "modified_packages": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "pynacl": getattr(nacl, "__version__", "unknown"),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpus": os.cpu_count(),
    }


def _jsonable(value: Any) -> Any:
    """Plain JSON types; non-finite floats become ``None`` (strict JSON has no NaN)."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, bytes | bytearray):
        return len(value)
    if isinstance(value, str | bool) or value is None:
        return value
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Iterable):
        return [_jsonable(v) for v in value]
    return str(value)


# ======================================================================== Markdown

ESTIMATOR_LABELS: Final = {
    "most_common_value": "Most common value",
    "collision": "Collision",
    "markov": "Markov",
    "compression": "Compression",
    "t_tuple": "t-Tuple",
    "lrs": "LRS",
    "multi_mcw": "MultiMCW prediction",
    "lag": "Lag prediction",
    "multi_mmc": "MultiMMC prediction",
    "lz78y": "LZ78Y prediction",
}
ESTIMATOR_SECTIONS: Final = {
    "most_common_value": "6.3.1",
    "collision": "6.3.2",
    "markov": "6.3.3",
    "compression": "6.3.4",
    "t_tuple": "6.3.5",
    "lrs": "6.3.6",
    "multi_mcw": "6.3.7",
    "lag": "6.3.8",
    "multi_mmc": "6.3.9",
    "lz78y": "6.3.10",
}
TEST_LABELS: Final = {
    "frequency": "Frequency (monobit)",
    "block_frequency": "Block frequency",
    "cumulative_sums": "Cumulative sums",
    "runs": "Runs",
    "longest_run_of_ones": "Longest run of ones",
    "binary_matrix_rank": "Binary matrix rank",
    "discrete_fourier_transform": "Spectral (DFT)",
    "non_overlapping_template_matching": "Non-overlapping templates",
    "overlapping_template_matching": "Overlapping templates",
    "universal": "Maurer's universal",
    "approximate_entropy": "Approximate entropy",
    "random_excursions": "Random excursions",
    "random_excursions_variant": "Random excursions variant",
    "serial": "Serial",
    "linear_complexity": "Linear complexity",
}
STREAM_LABELS: Final = {
    "sensor": "Sensor only",
    "fish": "Fish only",
    "combined": "Combined (sensor + fish)",
    "pufferrng": "PufferRNG output",
}
CHANNEL_LABELS: Final = {CHANNEL_SENSOR: "Sensor noise", CHANNEL_FISH: "Fish motion"}


def _num(value: Any, digits: int = 0) -> str:
    """``1,234`` / ``1,234.5``; ``n/a`` for missing values."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _bits(value: Any) -> str:
    """Bits or bits per second, with precision that suits the magnitude."""
    if value is None:
        return "n/a"
    v = float(value)
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return _num(v, 0)
    if abs(v) >= 10:
        return _num(v, 1)
    return _num(v, 2)


def _pct(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.{digits}f} %"


def _h(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _table(head: Sequence[str], rows: Iterable[Sequence[str]], align: str = "") -> str:
    """A Markdown table; ``align`` has one of ``l``/``r`` per column (default: first left)."""
    align = align or "l" + "r" * (len(head) - 1)
    rule = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _picture(name: str, alt: str, media_rel: Path) -> str:
    """A chart as a plain ``<img>`` (dark theme) plus a link to the light version.

    A ``<picture>`` element with a ``prefers-color-scheme`` source would pick
    the theme automatically on github.com, but the GitHub mobile app does not
    render ``<picture>`` at all. A plain image shows everywhere; the dark
    variant reads well on light and dark pages alike.
    """
    base = media_rel.as_posix().rstrip("/")
    return (
        f'<img alt="{alt}" src="{base}/{name}-dark.png" width="100%">\n\n'
        f"<sub>Light version: [{name}-light.png]({base}/{name}-light.png)</sub>"
    )


def describe_source(source: Mapping[str, Any], frames: int | None = None) -> str:
    """``simulated aquarium, seed 1, 640x480, 30 fps, 6 fish, 9,000 frames`` (or the video's)."""
    if source.get("kind") == "synthetic":
        parts = [
            "simulated aquarium",
            f"seed {source.get('seed')}",
            f"{source.get('width')}x{source.get('height')}",
            f"{_num(source.get('fps'), 0)} fps",
            f"{source.get('fish')} fish",
        ]
    elif source.get("kind") == "video":
        parts = [
            str(source.get("name")),
            f"{source.get('width')}x{source.get('height')}",
            f"{_num(source.get('fps'), 0)} fps",
        ]
    else:
        parts = [str(source.get("name", "unknown source"))]
    if frames is not None:
        parts.append(f"{_num(frames)} frames")
    return ", ".join(parts)


def _run_label(results: Mapping[str, Any], label: str) -> str:
    run = results["runs"][label]
    return f"{label} ({describe_source(run['source'], run['frames'])})"


def _stream_label(key: str) -> str:
    label, _, name = key.partition("/")
    return f"{STREAM_LABELS.get(name, name)}, {label}"


def render_markdown(
    results: Mapping[str, Any],
    *,
    charts: Mapping[str, str] | None = None,
    media_rel: Path = Path("../media"),
) -> str:
    """``RESULTS.md`` from a JSON-ready results mapping (as written to ``results.json``)."""
    charts = charts or {}
    parts = [
        _md_header(results),
        _md_summary(results),
        _md_runs(results),
        _md_min_entropy(results, charts, media_rel),
        _md_credit(results, charts, media_rel),
        _md_streams(results, charts, media_rel),
        _md_ent(results, charts, media_rel),
        _md_liveness(results, charts, media_rel),
        _md_tracking(results),
        _md_dieharder(results),
        _md_limits(results),
        _md_reproduce(results),
    ]
    return "\n\n".join(p.strip("\n") for p in parts if p) + "\n"


def _first_synthetic(results: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for run in results["runs"].values():
        if run["source"].get("kind") == "synthetic":
            return dict(run["source"])
    return None


def _md_header(results: Mapping[str, Any]) -> str:
    prov = results.get("provenance", {})
    commit = prov.get("commit", "unknown")
    modified = prov.get("modified_packages") or []
    code = f"`{commit}`" + (
        f" with uncommitted changes in {', '.join(modified)}" if modified else ""
    )
    date = str(results.get("generated_utc", ""))[:10]
    lines = [
        "# Results",
        "",
        "Statistical analysis of the puffer-cipher entropy channels, produced by "
        f"`puffer analyze` (`puffer.analysis.report`) on {date} from code {code}. "
        "Every figure is also in [`results.json`](results.json).",
        "",
        "> **Important.**",
    ]
    runs = list(results["runs"].values())
    src = _first_synthetic(results)
    if src is not None and all(r["source"].get("kind") == "synthetic" for r in runs):
        lines.append(
            "> **All data in this report comes from the simulated aquarium** "
            f"(`SyntheticAquarium`, seed {results.get('seed')}, {src.get('width')}x"
            f"{src.get('height')} pixels, {_num(src.get('fps'))} fps, {src.get('fish')} fish). "
            "No real fish and no real camera were involved."
        )
    else:
        listed = "; ".join(
            f"`{label}`: {describe_source(r['source'], r['frames'])}"
            for label, r in results["runs"].items()
        )
        lines.append(f"> **Data sources:** {listed}.")
    if src is not None:
        lines.append(
            "> The simulator draws its sensor noise and fish behaviour from a seeded "
            "pseudo-random generator, so to anyone who knows the seed its samples carry no "
            "entropy at all. The estimates below show how unpredictable the samples look to "
            "the SP 800-90B estimators, which is what the same code would measure on a real "
            "camera. They are not evidence that this run produced secret bits."
        )
    return "\n".join(lines)


def _assessed(results: Mapping[str, Any], label: str, channel: str) -> Any:
    return results["sp800_90b"].get(label, {}).get(channel, {}).get("min_entropy")


def _md_summary(results: Mapping[str, Any]) -> str:
    runs = results["runs"]
    labels = list(runs)
    head = ["", *(f"{CHANNEL_LABELS[c]}, {label}" for c in CAMERA_CHANNELS for label in labels)]
    cells = [(c, label) for c in CAMERA_CHANNELS for label in labels]
    rows = [
        ["Min-entropy, SP 800-90B (bits per sample)"]
        + [_h(_assessed(results, label, c)) for c, label in cells],
        ["Samples assessed (SP 800-90B asks for 1,000,000)"]
        + [_num(results["sp800_90b"][label][c].get("n_used")) for c, label in cells],
        ["Credited by the engine (bit/s, after start-up)"]
        + [_bits(runs[label]["overall"][c].get("steady_bits_per_s")) for c, label in cells],
    ]
    out = ["## Summary", "", _table(head, rows), ""]
    bullets = []
    live = results.get("liveness")
    if live:
        lv, rp = live["live"], live["replay"]
        bullets.append(
            f"**Liveness.** The live tank was VERIFIED after {_num(lv['time_to_verified_s'], 1)} s "
            f"(confidence {_num(lv['max_confidence_bits'], 1)} bits at best). A bit-exact replay "
            f"of the same footage peaked at {_num(rp['max_confidence_bits'], 1)} bits against the "
            f"{_num(live['threshold_bits'])}-bit threshold, never verified and earned "
            f"{_bits(rp['camera_credited_bits'])} camera bits."
        )
    final = next((k for k in results["sp800_22"] if k.endswith("/pufferrng")), None)
    if final is not None:
        nist = results["sp800_22"][final]
        if nist.get("mode") == "multi":
            inside = nist["subtest_count"] - len(nist["flagged"])
            names = [f"`{n}`" for n in nist["flagged"][:3]]
            if len(nist["flagged"]) > 3:
                names.append(f"{len(nist['flagged']) - 3} more")
            flagged = f" (outside: {', '.join(names)})" if names else ""
            bullets.append(
                f"**PufferRNG output, SP 800-22:** {nist['sequences']} sequences of "
                f"{_num(nist['n_bits'])} bits, {inside} of {nist['subtest_count']} sub-tests "
                f"inside the acceptance band{flagged}; a perfect generator leaves "
                f"{nist['expected_flagged']:.1f} outside on average. Expected for keyed BLAKE2b "
                "and ChaCha20 output, and not evidence of entropy."
            )
    dh = results.get("dieharder") or {}
    if dh.get("ran"):
        c = dh["counts"]
        bullets.append(
            f"**dieharder** ({len(dh['results'])} results over {len(dh['tests'])} tests): "
            f"{c['PASSED']} PASSED, {c['WEAK']} WEAK, {c['FAILED']} FAILED."
        )
    raw = results["ent"]["raw"]
    first = labels[0]
    sensor_raw = raw.get(f"{first}/{CHANNEL_SENSOR}")
    if sensor_raw:
        bullets.append(
            f"**Raw samples are not uniform, and should not be.** Raw sensor bytes ({first}) have "
            f"{sensor_raw['entropy']:.2f} bits of Shannon entropy per byte and fail ent's "
            "chi-square test; only their min-entropy, measured above, says what they are worth."
        )
    out += [f"* {b}" for b in bullets]
    return "\n".join(out)


def _md_runs(results: Mapping[str, Any]) -> str:
    rows = []
    for label, run in results["runs"].items():
        src = run["source"]
        rows.append(
            [
                f"`{label}`",
                src.get("name", ""),
                str(src.get("seed", "")),
                f"{src.get('width')}x{src.get('height')}",
                _num(run["frames"]),
                f"{_num(run['seconds'], 1)} s",
                f"{_num(run['engine_fps'], 1)} fps",
            ]
        )
    extra = []
    if results.get("cycle"):
        cyc = results["cycle"]
        src = cyc["source"]
        extra.append(
            [
                "`cycle`",
                f"{src.get('name')}, {src.get('cycle_day_s'):g} s day / "
                f"{src.get('cycle_night_s'):g} s night",
                str(src.get("seed", "")),
                f"{src.get('width')}x{src.get('height')}",
                _num(cyc["frames"]),
                f"{_num(cyc['seconds'], 1)} s",
                "",
            ]
        )
    if results.get("liveness"):
        live = results["liveness"]
        src = live["source"]
        for name in ("live", "replay", "control"):
            item = live.get(name)
            if not item:
                continue
            what = {
                "live": "simulated aquarium, LED shows challenge A",
                "replay": "FFV1 recording of `live`, fresh challenge",
                "control": "FFV1 recording of `live`, challenge A",
            }[name]
            extra.append(
                [
                    f"`{name}`",
                    what,
                    str(src.get("seed", "")),
                    f"{src.get('width')}x{src.get('height')}",
                    _num(item["frames"]),
                    f"{_num(item['seconds'], 1)} s",
                    "",
                ]
            )
    head = ["run", "source", "seed", "resolution", "frames", "simulated time", "frames/s here"]
    eng = results.get("engine", {})
    text = [
        "## 1. Data",
        "",
        "Each run is one `EntropyEngine` built by `puffer.runtime.build_engine`, stepped frame "
        "by frame; an output listener records every channel output, the credit the engine gave "
        "it and the channel's health state. Analysis runs use `liveness=off` and no LED; the "
        "liveness runs use the simulated LED and `liveness=required`. Engine settings are the "
        f"defaults: {_num(eng.get('sensor_sites'))} sensor sites, safety factor "
        f"{eng.get('safety_factor')}, credit caps of {eng.get('h_cap', {}).get('sensor')} "
        f"(sensor) and {eng.get('h_cap', {}).get('fish')} (fish) bits per sample. Simulated time "
        'runs at the source frame rate. "Frames/s here" is how fast the engine got through '
        f"the run during this report, with up to {results.get('workers', 1)} runs "
        "in parallel on the machine below: a record of this report, not a benchmark.",
        "",
        _table(head, rows + extra, "lllrrrr"),
    ]
    prov = results.get("provenance", {})
    text += [
        "",
        f"Machine: {prov.get('system')} {prov.get('machine')}, {prov.get('cpus')} CPUs; "
        f"Python {prov.get('python')}, NumPy {prov.get('numpy')}, SciPy {prov.get('scipy')}, "
        f"OpenCV {prov.get('opencv')}, PyNaCl {prov.get('pynacl')}.",
    ]
    return "\n".join(text)


def _estimator_rows(assessments: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for track, scale, suffix in (("original", 1, ""), ("bitstring", 8, " (bits x 8)")):
        names = [n for n in ESTIMATOR_LABELS if any(_find(a, n, track) for a in assessments)]
        for name in names:
            cells = []
            for a in assessments:
                result = _find(a, name, track)
                cells.append(_h(result["min_entropy"] * scale) if result else "n/a")
            label = f"{ESTIMATOR_LABELS[name]} ({ESTIMATOR_SECTIONS[name]}){suffix}"
            rows.append([label, *cells])
    return rows


def _find(assessment: Mapping[str, Any], name: str, track: str) -> Mapping[str, Any] | None:
    for result in assessment.get("results", []):
        if result["name"] == name and result["details"].get("track") == track:
            return dict(result)
    return None


def _md_min_entropy(results: Mapping[str, Any], charts: Mapping[str, str], media: Path) -> str:
    runs = list(results["runs"])
    keys = [(c, label) for c in CAMERA_CHANNELS for label in runs]
    assessments = [results["sp800_90b"][label][c] for c, label in keys]
    head = ["estimator", *(f"{CHANNEL_LABELS[c]}, {label}" for c, label in keys)]
    rows = _estimator_rows(assessments)
    rows.append(["**H, original track** (min)", *(_h(a.get("h_original")) for a in assessments)])
    rows.append(
        [
            "**H, bitstring track** (min, x 8)",
            *(
                _h(8 * a["h_bitstring"]) if a.get("h_bitstring") is not None else "n/a"
                for a in assessments
            ),
        ]
    )
    rows.append(
        ["**Assessed H** = min of both", *(f"**{_h(a.get('min_entropy'))}**" for a in assessments)]
    )
    rows.append(["samples assessed", *(_num(a.get("n_used")) for a in assessments)])
    rows.append(
        [
            "share of the 1,000,000 SP 800-90B asks for",
            *(_pct(a.get("fraction_of_standard")) for a in assessments),
        ]
    )
    rows.append(["bits on the bitstring track", *(_num(a.get("n_bitstring")) for a in assessments)])
    sources = "; ".join(_run_label(results, label) for label in runs)
    text = [
        "## 2. Raw min-entropy (SP 800-90B)",
        "",
        "The samples each channel credits, assessed with all estimators of SP 800-90B "
        "section 6.3 on the non-IID track (section 3.1.3, 8 bits per sample). The sensor "
        "channel's samples are temporal differences `(I_t - I_t-1) mod 256` of the green "
        "channel at fixed random pixel sites; the fish channel's are the low 4 bits of each "
        "axis of a confirmed track's Kalman innovation at 1/8-pixel resolution. Values are bits "
        "per sample; the bitstring track (each sample written as 8 bits, at most the first "
        "1,000,000 bits) is scaled by 8 to the same unit. The assessed entropy is the smallest "
        "number in each column.",
        "",
        f"Data: {sources}.",
        "",
        _table(head, rows),
        "",
        "Sample counts, stated plainly: the sensor channel yields 4,096 samples per frame, so "
        "the full 1,000,000 samples come from about 245 consecutive frames. The fish channel "
        "yields one sample per confirmed fish per frame; the counts above are everything the "
        "runs produced, a small fraction of what the standard asks for, so the fish estimates "
        "carry wide confidence bounds (all estimators use 99 % upper bounds, which makes them "
        "lower, not higher, on short data).",
    ]
    if "min-entropy" in charts:
        text += ["", _picture("min-entropy", charts["min-entropy"], media)]
    text += ["", _md_credit_check(results)]
    return "\n".join(text)


def _md_credit_check(results: Mapping[str, Any]) -> str:
    """Compare what the engine credits per sample with the full assessment."""
    lines = [
        "**Credit versus assessment.** The engine credits "
        "`n * min(H_est, cap) * safety_factor`, where `H_est` is a quick rolling estimate "
        "(minimum of MCV, Lag and MultiMMC on the last 4,096 samples, original track only). "
        "The table compares the credit per sample the engine actually gave with the full "
        "assessment above.",
        "",
    ]
    rows = []
    for label, run in results["runs"].items():
        for c in CAMERA_CHANNELS:
            stats = run["overall"][c]
            assessed = _assessed(results, label, c)
            given = stats.get("credited_per_sample")
            ratio = given / assessed if assessed and given is not None else None
            rows.append(
                [
                    f"{CHANNEL_LABELS[c]}, {label}",
                    _h(stats.get("h_engine_mean")),
                    _h(given),
                    _h(assessed),
                    "n/a" if ratio is None else f"{ratio:.2f}",
                ]
            )
    head = [
        "channel",
        "engine H_est, mean",
        "credited per sample",
        "assessed H",
        "credited / assessed",
    ]
    lines.append(_table(head, rows))
    over = [r[0] for r in rows if r[-1] != "n/a" and float(r[-1]) > 1.0]
    lines.append("")
    if over:
        lines.append(
            "A ratio above 1 means the engine credited more per sample than the full "
            f"SP 800-90B assessment supports: {', '.join(over)}. The quick estimate does not "
            "run the bitstring track, which is where the assessment found the most structure. "
            "Outputs stay as strong as `os.urandom` regardless (DESIGN.md rule 2); the receipt's "
            "credit figure is what is overstated."
        )
    else:
        lines.append(
            "Every ratio is at most 1: the engine never credited more per sample than the full "
            "assessment supports."
        )
    return "\n".join(lines)


def _md_credit(results: Mapping[str, Any], charts: Mapping[str, str], media: Path) -> str:
    runs = results["runs"]
    labels = list(runs)
    keys = [(c, label) for c in CAMERA_CHANNELS for label in labels]
    head = ["", *(f"{CHANNEL_LABELS[c]}, {label}" for c, label in keys)]

    def row(name: str, fn: Callable[[Mapping[str, Any]], str]) -> list[str]:
        return [name, *(fn(runs[label]["overall"][c]) for c, label in keys)]

    rows = [
        row("samples per frame", lambda s: _num(s["samples_per_frame"], 1)),
        row("SP 800-90B H (bits per sample)", lambda s: _h(s.get("h_assessed"))),
        row("engine H_est, mean (bits per sample)", lambda s: _h(s.get("h_engine_mean"))),
        row("credit cap (bits per sample)", lambda s: _num(s["h_cap"], 1)),
        row("credited per sample (bits)", lambda s: _h(s["credited_per_sample"])),
        row("credited, whole run (bit/s)", lambda s: _bits(s["bits_per_s"])),
        row("credited after start-up (bit/s)", lambda s: _bits(s["steady_bits_per_s"])),
        row("first credit after (s)", lambda s: _num(s.get("first_credit_s"), 1)),
        row("time OK", lambda s: _pct(s["health"]["ok"])),
        row("time DEGRADED", lambda s: _pct(s["health"]["degraded"])),
        row(
            "time STARTUP or FAILED", lambda s: _pct(s["health"]["startup"] + s["health"]["failed"])
        ),
    ]
    tracks = ["confirmed fish tracks per frame"]
    for c, label in keys:
        tracks.append(
            _num(runs[label]["overall"]["tracks_per_frame"], 2) if c == CHANNEL_FISH else ""
        )
    rows.append(tracks)
    text = [
        "## 3. Credit by day and by night",
        "",
        "What the engine credited, from its own per-output credit (not re-computed). "
        '"After start-up" starts at the channel\'s first credited output (the first 1,024 '
        "samples must pass the health tests first). The system channel adds 256 bits of "
        "`os.urandom` to every pool draw on top of these, whatever the tank does.",
        "",
        _table(head, rows),
    ]
    cyc = results.get("cycle")
    if cyc:
        src = cyc["source"]
        text += [
            "",
            f"### Day, night, day ({describe_source(src, cyc['frames'])})",
            "",
            f"The simulator in `mode=cycle`: {src.get('cycle_day_s'):g} s of day (the last "
            f"{min(src.get('transition_s', 8.0), src.get('cycle_day_s', 0.0) / 2):g} s dim like "
            f"dusk), {src.get('cycle_night_s'):g} s "
            "of night with the camera in IR mode (monochrome, dimmer, noisier: read noise "
            f"sigma {src.get('read_noise_night')} instead of {src.get('read_noise_day')}), then "
            "day again. The fish sleep at night: fewer confirmed tracks, smaller innovations.",
            "",
        ]
        prow = []
        for phase in cyc["phases"]:
            prow.append(
                [
                    phase["phase"],
                    f"{phase['start_s']:g} to {phase['end_s']:.0f} s",
                    _bits(phase[CHANNEL_SENSOR]["bits_per_s"]),
                    _bits(phase[CHANNEL_FISH]["bits_per_s"]),
                    _num(phase["tracks_per_frame"], 2),
                    _pct(phase[CHANNEL_FISH]["health"]["ok"]),
                ]
            )
        text.append(
            _table(
                [
                    "phase",
                    "time",
                    "sensor (bit/s)",
                    "fish (bit/s)",
                    "fish tracks per frame",
                    "fish channel OK",
                ],
                prow,
            )
        )
        text += ["", _cycle_story(cyc)]
        if "day-night-cycle" in charts:
            text += ["", _picture("day-night-cycle", charts["day-night-cycle"], media)]
    return "\n".join(text)


def _cycle_story(cyc: Mapping[str, Any]) -> str:
    """Plain-language reading of the cycle phases, from the numbers."""
    phases = {p["phase"]: p for p in cyc["phases"]}
    day, night = phases.get("day"), phases.get("night")
    late = phases.get("night, second half")
    if day is None or night is None:
        return ""
    sensor_day = day[CHANNEL_SENSOR]["bits_per_s"]
    sensor_night = night[CHANNEL_SENSOR]["bits_per_s"]
    fish_day = day[CHANNEL_FISH]["bits_per_s"]
    fish_late = (late or night)[CHANNEL_FISH]["bits_per_s"]
    when = "in the second half of the night" if late else "at night"
    tracks_day = _num(day["tracks_per_frame"], 1)
    tracks_late = _num((late or night)["tracks_per_frame"], 1)
    if fish_late < 0.9 * fish_day:
        fish = (
            f"Fish credit falls from {_bits(fish_day)} bit/s by day to {_bits(fish_late)} bit/s "
            f"{when}, as the sleeping fish move less and fewer of them stay tracked "
            f"({tracks_day} confirmed tracks per frame by day, {tracks_late} {when}); the drop "
            "is gradual because the fish settle over tens of seconds."
        )
    else:
        fish = (
            f"Fish credit is {_bits(fish_day)} bit/s by day and {_bits(fish_late)} bit/s {when} "
            f"({tracks_day} and {tracks_late} confirmed tracks per frame)."
        )
    verb = "rises" if sensor_night > sensor_day else "does not rise"
    sentences = [
        fish,
        f"The sensor channel earns {_bits(sensor_day)} bit/s by day and {_bits(sensor_night)} "
        f"bit/s at night: its credit {verb} with the IR mode's higher read noise. The two "
        "channels do not interact; each is credited from its own samples.",
        "For one second after each switch the fish channel credits nothing: the motion "
        "detector rebuilds its background model when the image changes between colour and IR.",
        "All of this describes the simulator's camera and fish models; a real NoIR camera and "
        "real fish need their own measurement.",
    ]
    return " ".join(sentences)


def _md_streams(results: Mapping[str, Any], charts: Mapping[str, str], media: Path) -> str:
    nist = results["sp800_22"]
    meta = results["streams"]
    rows = []
    for key, info in meta.items():
        res = nist[key]
        if res.get("mode") == "multi":
            verdict = (
                f"{len(res['flagged'])} of {res['subtest_count']} sub-tests outside "
                f"(chance: {res['expected_flagged']:.1f}, "
                f"P = {_chance_tail(len(res['flagged']), res['expected_flagged']):.2f})"
            )
            tests = f"{len(res['per_test'])} of 15"
        elif res.get("mode") == "single":
            ok = res["p_value_count"] - res["failed_count"]
            verdict = f"{ok} of {res['p_value_count']} p-values >= 0.01"
            tests = f"{res['applicable']} of 15"
        else:
            verdict, tests = "not tested", "0 of 15"
        h = info.get("h_per_sample")
        rows.append(
            [
                _stream_label(key),
                "yes" if key.endswith("/pufferrng") else "no",
                _num(info.get("samples_in")) if "samples_in" in info else "",
                _h(h) if h is not None else "",
                _num(info["bits"]),
                f"{res.get('sequences', 0)} x {_num(res.get('n_bits'))}",
                tests,
                verdict,
            ]
        )
    head = [
        "stream",
        "os.urandom",
        "raw samples in",
        "credited H per sample",
        "output bits",
        "sequences x bits",
        "tests run",
        "SP 800-22",
    ]
    text = [
        "## 4. Conditioned streams and PufferRNG output (SP 800-22)",
        "",
        "Conditioned streams use **no** operating-system randomness "
        "(`puffer.entropy.ConditionedStream`, analysis only): a block of raw samples is "
        "compressed to 256 bits with unkeyed BLAKE2b once its credited min-entropy reaches "
        "512 bits. Credited min-entropy per sample is the engine's rule applied to the "
        "assessment of section 2, `min(H, cap) * 0.5`, so each output bit is backed by at least "
        "four bits of assessed min-entropy. The combined stream interleaves both channels frame by "
        "frame at their sample-weighted rate. PufferRNG output is the production path "
        "(keyed with `os.urandom`), requested 1 MiB per `generate` call during the first run.",
        "",
        _table(head, rows, "llrrrrrl"),
        "",
        "Multi-sequence verdicts follow SP 800-22 section 4.2: for every sub-test the "
        "proportion of sequences with p >= 0.01 must lie inside `0.99 +- 3 sqrt(0.99 x 0.01 / m)`, "
        f"and from {MIN_SEQUENCES_FOR_UNIFORMITY} sequences on the p-values must also be "
        "uniform (P >= 0.0001). Even a perfect generator puts some sub-tests outside; the "
        '"chance" figure is how many it would on average with that many sequences, so the '
        "observed count is judged against it, not against zero; P is the chance of at least "
        "the observed count (Poisson approximation; the excursion states of one sequence are "
        "correlated, so the true spread is somewhat wider). With few sequences the band "
        "is wide and the test is weak; each stream gets as many full sequences as its data "
        "provided, up to the configured maximum.",
    ]
    multi = [k for k in meta if nist[k].get("mode") == "multi"]
    if multi:
        head2 = ["test", *(_stream_label(k) for k in multi)]
        rows2 = []
        for test in SUITE_TESTS:
            cells = []
            for k in multi:
                pt = nist[k]["per_test"].get(test)
                if pt is None:
                    cells.append("skipped")
                    continue
                flag = "" if pt["flagged"] == 0 else f" ({pt['flagged']} flagged)"
                cells.append(f"{pt['min_proportion']:.3f}{flag}")
            rows2.append([TEST_LABELS[test], *cells])
        counts = ", ".join(sorted({str(nist[k]["sequences"]) for k in multi}, key=int))
        rows2.append(
            [
                f"band lower edge at {counts} sequences",
                *(f"{_band_edge(nist[k]):.3f}" for k in multi),
            ]
        )
        text += [
            "",
            "Lowest pass proportion over each test's sub-tests (148 templates, 8 + 18 "
            "excursion states, 2 cumulative sums and 2 serial sub-tests). The random excursion "
            "tests only apply to sequences whose random walk returns to zero at least 500 times, "
            "so they run on fewer sequences, their band is wider, and one unusual sequence moves "
            "all of its 8 or 18 states together:",
            "",
            _table(head2, rows2),
        ]
        if "sp800-22-proportions" in charts:
            text += ["", _picture("sp800-22-proportions", charts["sp800-22-proportions"], media)]
        final = next((k for k in multi if k.endswith("/pufferrng")), None)
        if final is not None and "sp800-22-pvalues" in charts:
            text += ["", _picture("sp800-22-pvalues", charts["sp800-22-pvalues"], media)]
    short = [k for k in meta if nist[k].get("mode") in ("single", "none")]
    for key in short:
        res = nist[key]
        text += [
            "",
            f"**{_stream_label(key)}: {_num(res.get('n_bits'))} bits, shorter than one "
            f"{_num(results['config']['nist_bits'])}-bit sequence.** "
            + (
                "Only the tests whose recommended input size fits ran, once each:"
                if res.get("mode") == "single"
                else "Too short for any test."
            ),
        ]
        if res.get("mode") == "single":
            rows3 = []
            for t in res["tests"]:
                if t["skipped"]:
                    rows3.append([TEST_LABELS[t["name"]], "skipped", t["skipped"]])
                else:
                    p = t["min_p"]
                    sub = f" (lowest of {len(t['p_values'])})" if len(t["p_values"]) > 1 else ""
                    rows3.append(
                        [
                            TEST_LABELS[t["name"]],
                            "pass" if t["passed"] else "p < 0.01",
                            f"p = {p:.4f}{sub}",
                        ]
                    )
            text += ["", _table(["test", "result", "p-value or reason"], rows3, "lll")]
    skipped_multi = {k: nist[k]["skipped"] for k in multi if nist[k].get("skipped")}
    for key, skipped in skipped_multi.items():
        text += [
            "",
            f"{_stream_label(key)}: not applicable on these sequences: "
            + "; ".join(f"{TEST_LABELS[n]} ({why})" for n, why in skipped.items())
            + ".",
        ]
    return "\n".join(text)


def _md_ent(results: Mapping[str, Any], charts: Mapping[str, str], media: Path) -> str:
    ent = results["ent"]
    rows = []

    def add(name: str, stats: Mapping[str, Any] | None) -> None:
        if not stats:
            return
        serial = stats.get("serial_correlation")
        monte = stats.get("monte_carlo_error_percent")
        rows.append(
            [
                name,
                _num(stats["size"]),
                f"{stats['entropy']:.4f}",
                _num(stats["chi_square"], 1),
                _chi_p(stats["chi_square_percent"]),
                f"{stats['mean']:.3f}",
                "n/a" if monte is None else f"{monte:.3f} %",
                "n/a" if serial is None else f"{serial:+.5f}",
            ]
        )

    for key, stats in ent["raw"].items():
        label, _, channel = key.partition("/")
        add(f"raw {channel} samples, {label}", stats)
    for key, stats in ent["streams"].items():
        add(_stream_label(key), stats)
    head = [
        "stream",
        "bytes",
        "entropy (bits/byte)",
        "chi-square",
        "chi-square exceeded",
        "mean",
        "Monte Carlo pi error",
        "serial correlation",
    ]
    text = [
        "## 5. ent: raw samples versus conditioned output",
        "",
        "The in-process equivalent of John Walker's `ent` (`puffer.analysis.ent`). A random "
        "byte stream has 8 bits per byte, mean 127.5, a chi-square value that a random "
        "stream would exceed between 1 % and 99 % of the time, pi within a fraction of a "
        "percent and serial correlation near 0.",
        "",
        _table(head, rows),
        "",
        "Raw samples fail, as they should: sensor differences cluster around 0 (and 255) and "
        "fish innovations around small values, so their byte histogram is far from flat. "
        "Every hashed stream passes. This is the point of DESIGN.md rule 4: hashing makes any "
        "input look random, so these statistics cannot tell good entropy from none. The "
        "min-entropy of the raw samples (section 2) is the measurement that matters.",
    ]
    binary = ent.get("binary") or {}
    if binary.get("available") and binary.get("results"):
        rows_b = []
        for name, stats in binary["results"].items():
            if "error" in stats:
                rows_b.append([name, "error", stats["error"], ""])
            else:
                rows_b.append(
                    [
                        name,
                        f"{stats['entropy']:.6f}",
                        _num(stats["chi_square"], 2),
                        _chi_p(stats["chi_square_percent"]),
                    ]
                )
        text += [
            "",
            "Cross-check with the real `ent` binary on two of the streams:",
            "",
            _table(["stream", "entropy (bits/byte)", "chi-square", "chi-square exceeded"], rows_b),
        ]
    if "ent-raw-vs-conditioned" in charts:
        text += ["", _picture("ent-raw-vs-conditioned", charts["ent-raw-vs-conditioned"], media)]
    return "\n".join(text)


def _chi_p(percent: float) -> str:
    if percent < 0.01:
        return "< 0.01 %"
    if percent > 99.99:
        return "> 99.99 %"
    return f"{percent:.2f} %"


def _md_liveness(results: Mapping[str, Any], charts: Mapping[str, str], media: Path) -> str:
    live = results.get("liveness")
    if not live:
        reason = results.get("skipped_sections", {}).get("liveness", "not run")
        return f"## 6. Liveness: live tank versus replay\n\nNot run: {reason}."
    rows = []
    names = {
        "live": ("live simulator", "A (seeded)"),
        "replay": ("FFV1 replay of `live`", "fresh (os.urandom)"),
        "control": ("FFV1 replay of `live`", "A (seeded)"),
    }
    for key in ("live", "replay", "control"):
        item = live.get(key)
        if not item:
            continue
        what, challenge = names[key]
        rows.append(
            [
                f"`{key}`",
                what,
                challenge,
                "never"
                if item["time_to_verified_s"] is None
                else f"{item['time_to_verified_s']:.2f} s",
                _num(item["max_confidence_bits"], 1),
                f"{item['final_match_ratio']:.3f}",
                str(item["final_state"]).upper(),
                _bits(item["camera_credited_bits"]),
            ]
        )
    head = [
        "run",
        "video",
        "LED challenge",
        "VERIFIED after",
        "max confidence (bits)",
        "final match ratio",
        "final state",
        "camera credit (bits)",
    ]
    rec = live["recording"]
    lv = live["live"]
    text = [
        "## 6. Liveness: live tank versus replay",
        "",
        f"Data: {describe_source(live['source'], lv['frames'])}. The simulated LED shows "
        f"challenge A (seeded with {live['schedule_seed']}, {live['symbol_s']} s per "
        "Manchester-coded bit) and the live run is recorded, frame for frame, to a lossless "
        f"{rec['codec']} file ({_num(rec['bytes'] / 1e6, 0)} MB). The replay feeds that file "
        "to a new engine with `liveness=required` and a fresh challenge from `os.urandom`; "
        "the verifier is even told where the LED is, the most favourable setup for an attacker. "
        "The control replays the same file against challenge A.",
        "",
        _table(head, rows, "llllrrlr"),
        "",
        f"`confidence_bits = -log2(P[Binomial(n, 1/2) >= matches] x n_lags)` over the last "
        f"{live['window_bits']} bits; VERIFIED needs at least {_num(live['threshold_bits'])} "
        f"bits and a match ratio of at least {live['min_match']}. The live run can verify "
        f"no earlier than one full window ({live['window_bits'] * live['symbol_s']:.1f} s) "
        "after the start, and its first camera credit follows the 5 s health cooldown. The "
        f"replay's best judged window matched {_pct(live['replay']['max_match_ratio'], 0)} of the "
        "bits (bits the LED detector cannot decode count as misses), well short of the 90 % "
        "needed: a recording made before the challenge existed cannot contain it. The "
        "control passing shows that the replay fails because the challenge is fresh, not "
        "because of the codec or the detector.",
    ]
    if "liveness-replay" in charts:
        text += ["", _picture("liveness-replay", charts["liveness-replay"], media)]
    return "\n".join(text)


def _md_tracking(results: Mapping[str, Any]) -> str:
    tracking = results.get("tracking")
    if not tracking:
        reason = results.get("skipped_sections", {}).get("tracking", "not run")
        return f"## 7. Fish tracking quality\n\nNot run: {reason}."
    figures = [
        ("detection precision", "detection_precision", _pct),
        ("detection recall", "detection_recall", _pct),
        ("fish-track precision", "fish_track_precision", _pct),
        ("fish recall", "fish_recall", _pct),
        ("observed recall (fed the entropy channel)", "observed_recall", _pct),
        ("contamination (observations not on a fish)", "contamination", lambda v: _pct(v, 2)),
        ("track ID switches per minute", "id_switches_per_minute", lambda v: _num(v, 1)),
        ("fish-ID switches per minute", "fish_id_switches_per_minute", lambda v: _num(v, 1)),
        ("identity purity", "identity_purity", _pct),
        ("identity coverage", "identity_coverage", _pct),
        ("mean centre error (px)", "mean_center_error_px", lambda v: _num(v, 1)),
        ("ground-truth fish per frame", "fish_per_frame", lambda v: _num(v, 2)),
        ("tracker speed during this report (frames/s)", "process_fps", lambda v: _num(v, 0)),
    ]
    labels = list(tracking)
    rows = [
        [name, *(fmt(tracking[m]["metrics"].get(key)) for m in labels)]
        for name, key, fmt in figures
    ]
    first = tracking[labels[0]]
    head = ["", *labels]
    return "\n".join(
        [
            "## 7. Fish tracking quality",
            "",
            "Every fish observation is credited as entropy and attributed to a named fish, so a "
            "swaying plant or a bubble tracked as a fish would make both claims false. "
            "`puffer.vision.evaluate.evaluate_tracking` scores the default tracker (motion "
            "detector, ByteTrack, re-identification) against the simulator's ground truth: "
            f"{describe_source(first['source'])}, {_num(first['frames'])} frames per mode, the "
            f"first {first['warmup']} not scored, Hungarian matching at IoU >= 0.3. Contamination "
            "counts observations whose box lies mostly off every real fish.",
            "",
            _table(head, rows),
            "",
            "At night the IR image has no colour, so appearance re-identification has little to "
            "work with and the sleeping fish move little; `scripts/eval_tracking.py` runs the "
            "same evaluation over more seeds and scenarios.",
        ]
    )


def _md_dieharder(results: Mapping[str, Any]) -> str:
    dh = results.get("dieharder") or {}
    if not dh.get("ran"):
        return f"## 8. dieharder\n\nNot run: {dh.get('reason', 'not requested')}."
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in dh["results"]:
        groups.setdefault(row["test_name"], []).append(row)
    rows = []
    for name, items in groups.items():
        counts = {k: sum(r["assessment"] == k for r in items) for k in ("PASSED", "WEAK", "FAILED")}
        ntups = sorted({r["ntup"] for r in items})
        rows.append(
            [
                f"`{name}`",
                ", ".join(str(n) for n in ntups),
                str(len(items)),
                str(counts["PASSED"]),
                str(counts["WEAK"]),
                str(counts["FAILED"]),
                f"{min(r['p_value'] for r in items):.4f}",
            ]
        )
    c = dh["counts"]
    selection = list(dh.get("selection") or [])
    if selection == ["all"]:
        chosen = "the full battery (`-a`)."
    else:
        chosen = "these tests, one dieharder process each: " + ", ".join(
            f"`{' '.join(_dieharder_args(s))}`" for s in selection
        )
        if selection == list(DIEHARDER_SELECTION):
            chosen += (
                ". That is every test and ntuple of `dieharder -a` with two cuts: `diehard_sums` "
                '(`-d 14`) is left out because dieharder itself rates it "Do Not Use", and the '
                "RGB lagged sum runs lags 0 to 4, 8, 16 and 32 instead of all 33 (lag n reads "
                "n + 1 times the data; the full sweep alone takes about half an hour, the whole "
                "`-a` battery about 40 minutes on this machine)."
            )
        else:
            chosen += "."
    weak_note = (
        " WEAK means a p-value in the outer 0.5 % on either side; with this many results a "
        "few are expected from a perfect generator (dieharder's documentation says the same)."
        if c["WEAK"]
        else ""
    )
    return "\n".join(
        [
            "## 8. dieharder",
            "",
            f"{dh.get('version') or 'dieharder'}, raw input from a pipe (`-g 200`), so nothing is "
            "ever rewound. The pipe carries PufferRNG output, 1 MiB per `generate` call (one pool "
            "draw keyed with fresh `os.urandom` each), while an engine on "
            f"`{dh.get('spec')}` keeps absorbing frames in the background, exactly like "
            "`puffer random --stream | dieharder -g 200`. "
            f"Ran {chosen} Time: {_num(dh['elapsed_s'] / 60, 1)} min of a "
            f"{_num(dh['budget_s'] / 60, 0)} min budget"
            + (" (budget exhausted, later tests not run)" if dh.get("timed_out") else "")
            + f"; {_num(dh['bytes_fed'] / 2**30, 1)} GiB fed in {_num(dh['generate_calls'])} "
            f"calls; the engine processed {_num(dh.get('engine_frames'))} frames meanwhile.",
            "",
            f"**{c['PASSED']} PASSED, {c['WEAK']} WEAK, {c['FAILED']} FAILED** over "
            f"{len(dh['results'])} results.{weak_note}",
            "",
            _table(
                ["test", "ntup", "results", "PASSED", "WEAK", "FAILED", "lowest p"], rows, "llrrrrr"
            ),
        ]
    )


def _md_limits(results: Mapping[str, Any]) -> str:
    bullets = [
        "**Simulated data.** The tank, its fish, its light and its camera noise are models "
        "driven by a seeded generator. The numbers say how the pipeline behaves on that model; "
        "a real aquarium and camera need their own run of this report (`puffer analyze "
        "--source camera:0` or a lossless recording).",
        "**Entropy is only measured on raw samples.** SP 800-22, ent and dieharder on hashed or "
        "keyed output pass whatever the input; they check the conditioning and the generator, "
        "not the fish.",
        "**Short fish data.** SP 800-90B wants 1,000,000 samples; the fish channel produced a "
        "few percent of that, so its estimate is a rough lower bound, and the fish-only "
        "conditioned stream is too short for most SP 800-22 tests.",
        "**Not exactly reproducible.** The engine draws its sensor sample sites from "
        "`os.urandom` (as in production) and PufferRNG is keyed with `os.urandom`, so a re-run "
        "with the same seed gives close but not identical figures.",
        "**Liveness proves freshness, not authenticity.** An attacker who watches the LED live "
        "and composites it into fake video passes (DESIGN.md section 5). Even then outputs stay "
        "as strong as `os.urandom`.",
    ]
    return "## 9. What this does not show\n\n" + "\n".join(f"* {b}" for b in bullets)


def _md_reproduce(results: Mapping[str, Any]) -> str:
    cfg = results["config"]
    default = ReportConfig(sources=list(DEFAULT_SOURCES))
    cli = ["puffer", "analyze"]
    if list(cfg["sources"]) != list(DEFAULT_SOURCES):
        for spec in cfg["sources"]:
            cli += ["--source", f'"{spec}"']
    if cfg["frames"] != default.frames:
        cli += ["--frames", str(cfg["frames"])]
    if cfg["nist_sequences"] != default.nist_sequences:
        cli += ["--nist-sequences", str(cfg["nist_sequences"])]
    cli += ["--seed", str(results["seed"])]
    if cfg["dieharder"]:
        cli.append("--dieharder")
    extensions = {
        k: v
        for k, v in cfg.items()
        if k
        not in {
            "sources",
            "frames",
            "nist_sequences",
            "dieharder",
            "seed",
            "out_dir",
            "media_dir",
            "jobs",
        }
        and v != _jsonable(getattr(default, k))
    }
    text = [
        "## 10. Reproduce",
        "",
        "```sh",
        " ".join(cli),
        "```",
    ]
    if extensions:
        text += [
            "",
            "with these `ReportConfig` fields changed from their defaults (Python API: "
            "`puffer.analysis.report.run_report`): "
            + ", ".join(f"`{k}={v}`" for k, v in extensions.items())
            + ".",
        ]
    timings = results.get("timings", {})
    if timings:
        text += [
            "",
            f"This run took {_num(timings.get('total_s', 0) / 60, 1)} min: collection "
            f"{_num(timings.get('collection_s', 0), 0)} s, SP 800-90B "
            f"{_num(timings.get('sp800_90b_s', 0), 0)} s, streams and SP 800-22 "
            f"{_num(timings.get('streams_s', 0), 0)} s, dieharder "
            f"{_num(timings.get('dieharder_s', 0), 0)} s, with "
            f"{results.get('workers', 1)} worker processes.",
        ]
    return "\n".join(text)
