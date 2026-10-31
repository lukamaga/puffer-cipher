"""Wiring decisions of :mod:`puffer.runtime`: policies, LED, data dir, gallery, names."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from puffer.runtime import (
    FISH_GALLERY,
    build_engine,
    ensure_data_dir,
    resolve_led,
    resolve_liveness,
    source_kind,
)
from puffer.sources import SourceSpecError, SyntheticAquarium, SyntheticConfig
from puffer.types import LivenessState

from .helpers import HEIGHT, WIDTH, ScriptedSource, config, run_frames, sim_spec


@pytest.mark.parametrize(
    ("policy", "kind", "expected"),
    [
        ("auto", "synthetic", "required"),
        ("auto", "camera", "required"),
        ("auto", "video", "off"),
        ("required", "video", "required"),
        ("advisory", "synthetic", "advisory"),
        ("off", "camera", "off"),
    ],
)
def test_resolve_liveness(policy: str, kind: str, expected: str) -> None:
    assert resolve_liveness(policy, kind) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("led", "kind", "on_pi", "expected"),
    [
        ("auto", "synthetic", False, "simulated"),
        ("auto", "camera", True, "gpio:17"),
        ("auto", "camera", False, "none"),
        ("auto", "video", True, "none"),
        ("gpio:4", "camera", False, "gpio:4"),
        ("none", "synthetic", False, "none"),
    ],
)
def test_resolve_led(led: str, kind: str, on_pi: bool, expected: str) -> None:
    assert resolve_led(led, kind, on_pi=on_pi) == expected  # type: ignore[arg-type]


def test_unknown_sources_count_as_live_cameras() -> None:
    assert source_kind(ScriptedSource()) == "camera"
    small = SyntheticConfig(width=WIDTH, height=HEIGHT, led_xy=(WIDTH - 32, 32), seed=1)
    assert source_kind(SyntheticAquarium(small)) == "synthetic"


def test_synthetic_defaults_require_liveness_on_the_simulators_led(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path / "data"))
    try:
        assert engine.liveness_policy == "required"
        verifier = engine.liveness_verifier
        assert verifier is not None
        assert isinstance(engine.source, SyntheticAquarium)
        assert verifier.roi == engine.source.led_roi == engine.led_roi
        assert verifier.window_bits == 64
        assert verifier.schedule.symbol_s == 0.2
        assert engine.source.led_schedule is verifier.schedule  # one shared challenge
        assert engine.led is None  # the simulator renders the LED itself
        assert not engine.live_source
        snap = run_frames(engine, 2).engine.snapshot()
        assert snap is not None
        assert snap.liveness.state is LivenessState.CALIBRATING
    finally:
        engine.close()
    mode = stat.S_IMODE((tmp_path / "data").stat().st_mode)
    assert mode == 0o700


def test_liveness_off_builds_no_verifier_and_led_none_keeps_the_tank_dark(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path, liveness="off", led="none"))
    try:
        assert engine.liveness_verifier is None
        assert isinstance(engine.source, SyntheticAquarium)
        assert engine.source.led_schedule is None
        snap = run_frames(engine, 2).engine.snapshot()
        assert snap is not None
        assert snap.liveness.state is LivenessState.DISABLED
    finally:
        engine.close()


def test_configured_led_roi_and_window_reach_the_verifier(tmp_path: Path) -> None:
    roi = (280, 20, 24, 24)
    engine = build_engine(config(tmp_path, led_roi=roi, liveness_window_bits=48, symbol_s=0.25))
    try:
        verifier = engine.liveness_verifier
        assert verifier is not None
        assert verifier.roi == roi
        assert engine.led_roi == roi
        assert verifier.window_bits == 48
        assert verifier.schedule.symbol_s == 0.25
    finally:
        engine.close()


def test_gallery_persists_and_fish_names_override(tmp_path: Path) -> None:
    engine = build_engine(config(tmp_path, liveness="off"))
    try:
        run_frames(engine, 90)
        identities = dict(engine.tracker.identities)
        assert identities
    finally:
        engine.close()
    saved = json.loads((tmp_path / FISH_GALLERY).read_text())
    assert {item["fish_id"] for item in saved["identities"]} == set(identities)

    target = sorted(identities)[0]
    names = {target: "Sir Swims-a-Lot"}
    engine = build_engine(config(tmp_path, liveness="off", fish_names=names))
    try:
        assert set(engine.tracker.identities) == set(identities)  # loaded at start
        run_frames(engine, 1)
        assert engine.tracker.identities[target].name == "Sir Swims-a-Lot"
        assert engine.receipt_context().fish_names[target] == "Sir Swims-a-Lot"
        card = next(c for c in engine.fish_cards() if c["fish_id"] == target)
        assert card["name"] == "Sir Swims-a-Lot"
    finally:
        engine.close()


def test_corrupt_gallery_is_moved_aside(tmp_path: Path) -> None:
    (tmp_path / FISH_GALLERY).write_text("{not json")
    engine = build_engine(config(tmp_path, liveness="off"))
    try:
        assert engine.tracker.identities == {}
        assert (tmp_path / (FISH_GALLERY + ".bad")).read_text() == "{not json"
    finally:
        engine.close()


def test_build_failures_are_clean(tmp_path: Path) -> None:
    with pytest.raises(SourceSpecError, match="did you mean"):
        build_engine(config(tmp_path, source="synthetc"))
    with pytest.raises(ImportError, match="ultralytics"):
        build_engine(config(tmp_path, source=sim_spec(), detector="yolo:fish.pt"))


def test_ensure_data_dir_creates_private_directory(tmp_path: Path) -> None:
    path = ensure_data_dir(tmp_path / "a" / "b")
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert ensure_data_dir(path) == path  # existing: left as is
