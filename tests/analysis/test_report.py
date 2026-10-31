"""Smoke test of the statistics report on a tiny run, plus its pure helpers.

The real report takes half an hour; this runs every section (except
dieharder) on a few dozen frames at 320x240 and checks that the documents
and charts are produced and that ``results.json`` has the documented shape.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from puffer.analysis import report
from puffer.analysis.report import ReportConfig, run_report

SOURCES = [
    "synthetic:mode=day,width=320,height=240,fish=4",
    "synthetic:mode=night,width=320,height=240,fish=4",
]
CHARTS = (
    "min-entropy",
    "day-night-cycle",
    "liveness-replay",
    "sp800-22-proportions",
    "sp800-22-pvalues",
    "ent-raw-vs-conditioned",
)


@pytest.fixture(scope="module")
def tiny_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], Path]:
    pytest.importorskip("matplotlib")
    root = tmp_path_factory.mktemp("report")
    config = ReportConfig(
        sources=SOURCES,
        frames=48,
        out_dir=root / "results",
        media_dir=root / "media",
        nist_sequences=2,
        nist_bits=4096,
        output_sequences=3,
        max_90b_samples=8000,
        cycle_frames=60,
        cycle_day_s=1.0,
        cycle_night_s=1.0,
        liveness_s=1.5,
        tracking_frames=30,
        tracking_warmup=10,
        jobs=1,
        seed=3,
    )
    messages: list[str] = []
    path = run_report(config, progress=messages.append)
    assert messages, "progress callback was never called"
    results = json.loads((config.out_dir / "results.json").read_text(encoding="utf-8"))
    return path, results, config.media_dir


def test_documents_and_charts_are_written(tiny_report: tuple[Path, dict[str, Any], Path]) -> None:
    path, _, media = tiny_report
    assert path.name == "RESULTS.md"
    assert path.is_file()
    for name in CHARTS:
        for theme in ("light", "dark"):
            png = media / f"{name}-{theme}.png"
            assert png.is_file(), png
            assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_markdown_states_the_data_source_and_links_charts(
    tiny_report: tuple[Path, dict[str, Any], Path],
) -> None:
    path, _, _ = tiny_report
    text = path.read_text(encoding="utf-8")
    assert "simulated aquarium" in text
    assert "seed 3" in text
    assert "No real fish and no real camera were involved" in text
    assert "\u2014" not in text  # no em dashes
    assert "\u2013" not in text  # no en dashes
    for name in CHARTS:
        assert f"../media/{name}-light.png" in text
        assert f"../media/{name}-dark.png" in text
    assert "dieharder" in text
    assert "Not run: not requested" in text


def test_results_json_structure(tiny_report: tuple[Path, dict[str, Any], Path]) -> None:
    _, results, _ = tiny_report
    assert results["report_version"] == report.REPORT_VERSION
    assert results["seed"] == 3
    assert set(results["runs"]) == {"day", "night"}
    for label in ("day", "night"):
        run = results["runs"][label]
        assert run["source"]["kind"] == "synthetic"
        assert run["source"]["seed"] == 3
        assert run["source"]["width"] == 320
        assert run["frames"] == 48
        assert run["overall"]["sensor"]["samples"] > 0
        for channel in ("sensor", "fish"):
            assessment = results["sp800_90b"][label][channel]
            assert assessment["standard_samples"] == 1_000_000
            assert assessment["n_used"] <= 8000
    assert results["runs"]["night"]["night_fraction"] == 1.0
    sensor = results["sp800_90b"]["day"]["sensor"]
    assert 0 < sensor["min_entropy"] <= 8
    tracks = {r["details"]["track"] for r in sensor["results"]}
    assert tracks == {"original", "bitstring"}

    streams = results["streams"]
    for key in ("day/sensor", "day/fish", "day/combined", "day/pufferrng", "night/sensor"):
        assert key in streams
    assert streams["day/pufferrng"]["bytes"] == 3 * 4096 // 8
    final = results["sp800_22"]["day/pufferrng"]
    assert final["mode"] == "multi"
    assert final["sequences"] == 3
    assert final["n_bits"] == 4096
    assert sum(final["pooled_histogram"]) == sum(len(v) for v in final["p_values"].values())
    assert 0 <= final["expected_flagged"] < final["subtest_count"]
    assert results["sp800_22"]["day/sensor"]["mode"] in ("multi", "single", "none")

    assert results["ent"]["raw"]["day/sensor"]["entropy"] < 8.0
    assert results["ent"]["streams"]["day/pufferrng"]["size"] == 3 * 4096 // 8

    cycle = results["cycle"]
    assert cycle["frames"] == 60
    assert any(cycle["bins"]["night"])
    assert not all(cycle["bins"]["night"])
    assert {p["phase"] for p in cycle["phases"]} >= {"day", "night"}

    live = results["liveness"]
    assert live["threshold_bits"] == 24.0
    for key in ("live", "replay", "control"):
        assert live[key]["frames"] > 0
    assert live["replay"]["time_to_verified_s"] is None
    assert live["replay"]["camera_credited_bits"] == 0.0
    assert live["recording"]["codec"] == "FFV1"

    assert set(results["tracking"]) == {"day", "night"}
    assert "fish_recall" in results["tracking"]["day"]["metrics"]
    assert results["dieharder"]["ran"] is False


def test_results_json_is_strict_json(tiny_report: tuple[Path, dict[str, Any], Path]) -> None:
    path, _, _ = tiny_report
    raw = (path.parent / "results.json").read_text(encoding="utf-8")
    json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))


def test_markdown_rerenders_from_saved_results(
    tiny_report: tuple[Path, dict[str, Any], Path],
) -> None:
    _path, results, _ = tiny_report
    assert report.render_markdown(results, charts={}).startswith("# Results")


# ------------------------------------------------------------------ pure helpers


def test_seed_and_led_are_filled_into_synthetic_specs() -> None:
    items = report._synthetic_items("synthetic:mode=night,width=320,height=240")
    assert items is not None
    prepared = dict(report._prepare_synthetic(items, seed=9))
    assert prepared["seed"] == "9"
    assert prepared["realtime"] == "0"
    assert prepared["led"] == "304:16"  # the 640x480 default position, scaled
    explicit = dict(report._prepare_synthetic(report._synthetic_items("synthetic:seed=2") or [], 9))
    assert explicit["seed"] == "2"
    assert report._synthetic_items("video:tank.mkv") is None


def test_plan_derives_cycle_liveness_and_tracking_from_the_first_simulator() -> None:
    plan = report._plan(ReportConfig(sources=["video:a.mkv", *SOURCES], seed=5, frames=10))
    assert [job.label for job in plan.runs] == ["a.mkv", "day", "night"]
    assert plan.runs[0].output_bytes > 0  # the final output comes from the first run
    assert plan.cycle is not None
    assert "mode=cycle" in plan.cycle.spec
    assert plan.liveness is not None
    assert "mode=day" in plan.liveness.spec
    assert [t.label for t in plan.tracking] == ["day", "night"]
    only_video = report._plan(ReportConfig(sources=["video:a.mkv"], seed=5, frames=10))
    assert only_video.cycle is None
    assert only_video.liveness is None
    assert set(only_video.skipped) == {"cycle", "liveness", "tracking"}


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="source"):
        ReportConfig(sources=[])
    with pytest.raises(ValueError, match="tracking_frames"):
        ReportConfig(sources=SOURCES, tracking_frames=10, tracking_warmup=20)
    with pytest.raises(ValueError, match="dieharder_tests"):
        ReportConfig(sources=SOURCES, dieharder_tests=("x",))
    cfg = ReportConfig(sources=SOURCES, dieharder_tests=("0", "203:8", "all"))
    assert cfg.dieharder_tests == ("0", "203:8", "all")
    assert report._dieharder_args("203:8") == ["-d", "203", "-n", "8"]
    assert report._dieharder_args("all") == ["-a"]


def test_chance_of_a_flagged_subtest() -> None:
    # 10 sequences: the band's lower edge is 0.896, so two or more failures flag it.
    p = report._chance_flag(10, (0.896, 1.084))
    assert p == pytest.approx(1 - 0.99**10 - 10 * 0.01 * 0.99**9, rel=1e-9)
    assert report._chance_flag(100, (0.960, 1.020)) > 0.0001


def test_jsonable_drops_non_finite_numbers() -> None:
    out = report._jsonable({"a": math.nan, "b": [1.0, math.inf], "c": b"xyz", "d": (1, 2)})
    assert out == {"a": None, "b": [1.0, None], "c": 3, "d": [1, 2]}


class _FakeRng:
    def generate(self, n: int, purpose: str) -> tuple[bytes, None]:
        return bytes(n), None


def test_dieharder_pipe_feeds_until_the_reader_stops() -> None:
    row = "   diehard_birthdays|   0|       100|     100|0.50000000|  PASSED  "
    script = f"import sys; sys.stdin.buffer.read(3 << 20); print({row!r})"
    text, fed, calls, expired = report._dieharder_once(
        [sys.executable, "-c", script], _FakeRng(), timeout_s=60
    )
    assert not expired
    assert fed >= 3 << 20
    assert calls >= 3
    [result] = report.parse_dieharder_output(text)
    assert result.test_name == "diehard_birthdays"
    assert result.assessment.value == "PASSED"


def test_dieharder_budget_kills_a_slow_reader() -> None:
    script = "import time; time.sleep(30)"
    _, _, _, expired = report._dieharder_once(
        [sys.executable, "-c", script], _FakeRng(), timeout_s=0.5
    )
    assert expired
