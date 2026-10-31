"""FishStats: daily aggregates, frame-clock flushing, leaderboard, persistence."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from puffer.fishstats import SCHEMA_VERSION, FishStats, FishStatsError, utc_day

DAY0 = 1_793_404_800.0  # 2026-10-31T00:00:00Z
DAY = 86_400.0


def rows(path: Path) -> list[tuple[object, ...]]:
    """What is durably in the database (a separate connection)."""
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT fish_id, day, name, bits, seen_s, distance_px, chaos_sum, chaos_count "
            "FROM fish_daily ORDER BY fish_id, day"
        ).fetchall()


def test_utc_day() -> None:
    assert utc_day(DAY0) == "2026-10-31"
    assert utc_day(DAY0 - 1) == "2026-10-30"


def test_records_accumulate_and_flush_every_ten_seconds_of_frame_time(tmp_path: Path) -> None:
    path = tmp_path / "puffer.db"
    stats = FishStats(path, clock=lambda: DAY0 + 3600)
    for i in range(300):  # 10 s at 30 fps
        stats.record("f1", t=i / 30, wall_time=DAY0 + 3600, name="Nemo", bits=1.5, seen_s=1 / 30)
    assert rows(path) == []  # batched in memory
    stats.record("f1", t=10.0, wall_time=DAY0 + 3600, bits=1.5, chaos_score=0.4)
    (row,) = rows(path)
    assert row[:3] == ("f1", "2026-10-31", "Nemo")
    assert row[3] == pytest.approx(301 * 1.5)
    assert row[4] == pytest.approx(10.0)
    assert row[6:] == (pytest.approx(0.4), 1)
    stats.record("f1", t=10.5, wall_time=DAY0 + 3600, bits=2.0)
    stats.close()
    stats.close()  # idempotent
    assert rows(path)[0][3] == pytest.approx(301 * 1.5 + 2.0)  # flushed on close


def test_leaderboard_ranks_by_mean_chaos_within_the_window(tmp_path: Path) -> None:
    now = DAY0 + 12 * 3600
    with FishStats(tmp_path / "s.db", clock=lambda: now) as stats:
        # f1: very chaotic but 10 days ago (outside a 7-day window)
        stats.record("f1", t=0, wall_time=now - 10 * DAY, name="Old", bits=500, chaos_score=0.99)
        stats.record("f2", t=1, wall_time=now - DAY, name="Calm", bits=100, chaos_score=0.2)
        stats.record("f2", t=2, wall_time=now, name="Calm Two", bits=50, chaos_score=0.4)
        stats.record("f3", t=3, wall_time=now, name="Wild", bits=10, chaos_score=0.8)
        stats.record("f4", t=4, wall_time=now, name="Silent", bits=900)  # no chaos samples
        stats.record("f5", t=5, wall_time=now, name="Twin", bits=20, chaos_score=0.3)
        board = stats.leaderboard(days=7)
        assert [e["fish_id"] for e in board] == ["f3", "f2", "f5", "f4"]
        assert board[0] == {
            "rank": 1,
            "fish_id": "f3",
            "name": "Wild",
            "chaos_score": 0.8,
            "bits": 10.0,
        }
        assert board[1]["chaos_score"] == pytest.approx(0.3)  # mean of 0.2 and 0.4
        assert board[1]["name"] == "Calm Two"
        assert board[1]["bits"] == 150.0
        assert board[3]["chaos_score"] is None
        assert stats.leaderboard(days=30)[0]["fish_id"] == "f1"
        assert [e["fish_id"] for e in stats.leaderboard(days=1)] == ["f3", "f2", "f5", "f4"]
        with pytest.raises(ValueError, match="positive integer"):
            stats.leaderboard(days=0)


def test_totals_split_today_from_all_time(tmp_path: Path) -> None:
    now = DAY0 + 600
    with FishStats(tmp_path / "s.db", clock=lambda: now) as stats:
        stats.record("f1", t=0, wall_time=now - DAY, bits=40, seen_s=5, distance_px=100)
        stats.record("f1", t=1, wall_time=now, bits=2, seen_s=1, distance_px=7, chaos_score=0.5)
        totals = stats.totals()["f1"]
        assert totals.bits_total == 42
        assert totals.bits_today == 2
        assert totals.seen_s == 6
        assert totals.distance_px == 107
        assert totals.chaos_score == 0.5
        assert totals.name == "f1"
        assert stats.totals(days=1)["f1"].bits_total == 2


def test_persistence_and_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    with FishStats(path) as stats:
        assert stats.schema_version == SCHEMA_VERSION
        stats.record("f7", t=0, wall_time=DAY0, name="Zap", bits=3)
    with FishStats(path, clock=lambda: DAY0) as stats:
        assert stats.totals()["f7"].bits_total == 3
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(FishStatsError, match="newer"):
        FishStats(path)


def test_rejects_bad_amounts_and_use_after_close() -> None:
    stats = FishStats()
    for bad in ({"bits": -1.0}, {"seen_s": float("nan")}, {"distance_px": float("inf")}):
        with pytest.raises(ValueError, match="finite and non-negative"):
            stats.record("f1", t=0, wall_time=DAY0, **bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fish_id"):
        stats.record("", t=0, wall_time=DAY0)
    stats.close()
    with pytest.raises(FishStatsError, match="closed"):
        stats.record("f1", t=0, wall_time=DAY0)


def test_concurrent_writer_and_readers(tmp_path: Path) -> None:
    stats = FishStats(tmp_path / "s.db", flush_every_s=0.01, clock=lambda: DAY0)
    errors: list[BaseException] = []

    def write() -> None:
        try:
            for i in range(2000):
                stats.record(f"f{i % 5}", t=i * 0.005, wall_time=DAY0, bits=1, chaos_score=0.5)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def read() -> None:
        try:
            for _ in range(50):
                stats.leaderboard()
                stats.totals()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=write)] + [threading.Thread(target=read) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sum(t.bits_total for t in stats.totals().values()) == 2000
    stats.close()
