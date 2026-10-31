"""Entropy receipts: DESIGN.md 4.5 JSON shape, round trip, integer-only beacon summary."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from puffer.entropy.pool import PoolDraw
from puffer.entropy.receipt import EntropyReceipt, ReceiptContext
from puffer.types import HealthState, LivenessState, LivenessStatus

CONTEXT = ReceiptContext(
    mode="day",
    liveness=LivenessStatus(LivenessState.VERIFIED, confidence_bits=54.2),
    health={"sensor": HealthState.OK, "fish": HealthState.DEGRADED},
    fish_names={"f3": "Nemo", "f1": "Dory"},
)


def make_draw(**overrides: Any) -> PoolDraw:
    fields: dict[str, Any] = {
        "draw_index": 4,
        "material": b"\x00" * 128,
        "channels": ("fish", "sensor"),
        "credits": {"sensor": 812.0, "fish": 512.5},
        "fish_credits": {"f3": 120.5, "f1": 300.0, "track-7": 92.0},
        "frames": (1200, 1290),
    }
    fields.update(overrides)
    return PoolDraw(**fields)


def make_receipt(draw: PoolDraw | None = None, context: ReceiptContext = CONTEXT) -> EntropyReceipt:
    return EntropyReceipt.from_draw(
        draw or make_draw(),
        purpose="key",
        n_bytes=32,
        timestamp_ms=1_793_454_000_000,
        context=context,
    )


def test_to_dict_has_exactly_the_design_shape():
    d = make_receipt().to_dict()
    assert list(d) == [
        "receipt_version",
        "purpose",
        "bytes",
        "timestamp_ms",
        "frames",
        "mode",
        "liveness",
        "channels",
        "fish",
    ]
    assert d["receipt_version"] == 1
    assert (d["purpose"], d["bytes"], d["timestamp_ms"]) == ("key", 32, 1_793_454_000_000)
    assert d["frames"] == [1200, 1290]
    assert d["mode"] == "day"
    assert d["liveness"] == {"state": "verified", "confidence_bits": 54.2}
    assert list(d["channels"]) == ["sensor", "fish", "system"]
    assert d["channels"]["sensor"] == {"credited_bits": 812.0, "share": 0.6131, "health": "ok"}
    assert d["channels"]["fish"] == {"credited_bits": 512.5, "share": 0.3869, "health": "degraded"}
    assert d["channels"]["system"] == {"credited_bits": 256.0, "share": None, "health": "ok"}
    assert d["fish"][0] == {"fish_id": "f1", "name": "Dory", "bits": 300.0, "share": 0.5854}
    assert [f["fish_id"] for f in d["fish"]] == ["f1", "f3", "track-7"]  # by bits, descending
    assert d["fish"][2]["name"] == "track-7"  # unnamed fish fall back to their id
    json.dumps(d, allow_nan=False)


def test_shares_cover_camera_channels_and_the_fish_channel():
    receipt = make_receipt()
    camera = [c.share for name, c in receipt.channels.items() if name != "system"]
    assert sum(s for s in camera if s is not None) == pytest.approx(1.0)
    assert sum(f.share for f in receipt.fish) == pytest.approx((300 + 120.5 + 92) / 512.5)
    assert receipt.camera_bits == pytest.approx(1324.5)


def test_round_trip():
    d = make_receipt().to_dict()
    again = EntropyReceipt.from_dict(json.loads(json.dumps(d)))
    assert again.to_dict() == d
    assert again.liveness_state is LivenessState.VERIFIED
    assert again.channels["fish"].health is HealthState.DEGRADED


def test_empty_draw_without_context():
    receipt = make_receipt(
        make_draw(credits={}, fish_credits={}, frames=None, channels=(), material=b""),
        ReceiptContext(),
    )
    d = receipt.to_dict()
    assert d["frames"] == []
    assert d["mode"] == "unknown"
    assert d["liveness"] == {"state": "disabled", "confidence_bits": 0.0}
    assert d["channels"]["sensor"] == {"credited_bits": 0.0, "share": None, "health": None}
    assert d["fish"] == []
    assert EntropyReceipt.from_dict(d).to_dict() == d


def test_non_finite_amounts_claim_nothing():
    context = ReceiptContext(
        liveness=LivenessStatus(LivenessState.VERIFIED, confidence_bits=math.inf)
    )
    receipt = make_receipt(make_draw(credits={"sensor": math.nan, "fish": -3.0}), context)
    assert receipt.liveness_confidence_bits == 0.0
    assert receipt.channels["sensor"].credited_bits == 0.0
    assert receipt.camera_bits == 0.0
    json.dumps(receipt.to_dict(), allow_nan=False)


def _only_beacon_types(value: Any) -> bool:
    if isinstance(value, dict):
        return all(isinstance(k, str) and _only_beacon_types(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_only_beacon_types(v) for v in value)
    return isinstance(value, (int, str, bool)) and not isinstance(value, float)


def test_beacon_summary_is_integer_only_and_rounds_down():
    summary = make_receipt().summary_for_beacon()
    assert _only_beacon_types(summary)
    assert summary == {
        "fish_count": 3,
        "fish_bits": 512,
        "sensor_bits": 812,
        "system_bits": 256,
        "mode": "day",
        "liveness": "verified",
        "liveness_confidence_bits": 54,
        "health": {"sensor": "ok", "fish": "degraded", "system": "ok"},
        "top_fish": [
            {"fish_id": "f1", "name": "Dory", "bits": 300},
            {"fish_id": "f3", "name": "Nemo", "bits": 120},
            {"fish_id": "track-7", "name": "track-7", "bits": 92},
        ],
        "frames": [1200, 1290],
    }
    json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_beacon_summary_lists_at_most_three_fish():
    many = {f"f{i}": float(i) for i in range(1, 8)}
    summary = make_receipt(make_draw(fish_credits=many)).summary_for_beacon()
    assert summary["fish_count"] == 7
    assert [f["fish_id"] for f in summary["top_fish"]] == ["f7", "f6", "f5"]
    empty = make_receipt(make_draw(frames=None), ReceiptContext()).summary_for_beacon()
    assert empty["frames"] == []
    assert empty["health"] == {"system": "ok"}
    assert _only_beacon_types(empty)


def test_from_dict_rejects_unknown_versions():
    d = make_receipt().to_dict()
    with pytest.raises(ValueError, match="receipt_version"):
        EntropyReceipt.from_dict({**d, "receipt_version": 2})
    with pytest.raises(ValueError, match="frames"):
        EntropyReceipt.from_dict({**d, "frames": [1]})
