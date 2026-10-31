"""Receipt-to-summary mapping (DESIGN.md 4.5 receipt -> 3.4 summary fields)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from puffer.beacon.pulse import canonical_summary
from puffer.beacon.summary import floor_bits, summary_from_receipt

# The example receipt from DESIGN.md section 4.5.
RECEIPT: dict[str, Any] = {
    "receipt_version": 1,
    "purpose": "beacon",
    "bytes": 64,
    "timestamp_ms": 1793454000000,
    "frames": [1200, 1290],
    "mode": "day",
    "liveness": {"state": "verified", "confidence_bits": 54.2},
    "channels": {
        "sensor": {"credited_bits": 812.0, "share": 0.61, "health": "ok"},
        "fish": {"credited_bits": 512.5, "share": 0.39, "health": "ok"},
        "system": {"credited_bits": 256.0, "share": None, "health": "ok"},
    },
    "fish": [
        {"fish_id": "f3", "name": "Nemo", "bits": 120.5, "share": 0.31},
        {"fish_id": "f1", "name": "Dory", "bits": 200.9, "share": 0.5},
        {"fish_id": "f2", "name": "Puffy", "bits": 120.5, "share": 0.19},
        {"fish_id": "f4", "name": "Bubbles", "bits": 1.0, "share": 0.0},
    ],
}


def test_design_example_maps_to_the_documented_fields() -> None:
    summary = summary_from_receipt(RECEIPT)
    assert summary == {
        "fish_count": 4,
        "fish_bits": 512,
        "sensor_bits": 812,
        "system_bits": 256,
        "mode": "day",
        "liveness": "verified",
        "liveness_confidence_bits": 54,
        "health": {"sensor": "ok", "fish": "ok", "system": "ok"},
        "top_fish": [
            {"fish_id": "f1", "name": "Dory", "bits": 200},
            {"fish_id": "f2", "name": "Puffy", "bits": 120},  # tie broken by fish_id
            {"fish_id": "f3", "name": "Nemo", "bits": 120},
        ],
        "frames": [1200, 1290],
    }
    canonical_summary(summary)  # integers only: signable as-is


def test_top_n_and_missing_camera_data() -> None:
    receipt = copy.deepcopy(RECEIPT)
    del receipt["channels"]["fish"]
    receipt["fish"] = []
    receipt["frames"] = None
    summary = summary_from_receipt(receipt, top_n=0)
    assert summary["fish_bits"] == 0
    assert summary["fish_count"] == 0
    assert summary["top_fish"] == []
    assert summary["frames"] == []
    assert summary_from_receipt(RECEIPT, top_n=1)["top_fish"] == [
        {"fish_id": "f1", "name": "Dory", "bits": 200}
    ]


def test_floor_bits_never_rounds_up() -> None:
    assert floor_bits(0.999) == 0
    assert floor_bits(7) == 7
    assert floor_bits(255.9999) == 255


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.5])
def test_floor_bits_rejects_impossible_amounts(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        floor_bits(value)


@pytest.mark.parametrize("value", [True, "12", None])
def test_floor_bits_rejects_non_numbers(value: object) -> None:
    with pytest.raises(TypeError, match="number"):
        floor_bits(value)


def test_negative_top_n_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_n"):
        summary_from_receipt(RECEIPT, top_n=-1)


def test_missing_required_field_is_loud() -> None:
    receipt = copy.deepcopy(RECEIPT)
    del receipt["liveness"]
    with pytest.raises(KeyError, match="liveness"):
        summary_from_receipt(receipt)
