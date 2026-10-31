"""Wire serializers (puffer.api.wire): sanitising, adaptations and the dashboard contract."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
from fastapi.responses import JSONResponse

from puffer.api import wire
from puffer.beacon import canonical_summary, create_pulse
from puffer.beacon.pulse import GENESIS_PREV_OUTPUT
from puffer.beacon.vectors import vector_signing_key
from puffer.engine import FishLabel
from puffer.types import HealthReport, HealthState, LivenessState, LivenessStatus

from .dashboard_protocol import MESSAGE_DECODERS, REST_DECODERS
from .fakes import FakeChannelStats, FakeEngine


@pytest.fixture
def snapshot():
    snap = FakeEngine(auto_advance=False).advance()
    return snap


def test_dumps_matches_fastapi_and_refuses_nan() -> None:
    value = {"a": [1, 2.5, "é"], "b": None}
    assert wire.dumps(value).encode() == JSONResponse(value).body
    with pytest.raises(ValueError, match="JSON compliant"):
        wire.dumps({"x": math.nan})


def test_pulse_json_is_byte_identical_for_rest_and_socket() -> None:
    pulse = create_pulse(
        vector_signing_key(),
        index=0,
        timestamp_ms=1_793_454_000_000,
        period_ms=60_000,
        local_random=bytes(64),
        next_local_random=bytes(range(64)),
        prev_output=GENESIS_PREV_OUTPUT,
        summary=canonical_summary({"fish_bits": 1, "name": 'Ünïcode "quoted"'}),
    )
    text = wire.pulse_text(pulse)
    assert text.encode() == JSONResponse(pulse.to_json()).body
    message = wire.raw_envelope("beacon", text)
    assert json.loads(message) == {"type": "beacon", "data": pulse.to_json()}
    assert message == wire.envelope("beacon", pulse.to_json())
    assert wire.pulses_text([pulse, pulse]) == f"[{text},{text}]"
    assert wire.pulses_text([]) == "[]"


def test_every_snapshot_message_satisfies_the_dashboard(snapshot) -> None:
    identities = {"f1": ("Nemo", "#ff8800")}
    MESSAGE_DECODERS["frame"](wire.frame_message(snapshot, b"\xff\xd8", identities))
    MESSAGE_DECODERS["entropy"](wire.entropy_message(snapshot, 4096))
    MESSAGE_DECODERS["health"](wire.health_message(snapshot))
    MESSAGE_DECODERS["led"](wire.liveness_status(snapshot.liveness))
    MESSAGE_DECODERS["bits"](wire.bits_message(1.5, "0101"))
    REST_DECODERS["status"](wire.status_message(snapshot, source="x", uptime_s=3.0))


def test_non_finite_numbers_never_reach_the_wire(snapshot) -> None:
    bad = FakeChannelStats("sensor", math.nan, math.inf, -math.inf, HealthState.OK)
    snap = replace(
        snapshot,
        t=math.nan,
        fps=math.inf,
        channels={**snapshot.channels, "sensor": bad},
        pool={"sensor": math.nan, "fish": 1.0},
    )
    data = wire.entropy_message(snap, math.nan)
    wire.dumps(data)  # would raise on NaN/Infinity
    sensor = data["channels"]["sensor"]
    assert (sensor["bits_per_s"], sensor["h_per_sample"], sensor["credited_total"]) == (
        0.0,
        None,
        0.0,
    )
    assert data["pool"] == {"sensor": 0.0, "fish": 1.0, "capacity_bits": 0.0}
    wire.dumps(wire.status_message(snap, source="x", uptime_s=math.nan))


def test_missing_channels_and_unknown_states_fail_closed(snapshot) -> None:
    snap = replace(snapshot, channels={"sensor": snapshot.channels["sensor"]}, overall="weird")
    data = wire.entropy_message(snap, 4096)
    assert data["channels"]["fish"] == {
        "bits_per_s": 0.0,
        "h_per_sample": None,
        "credited_total": 0.0,
        "health": "failed",
    }
    assert wire.health_message(snap)["overall"] == "failed"
    report = HealthReport("fish", HealthState.OK, ("x",), {"rct": "bogus"})
    assert wire.health_report(report, with_channel=False)["checks"] == {"rct": "failed"}


@pytest.mark.parametrize(
    ("observed", "wire_observed"),
    [("0110", "0110"), ("01?0", "01?0"), ("????", "????"), ("01x0", ""), ("", "")],
)
def test_undecodable_liveness_bits_pass_through(observed, wire_observed) -> None:
    status = LivenessStatus(LivenessState.CALIBRATING, expected="0110", observed=observed)
    data = wire.liveness_status(status)
    assert data["observed"] == wire_observed
    assert "observed_raw" not in data


def test_receipt_frames_null_when_empty() -> None:
    class Receipt:
        def __init__(self, frames):
            self.frames = frames

        def to_dict(self):
            return {"frames": self.frames, "purpose": "key"}

    assert wire.receipt(Receipt([]))["frames"] is None
    assert wire.receipt(Receipt([3, 9]))["frames"] == [3, 9]


def test_fish_cards_are_normalised() -> None:
    card = wire.fish_card(
        {
            "fish_id": "f9",
            "color": "red; background: url(x)",
            "present": 1,
            "track_id": True,
            "stats": {"activity": 7.0, "chaos_score": -1, "speed_avg": math.nan},
            "secret_internal_field": "dropped",
        }
    )
    REST_DECODERS["fish"]([card])
    assert card["name"] == "f9"
    assert card["color"] == "#8a8a8a"
    assert card["present"] is True
    assert card["track_id"] is None
    assert (card["stats"]["activity"], card["stats"]["chaos_score"]) == (1.0, 0.0)
    assert card["stats"]["speed_avg"] == 0.0
    assert "secret_internal_field" not in card


def test_leaderboard_rank_defaults_to_position() -> None:
    entry = wire.leaderboard_entry({"fish_id": "f2", "chaos_score": 0.4, "bits": 12}, 5)
    assert entry == {"rank": 5, "fish_id": "f2", "name": "f2", "chaos_score": 0.4, "bits": 12.0}


@pytest.mark.parametrize(
    ("source_id", "label"),
    [
        ("synthetic", "synthetic simulator"),
        ("synthetic:seed=7", "synthetic simulator"),
        ("video:tank.mp4", "recorded video tank.mp4"),
        ("video", "recorded video"),
        ("camera:0", "live camera 0"),
        ("camera:rtsp://user:hunter2@10.0.0.7/stream", "live camera (network stream)"),
        ("rtsp://user:hunter2@10.0.0.7/stream", "network stream"),
        ("picamera", "live camera (Pi camera)"),
        ("", "unknown"),
    ],
)
def test_source_labels_never_leak_stream_urls(source_id: str, label: str) -> None:
    assert wire.describe_source(source_id) == label
    assert "hunter2" not in wire.describe_source(source_id)


def test_frame_without_jpeg_and_track_identities(snapshot) -> None:
    data = wire.frame_message(snapshot, None, {})
    assert data["jpeg"] is None
    assert wire.frame_message(snapshot, b"", {})["jpeg"] is None
    assert all(t["name"] is None for t in data["tracks"])


def test_identities_from_engine_labels_and_cards() -> None:
    labels = {"f1": FishLabel("Nemo", "#ff8800"), "f2": FishLabel("Dory", "blue"), 3: "x"}
    assert wire.identities_from_labels(labels) == {
        "f1": ("Nemo", "#ff8800"),
        "f2": ("Dory", None),
    }
    assert wire.identities_from_labels(None) == {}
    cards = [{"fish_id": "f1", "name": "Nemo", "color": "#ff8800"}, {"name": "no id"}]
    assert wire.identities_from_cards(cards) == {"f1": ("Nemo", "#ff8800")}


def test_bits_must_be_binary() -> None:
    with pytest.raises(ValueError, match="0/1"):
        wire.bits_message(0.0, "0102")
