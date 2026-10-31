"""REST endpoints of DESIGN.md 7.1: shapes, validation, error mapping and limits."""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import pytest

from puffer.beacon import BeaconService, BeaconStore, Pulse, verify_chain
from puffer.crypto import (
    b64url_decode,
    b64url_encode,
    decrypt_message_b64,
    encrypt_message_b64,
    key_id,
)

from .conftest import BEACON_PERIOD_MS, FAST, ManualClock
from .dashboard_protocol import MESSAGE_DECODERS, REST_DECODERS
from .fakes import FakeEngine

# --------------------------------------------------------------------------
# /status
# --------------------------------------------------------------------------


def test_status_matches_the_dashboard_decoder(client) -> None:
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    REST_DECODERS["status"](body)
    assert body["source"] == "synthetic simulator"
    assert body["mode"] == "day"
    assert body["frames"] >= 1
    assert body["pool"] == {"sensor": 2048.0, "fish": 77.5}
    assert body["health"]["fish"]["channel"] == "fish"
    assert body["health"]["fish"]["checks"]["fish_presence"] == "degraded"
    assert body["health"]["fish"]["reasons"] == ["fewer fish than expected"]


def test_status_before_the_first_frame_is_503_with_retry_after(make_client) -> None:
    client = make_client(engine=FakeEngine(ready=False))
    response = client.get("/api/status")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert isinstance(response.json()["detail"], str)


# --------------------------------------------------------------------------
# /random
# --------------------------------------------------------------------------


def test_random_default_is_32_hex_bytes_with_a_receipt(client, engine: FakeEngine) -> None:
    response = client.get("/api/random")
    assert response.status_code == 200
    body = response.json()
    REST_DECODERS["random"](body)
    assert len(bytes.fromhex(body["data"])) == 32
    assert body["receipt"]["purpose"] == "random"
    assert body["receipt"]["bytes"] == 32
    assert body["receipt"]["frames"] is None  # nothing absorbed: null, not []
    assert engine.rng.calls == [("generate", 32, "random")]
    assert response.headers["cache-control"] == "no-store"


def test_random_base64_and_bounds(client) -> None:
    for n in (1, 1024):
        body = client.get("/api/random", params={"n": n, "format": "base64"}).json()
        assert len(base64.b64decode(body["data"], validate=True)) == n
    first = client.get("/api/random", params={"n": 16}).json()["data"]
    second = client.get("/api/random", params={"n": 16}).json()["data"]
    assert first != second


@pytest.mark.parametrize(
    "params",
    [
        {"n": 0},
        {"n": 1025},
        {"n": -1},
        {"n": "lots"},
        {"n": 1.5},
        {"format": "raw"},
        {"format": "HEX"},
    ],
)
def test_random_rejects_invalid_parameters(client, engine: FakeEngine, params) -> None:
    response = client.get("/api/random", params=params)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert engine.rng.calls == []  # validation happens before any entropy is spent


def test_random_receipt_reports_camera_credit_and_fish(client, engine: FakeEngine) -> None:
    engine.credit(sensor_bits=800.0, fish_bits=300.0)
    receipt = client.get("/api/random").json()["receipt"]
    REST_DECODERS["random"]({"data": "", "receipt": receipt})
    assert receipt["frames"] == [10, 12]
    assert receipt["channels"]["sensor"]["credited_bits"] == 800.0
    assert receipt["channels"]["fish"]["credited_bits"] == 300.0
    assert receipt["mode"] == "day"
    assert [f["name"] for f in receipt["fish"]] == ["Nemo", "Dory"]


# --------------------------------------------------------------------------
# /demo/encrypt and /demo/decrypt
# --------------------------------------------------------------------------


def _encrypt(client, plaintext: str = "Meet me by the coral reef at noon."):
    response = client.post("/api/demo/encrypt", json={"plaintext": plaintext})
    assert response.status_code == 200, response.text
    return response.json()


def test_encrypt_returns_design_3_3_wire_format(client, engine: FakeEngine) -> None:
    engine.credit()
    body = _encrypt(client)
    REST_DECODERS["encrypt"](body)
    key = b64url_decode(body["key"])
    assert len(key) == 32
    assert body["key_id"] == key_id(key)
    # Independent check with the crypto layer: nonce || ciphertext, aad puffer-msg-v1.
    assert decrypt_message_b64(body["ciphertext"], key) == b"Meet me by the coral reef at noon."
    assert len(b64url_decode(body["ciphertext"])) == 24 + 34 + 16
    # The key draws the pool (and carries the fish credit); the nonce is public.
    assert engine.rng.methods() == ["generate_key", "nonce"]
    assert body["receipt"]["purpose"] == "key"
    assert body["receipt"]["channels"]["fish"]["credited_bits"] == 300.0


def test_encrypt_decrypt_round_trip_with_unicode(client) -> None:
    text = "🐡 Fugu, ünïcödé and \u0000 control"
    sealed = _encrypt(client, text)
    response = client.post(
        "/api/demo/decrypt", json={"ciphertext": sealed["ciphertext"], "key": sealed["key"]}
    )
    assert response.status_code == 200
    assert response.json() == {"plaintext": text}
    REST_DECODERS["decrypt"](response.json())


def test_every_encryption_uses_a_fresh_key_and_nonce(client) -> None:
    a, b = _encrypt(client, "same"), _encrypt(client, "same")
    assert a["key"] != b["key"]
    assert b64url_decode(a["ciphertext"])[:24] != b64url_decode(b["ciphertext"])[:24]


def test_plaintext_limit_is_16_kib_of_utf8(client) -> None:
    assert client.post("/api/demo/encrypt", json={"plaintext": "a" * 16384}).status_code == 200
    too_long = client.post("/api/demo/encrypt", json={"plaintext": "a" * 16385})
    assert too_long.status_code == 413
    assert "16384" in too_long.json()["detail"]
    # 6000 characters but 18000 bytes of UTF-8: the limit is on bytes.
    assert client.post("/api/demo/encrypt", json={"plaintext": "€" * 6000}).status_code == 413


@pytest.mark.parametrize(
    "body",
    [b"{}", b'{"plaintext": 42}', b"[]", b"not json", b'{"plaintext": "\\ud800"}'],
)
def test_encrypt_rejects_malformed_bodies(client, engine: FakeEngine, body: bytes) -> None:
    response = client.post(
        "/api/demo/encrypt", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert "generate_key" not in engine.rng.methods()


def _tamper(token: str) -> str:
    blob = bytearray(b64url_decode(token))
    blob[30] ^= 0x01
    return b64url_encode(bytes(blob))


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        (lambda s: {**s, "ciphertext": _tamper(s["ciphertext"])}, "authentication failed"),
        (lambda s: {**s, "key": b64url_encode(bytes(32))}, "authentication failed"),
        (lambda s: {**s, "key": b64url_encode(bytes(16))}, "key must be 32 bytes"),
        (lambda s: {**s, "key": "not base64!"}, "key is not valid base64url"),
        (lambda s: {**s, "ciphertext": "a+b/"}, "ciphertext is not valid base64url"),
        (lambda s: {**s, "ciphertext": b64url_encode(bytes(10))}, "too short"),
    ],
)
def test_decrypt_failures_are_400_with_a_string_detail(client, change, fragment: str) -> None:
    sealed = _encrypt(client, "secret")
    body = change({"ciphertext": sealed["ciphertext"], "key": sealed["key"]})
    response = client.post("/api/demo/decrypt", json=body)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert fragment in detail
    assert sealed["key"] not in detail  # never echo the secret back


def test_decrypt_of_non_utf8_plaintext_is_400(client) -> None:
    key = bytes(range(32))
    token = encrypt_message_b64(b"\xff\xfe binary", key)
    response = client.post(
        "/api/demo/decrypt", json={"ciphertext": token, "key": b64url_encode(key)}
    )
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_decrypt_requires_both_fields(client) -> None:
    assert client.post("/api/demo/decrypt", json={"ciphertext": "AAAA"}).status_code == 422
    assert client.post("/api/demo/decrypt", json={"key": "AAAA"}).status_code == 422


def test_oversized_bodies_are_413_before_parsing(make_client) -> None:
    client = make_client(settings=replace(FAST, max_body_bytes=1024))
    declared = client.post("/api/demo/encrypt", json={"plaintext": "x" * 2000})
    assert declared.status_code == 413
    assert declared.json()["detail"] == "request body is limited to 1024 bytes"
    assert declared.headers["x-content-type-options"] == "nosniff"

    def chunks():  # no Content-Length: counted while streaming
        yield b'{"plaintext": "'
        for _ in range(20):
            yield b"y" * 100
        yield b'"}'

    streamed = client.post(
        "/api/demo/encrypt", content=chunks(), headers={"content-type": "application/json"}
    )
    assert streamed.status_code == 413


# --------------------------------------------------------------------------
# Rate limits
# --------------------------------------------------------------------------


def test_random_and_demo_are_rate_limited_per_client(make_client) -> None:
    client = make_client(settings=replace(FAST, rate_per_s=0.001, rate_burst=3))
    for _ in range(3):
        assert client.get("/api/random", params={"n": 1}).status_code == 200
    limited = client.get("/api/random", params={"n": 1})
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    assert isinstance(limited.json()["detail"], str)
    # Separate bucket for the demo, and unlimited read-only endpoints.
    assert _encrypt(client)["key"]
    for _ in range(5):
        assert client.get("/api/status").status_code == 200


def test_rate_limit_refills_with_time(make_client) -> None:
    now = [0.0]
    client = make_client(settings=replace(FAST, rate_per_s=1.0, rate_burst=1), clock=lambda: now[0])
    assert client.get("/api/random", params={"n": 1}).status_code == 200
    assert client.get("/api/random", params={"n": 1}).status_code == 429
    now[0] += 1.0
    assert client.get("/api/random", params={"n": 1}).status_code == 200


# --------------------------------------------------------------------------
# /fish, /leaderboard
# --------------------------------------------------------------------------


def test_fish_cards_match_the_dashboard_decoder(client) -> None:
    body = client.get("/api/fish").json()
    REST_DECODERS["fish"](body)
    assert [card["name"] for card in body] == ["Nemo", "Dory"]
    assert body[1]["track_id"] is None
    assert body[0]["color"] == "#ff8800"


def test_leaderboard_days_parameter(client) -> None:
    week = client.get("/api/leaderboard").json()
    REST_DECODERS["leaderboard"](week)
    assert [row["rank"] for row in week] == [1, 2]
    assert week[0]["fish_id"] == "f1"
    month = client.get("/api/leaderboard", params={"days": 28}).json()
    assert month[0]["bits"] == pytest.approx(4 * week[0]["bits"])
    for days in (0, 366, "week"):
        assert client.get("/api/leaderboard", params={"days": days}).status_code == 422


# --------------------------------------------------------------------------
# /entropy/history
# --------------------------------------------------------------------------


def test_entropy_history_is_ascending_entropy_data(client) -> None:
    with client.websocket_connect("/ws/live") as ws:  # let the hub run a few ticks
        for _ in range(30):
            ws.receive_text()
    history = client.get("/api/entropy/history", params={"seconds": 60}).json()
    assert len(history) >= 2
    ts = [entry["t"] for entry in history]
    assert ts == sorted(ts)
    assert len(set(ts)) == len(ts)
    for entry in history:
        MESSAGE_DECODERS["entropy"](entry)
        assert entry["pool"]["capacity_bits"] == 4096.0


@pytest.mark.parametrize("seconds", [0, 601, "all"])
def test_entropy_history_window_is_bounded(client, seconds) -> None:
    assert client.get("/api/entropy/history", params={"seconds": seconds}).status_code == 422


# --------------------------------------------------------------------------
# Beacon
# --------------------------------------------------------------------------


def _tick(beacon: BeaconService, clock: ManualClock, n: int) -> list[Pulse]:
    pulses = []
    for _ in range(n):
        pulses.append(beacon.tick())
        clock.now_ms += BEACON_PERIOD_MS
    return pulses


def test_beacon_endpoints_when_disabled(client) -> None:
    for path in (
        "/api/beacon/public-key",
        "/api/beacon/latest",
        "/api/beacon/pulses",
        "/api/beacon/pulse/0",
    ):
        response = client.get(path)
        assert response.status_code == 404, path
        assert isinstance(response.json()["detail"], str)


def test_beacon_before_the_first_pulse(make_client, beacon: BeaconService) -> None:
    client = make_client(beacon_service=beacon)
    key = client.get("/api/beacon/public-key").json()
    REST_DECODERS["publicKey"](key)
    assert key == {"algorithm": "Ed25519", "public_key": beacon.public_key.hex()}
    assert client.get("/api/beacon/latest").status_code == 404
    assert client.get("/api/beacon/pulses").json() == []


def test_beacon_pulses_are_served_verbatim(
    make_client, beacon: BeaconService, beacon_clock: ManualClock
) -> None:
    client = make_client(beacon_service=beacon)
    pulses = _tick(beacon, beacon_clock, 3)

    latest = client.get("/api/beacon/latest")
    assert latest.headers["content-type"] == "application/json"
    REST_DECODERS["pulse"](latest.json())
    assert latest.json() == pulses[-1].to_json()

    page = client.get("/api/beacon/pulses", params={"start": 1, "limit": 2}).json()
    REST_DECODERS["pulses"](page)
    assert [p["index"] for p in page] == [1, 2]
    one = client.get("/api/beacon/pulse/1").json()
    assert one == pulses[1].to_json()
    assert client.get("/api/beacon/pulse/99").status_code == 404

    # What the API serves is a chain anyone can verify offline.
    chain = [Pulse.from_json(p) for p in client.get("/api/beacon/pulses").json()]
    report = verify_chain(chain, public_key=beacon.public_key)
    assert report.ok, report


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/api/beacon/pulses", {"limit": 0}),
        ("/api/beacon/pulses", {"limit": 1001}),
        ("/api/beacon/pulses", {"start": -1}),
        ("/api/beacon/pulses", {"start": 2**53}),
        ("/api/beacon/pulse/-1", {}),
        ("/api/beacon/pulse/x", {}),
    ],
)
def test_beacon_parameters_are_validated(make_client, beacon, path, params) -> None:
    client = make_client(beacon_service=beacon)
    assert client.get(path, params=params).status_code == 422


def test_beacon_limit_cap_is_1000(make_client, beacon) -> None:
    client = make_client(beacon_service=beacon)
    assert client.get("/api/beacon/pulses", params={"limit": 1000}).status_code == 200


def test_beacon_store_without_a_service_is_a_read_only_mirror(
    make_client, beacon: BeaconService, beacon_clock: ManualClock
) -> None:
    _tick(beacon, beacon_clock, 2)
    mirror = BeaconStore()
    for pulse in beacon.store.range():
        mirror.append(pulse)
    client = make_client(beacon_store=mirror)
    assert client.get("/api/beacon/public-key").json()["public_key"] == beacon.public_key.hex()
    assert client.get("/api/beacon/latest").json()["index"] == 1
    mirror.close()


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------


def test_unknown_api_paths_are_json_404(client) -> None:
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_openapi_is_published_but_the_cdn_docs_ui_is_not(client) -> None:
    schema = client.get("/api/openapi.json")
    assert schema.status_code == 200
    paths = json.loads(schema.text)["paths"]
    assert {"/api/status", "/api/random", "/api/demo/encrypt", "/api/entropy/history"} <= set(paths)
    assert client.get("/docs").status_code == 404
