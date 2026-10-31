"""End to end: the real engine on the seeded simulator, wired exactly like ``puffer serve``.

``server.running`` builds the engine and beacon with ``puffer.runtime``,
starts the engine thread and the beacon thread, and the app is driven through
the ASGI test client. Every message is checked against the dashboard's
decoders (the Python port always, the real ``protocol.ts`` when Node can run
it). Skipped, with the reason, if the engine modules are not importable.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

pytest.importorskip("puffer.engine", reason="puffer.engine is not available yet")
pytest.importorskip("puffer.runtime", reason="puffer.runtime is not available yet")

from puffer.api import server
from puffer.beacon import Pulse, verify_chain
from puffer.config import PufferConfig
from puffer.crypto import b64url_decode, b64url_encode

from .conftest import TestClient
from .dashboard_protocol import MESSAGE_DECODERS, REST_DECODERS, parse_server_message
from .node_decoder import decode_with_dashboard, node_skip_reason

REPO_DIST = Path(__file__).resolve().parents[2] / "dashboard" / "dist"
STREAM_TYPES = {"hello", "frame", "entropy", "bits", "health", "led", "fish", "beacon"}


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    config = PufferConfig(
        source="synthetic:seed=7",
        data_dir=tmp_path_factory.mktemp("e2e-data"),
        beacon_period_s=1.0,  # a pulse every second so the test sees the push
        static_dir=REPO_DIST,
    )
    with server.running(config) as app, TestClient(app) as client:
        yield client


def _complete(seen: dict[str, list[str]]) -> bool:
    """Every type seen, and the simulator's fish already re-identified and carded."""
    if not set(seen) >= STREAM_TYPES or len(seen["frame"]) < 5:
        return False
    named = any(t["name"] for t in data(seen["frame"][-1])["tracks"])
    return named and bool(data(seen["fish"][-1])["fish"])


@pytest.fixture(scope="module")
def stream(live: TestClient) -> dict[str, list[str]]:
    """Raw messages of one connection, until :func:`_complete` (~2-4 s of simulation)."""
    seen: dict[str, list[str]] = {}
    deadline = time.monotonic() + 30
    with live.websocket_connect("/ws/live") as ws:
        while not _complete(seen):
            assert time.monotonic() < deadline, f"incomplete stream: {sorted(seen)}"
            raw = ws.receive_text()
            message_type, _ = parse_server_message(raw)
            seen.setdefault(message_type, []).append(raw)
    return seen


def data(raw: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(raw)["data"]
    return payload


def test_hello_comes_first_and_pins_the_beacon_key(live: TestClient) -> None:
    with live.websocket_connect("/ws/live") as ws:
        first = json.loads(ws.receive_text())
    assert first["type"] == "hello"
    hello = first["data"]
    MESSAGE_DECODERS["hello"](hello)
    assert hello["source"] == "synthetic simulator"
    assert (hello["width"], hello["height"]) == (640, 480)
    assert 0 < hello["fps"] <= 10
    key = live.get("/api/beacon/public-key").json()["public_key"]
    assert hello["beacon_public_key"] == key


def test_every_stream_message_decodes(stream: dict[str, list[str]]) -> None:
    assert set(stream) == STREAM_TYPES
    reason = node_skip_reason()
    if reason is None:
        messages = [raw for raws in stream.values() for raw in raws[:20]]
        failures = [r for r in decode_with_dashboard(messages) if not r["ok"]]
        assert failures == []


def test_frames_carry_a_real_jpeg_and_named_tracks(stream: dict[str, list[str]]) -> None:
    frames = [data(raw) for raw in stream["frame"]]
    indices = [f["index"] for f in frames]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    frame = frames[-1]
    image = cv2.imdecode(np.frombuffer(base64.b64decode(frame["jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[1] == 640
    named = [t for f in frames for t in f["tracks"] if t["name"]]
    assert named, "re-identified fish should be labelled in the overlay"
    assert all(t["color"] and t["color"].startswith("#") for t in named)
    for track in frame["tracks"]:
        x1, y1, x2, y2 = track["bbox"]
        assert 0 <= x1 < x2 <= frame["width"] + 1
        assert 0 <= y1 < y2 <= frame["height"] + 1


def test_entropy_led_health_and_fish_reflect_the_engine(stream: dict[str, list[str]]) -> None:
    entropy = data(stream["entropy"][-1])
    assert set(entropy["channels"]) == {"sensor", "fish", "system"}
    assert entropy["pool"]["capacity_bits"] == 4096.0
    assert entropy["channels"]["system"]["h_per_sample"] is None
    led = data(stream["led"][-1])
    assert led["state"] in {"calibrating", "verified", "failed"}  # liveness is required here
    assert set(data(stream["health"][-1])["channels"]) >= {"sensor", "fish", "system"}
    cards = data(stream["fish"][-1])["fish"]
    assert cards
    assert all(card["color"].startswith("#") for card in cards)
    assert len(data(stream["bits"][-1])["bits"]) == 256


def test_beacon_push_matches_rest(live: TestClient, stream: dict[str, list[str]]) -> None:
    raw = stream["beacon"][0]
    index = data(raw)["index"]
    rest = live.get(f"/api/beacon/pulse/{index}")
    assert raw == '{"type":"beacon","data":' + rest.text + "}"
    key = bytes.fromhex(live.get("/api/beacon/public-key").json()["public_key"])
    chain = [Pulse.from_json(p) for p in live.get("/api/beacon/pulses").json()]
    assert verify_chain(chain, public_key=key).ok


def test_demo_encrypt_decrypt_and_tamper(live: TestClient) -> None:
    sealed = live.post("/api/demo/encrypt", json={"plaintext": "Fugu at noon 🐡"})
    assert sealed.status_code == 200
    body = sealed.json()
    REST_DECODERS["encrypt"](body)
    assert body["receipt"]["purpose"] == "key"
    opened = live.post(
        "/api/demo/decrypt", json={"ciphertext": body["ciphertext"], "key": body["key"]}
    )
    assert opened.json() == {"plaintext": "Fugu at noon 🐡"}

    blob = bytearray(b64url_decode(body["ciphertext"]))
    blob[-1] ^= 0x80  # flip a tag bit
    tampered = live.post(
        "/api/demo/decrypt", json={"ciphertext": b64url_encode(bytes(blob)), "key": body["key"]}
    )
    assert tampered.status_code == 400
    assert "authentication failed" in tampered.json()["detail"]


def test_rest_endpoints_with_the_real_engine(live: TestClient, stream: dict[str, Any]) -> None:
    status = live.get("/api/status").json()
    REST_DECODERS["status"](status)
    assert status["source"] == "synthetic simulator"
    assert status["frames"] > 0
    REST_DECODERS["fish"](live.get("/api/fish").json())
    REST_DECODERS["leaderboard"](live.get("/api/leaderboard", params={"days": 7}).json())
    REST_DECODERS["random"](live.get("/api/random", params={"n": 64}).json())
    history = live.get("/api/entropy/history", params={"seconds": 60}).json()
    assert history
    for entry in history:
        MESSAGE_DECODERS["entropy"](entry)


@pytest.mark.skipif(not (REPO_DIST / "index.html").is_file(), reason="dashboard/dist not built")
def test_the_built_dashboard_is_served(live: TestClient) -> None:
    index = live.get("/")
    assert index.status_code == 200
    assert '<div id="root">' in index.text
    assert "'sha256-" in index.headers["content-security-policy"]
