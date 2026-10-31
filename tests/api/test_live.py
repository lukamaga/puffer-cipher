"""``/ws/live`` through the ASGI app (DESIGN.md 7.2): order, shapes, beacon, limits."""

from __future__ import annotations

import base64
import json
import math
import time
from dataclasses import replace

import pytest
from starlette.websockets import WebSocketDisconnect

from puffer.beacon import BeaconService

from .conftest import BEACON_PERIOD_MS, FAST, ClientFactory, ManualClock, receive_until
from .dashboard_protocol import MESSAGE_DECODERS, parse_server_message
from .fakes import FAKE_JPEG, FakeEngine

ALL_TYPES = {"hello", "frame", "entropy", "bits", "health", "led", "fish"}


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


def test_hello_is_the_first_message_on_every_connection(client) -> None:
    for _ in range(3):
        with client.websocket_connect("/ws/live") as a, client.websocket_connect("/ws/live") as b:
            for ws in (a, b):
                message = json.loads(ws.receive_text())
                assert message["type"] == "hello"
                MESSAGE_DECODERS["hello"](message["data"])


def test_hello_describes_source_and_stream(client) -> None:
    with client.websocket_connect("/ws/live") as ws:
        hello = json.loads(ws.receive_text())["data"]
    assert hello == {
        "version": "0.1.0",
        "source": "synthetic simulator",
        "width": 640,
        "height": 360,
        "fps": 10.0,  # measured 9.97 fps rounded to 0.1 (the header shows it verbatim)
        "beacon_public_key": None,
    }


def test_every_message_type_arrives_and_decodes(client) -> None:
    seen: dict[str, dict] = {}
    with client.websocket_connect("/ws/live") as ws:
        for _ in range(2000):
            raw = ws.receive_text()
            message_type, _ = parse_server_message(raw)  # raises on any contract violation
            seen.setdefault(message_type, json.loads(raw)["data"])
            if set(seen) >= ALL_TYPES:
                break
    assert set(seen) == ALL_TYPES

    entropy = seen["entropy"]
    assert entropy["pool"] == {"sensor": 2048.0, "fish": 77.5, "capacity_bits": 4096.0}
    assert entropy["channels"]["system"]["h_per_sample"] is None  # NaN upstream
    assert entropy["channels"]["sensor"]["bits_per_s"] == 9876.543
    assert len(seen["bits"]["bits"]) == 256
    assert seen["health"]["overall"] == "degraded"
    assert seen["health"]["channels"]["fish"]["checks"]["fish_presence"] == "degraded"
    assert [card["name"] for card in seen["fish"]["fish"]] == ["Nemo", "Dory"]


def test_led_message_keeps_undecodable_bits(client) -> None:
    with client.websocket_connect("/ws/live") as ws:
        led = receive_until(ws, "led")["data"]
    assert led["state"] == "verified"
    assert led["expected"] == "01101"
    assert led["observed"] == "01?01"


def test_frame_carries_jpeg_and_confirmed_tracks_with_identities(client) -> None:
    with client.websocket_connect("/ws/live") as ws:
        for _ in range(200):
            frame = receive_until(ws, "frame")["data"]
            if any(t["name"] for t in frame["tracks"]):
                break
    assert base64.b64decode(frame["jpeg"]) == FAKE_JPEG
    assert (frame["width"], frame["height"]) == (640, 360)
    tracks = {t["track_id"]: t for t in frame["tracks"]}
    assert set(tracks) == {3, 7}  # the tentative track 9 is not shown
    nemo = tracks[3]
    assert (nemo["fish_id"], nemo["name"], nemo["color"]) == ("f1", "Nemo", "#ff8800")
    assert nemo["bbox"] == [100.0, 50.0, 160.0, 90.0]
    assert nemo["trail"] == [[120.0, 60.0], [125.0, 65.0], [130.0, 70.0]]
    assert nemo["speed"] == 50.0
    assert nemo["heading"] == pytest.approx(math.atan2(40, 30), abs=1e-4)
    unknown = tracks[7]
    assert (unknown["fish_id"], unknown["name"], unknown["color"]) == (None, None, None)


def test_frames_use_the_configured_stream_size(make_client, engine: FakeEngine) -> None:
    client = make_client(settings=replace(FAST, stream_width=320, jpeg_quality=55))
    with client.websocket_connect("/ws/live") as ws:
        receive_until(ws, "frame")
    assert engine.jpeg_args[0] == (320, 55)


def test_new_clients_get_entropy_history_then_latest_state(client) -> None:
    with client.websocket_connect("/ws/live") as ws:  # let history accumulate
        for _ in range(60):
            ws.receive_text()
    with client.websocket_connect("/ws/live") as ws:
        welcome = [json.loads(ws.receive_text()) for _ in range(8)]
    assert welcome[0]["type"] == "hello"
    replay = [m for m in welcome[1:] if m["type"] == "entropy"]
    assert len(replay) >= 3
    ts = [m["data"]["t"] for m in replay]
    assert ts == sorted(ts)


def test_engine_without_frames_still_says_hello(make_client) -> None:
    engine = FakeEngine(ready=False)
    client = make_client(engine=engine, settings=replace(FAST, hello_wait_s=0.05))
    with client.websocket_connect("/ws/live") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello"
        assert (hello["data"]["width"], hello["data"]["height"]) == (0, 0)
        assert hello["data"]["source"] == "synthetic simulator"  # from the configured kind
        engine.advance()  # the first frame arrives: the stream starts
        frame = receive_until(ws, "frame")["data"]
        assert frame["index"] >= 1


def test_beacon_pulses_are_pushed_byte_identical_to_rest(
    make_client, beacon: BeaconService, beacon_clock: ManualClock
) -> None:
    client = make_client(beacon_service=beacon)
    with client.websocket_connect("/ws/live") as ws:
        hello = json.loads(ws.receive_text())["data"]
        assert hello["beacon_public_key"] == beacon.public_key.hex()
        for _ in range(2):
            beacon.tick()
            beacon_clock.now_ms += BEACON_PERIOD_MS
            message = receive_until(ws, "beacon")
            rest = client.get(f"/api/beacon/pulse/{message['data']['index']}")
            assert json.dumps(message, separators=(",", ":"), ensure_ascii=False) == (
                '{"type":"beacon","data":' + rest.text + "}"
            )
    assert message["data"]["index"] == 1


def test_client_messages_are_ignored(client) -> None:
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_text()
        ws.send_text("hello server")
        ws.send_bytes(b"\x00\x01")
        receive_until(ws, "frame")


def test_disconnected_clients_are_removed_and_frames_stop(client, engine: FakeEngine) -> None:
    hub = client.app.state.hub
    with client.websocket_connect("/ws/live") as ws:
        receive_until(ws, "frame")
        assert hub.client_count == 1
    _wait_for(lambda: hub.client_count == 0)
    calls = engine.jpeg_calls
    time.sleep(0.15)
    assert engine.jpeg_calls == calls  # nobody watching: no JPEG encoding


def test_no_frames_are_encoded_without_viewers(client, engine: FakeEngine) -> None:
    time.sleep(0.2)
    assert engine.jpeg_calls == 0
    assert client.get("/api/entropy/history").json()  # but history keeps recording


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        (None, True),  # non-browser client
        ("http://testserver", True),  # same origin
        ("http://localhost:5173", True),  # Vite dev server proxy
        ("https://evil.example", False),
        ("null", False),
        ("http://testserver.evil.example", False),
    ],
)
def test_cross_site_websocket_connections_are_refused(client, origin, allowed) -> None:
    headers = {"origin": origin} if origin else {}
    if allowed:
        with client.websocket_connect("/ws/live", headers=headers) as ws:
            assert json.loads(ws.receive_text())["type"] == "hello"
    else:
        with (
            pytest.raises(WebSocketDisconnect) as excinfo,
            client.websocket_connect("/ws/live", headers=headers),
        ):
            pass
        assert excinfo.value.code == 1008


def test_client_limit(make_client: ClientFactory) -> None:
    client = make_client(settings=replace(FAST, max_clients=1))
    with client.websocket_connect("/ws/live") as first:
        assert json.loads(first.receive_text())["type"] == "hello"
        with client.websocket_connect("/ws/live") as second:
            with pytest.raises(WebSocketDisconnect) as excinfo:
                second.receive_text()
            assert excinfo.value.code == 1013


def test_lifespan_unsubscribes_from_engine_and_beacon(
    make_client, engine: FakeEngine, beacon: BeaconService, beacon_clock: ManualClock
) -> None:
    client = make_client(beacon_service=beacon)
    assert engine.subscriber_count == 1
    client.__exit__(None, None, None)
    assert engine.subscriber_count == 0
    beacon.tick()  # no listener left: must not touch the stopped hub
    assert beacon.store.count() == 1
