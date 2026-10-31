"""Everything the server sends, decoded by the dashboard's own ``protocol.ts`` under Node.

``dashboard_protocol.py`` is a Python port of those decoders; this test is the
ground truth that the port cannot drift from, run whenever a Node that can
import TypeScript natively is available (skipped otherwise, e.g. older CI
images).
"""

from __future__ import annotations

import json

from puffer.beacon import BeaconService

from .conftest import BEACON_PERIOD_MS, ManualClock
from .node_decoder import decode_with_dashboard, requires_node

pytestmark = requires_node


def test_the_harness_rejects_contract_violations() -> None:
    bad = json.dumps({"type": "bits", "data": {"t": 1, "bits": "0?1"}})
    good = json.dumps({"type": "bits", "data": {"t": 1, "bits": "011"}})
    unknown = json.dumps({"type": "future-message", "data": {}})
    results = decode_with_dashboard([bad, good, unknown], [("decrypt", {"plaintext": 1})])
    assert [r["ok"] for r in results] == [False, True, True, False]


def test_every_server_message_and_rest_body_decodes(
    make_client, beacon: BeaconService, beacon_clock: ManualClock, engine
) -> None:
    client = make_client(beacon_service=beacon)
    engine.credit()
    wanted = {"hello", "frame", "entropy", "bits", "health", "led", "fish", "beacon"}
    collected: dict[str, str] = {}
    with client.websocket_connect("/ws/live") as ws:
        collected["hello"] = ws.receive_text()  # once hello arrived, we are registered
        beacon.tick()
        beacon_clock.now_ms += BEACON_PERIOD_MS
        for _ in range(3000):
            raw = ws.receive_text()
            collected.setdefault(json.loads(raw)["type"], raw)
            if wanted <= set(collected):
                break
    assert set(collected) == wanted

    beacon.tick()
    sealed = client.post("/api/demo/encrypt", json={"plaintext": "fish"}).json()
    rest = [
        ("status", client.get("/api/status").json()),
        ("random", client.get("/api/random", params={"n": 8, "format": "base64"}).json()),
        ("encrypt", sealed),
        ("decrypt", client.post("/api/demo/decrypt", json=sealed).json()),
        ("fish", client.get("/api/fish").json()),
        ("leaderboard", client.get("/api/leaderboard").json()),
        ("publicKey", client.get("/api/beacon/public-key").json()),
        ("pulse", client.get("/api/beacon/latest").json()),
        ("pulses", client.get("/api/beacon/pulses").json()),
    ]
    results = decode_with_dashboard(list(collected.values()), rest)
    failures = [r for r in results if not r["ok"]]
    assert failures == []
