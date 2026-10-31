"""``puffer beacon verify`` / ``latest`` against the shared vectors (files and a local server)."""

from __future__ import annotations

import json
import socket
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from .conftest import Invoke, Tank

VECTORS = json.loads(
    (Path(__file__).resolve().parents[1] / "vectors" / "beacon_vectors.json").read_text()
)
PUBLIC_KEY: str = VECTORS["public_key"]
CHAIN: list[dict[str, Any]] = VECTORS["valid_chain"]
INVALID = {case["name"]: case for case in VECTORS["invalid"]}


def _write(path: str, data: object) -> str:
    Path(path).write_text(json.dumps(data))
    return path


def _tampered_chain(case: dict[str, Any]) -> list[dict[str, Any]]:
    """The valid chain with the vector's pulse in its place (pulse 2, or 0 for genesis)."""
    chain = [dict(p) for p in CHAIN]
    position = 0 if case["prev"] is None else 2
    chain[position] = case["pulse"]
    return chain


# -------------------------------------------------------------------- files


def test_valid_chain_verifies_with_pinned_key(cli: Invoke) -> None:
    result = cli("beacon", "verify", _write("chain.json", CHAIN), "--public-key", PUBLIC_KEY)

    assert result.exit_code == 0, result.output
    assert "✓ verified 5 pulse(s), index 0-4" in result.stdout
    assert "(pinned)" in result.stdout
    assert "Nemo" in result.stdout  # summaries are shown
    assert "trust on first use" not in result.stderr


def test_unpinned_verification_warns_loudly(cli: Invoke) -> None:
    result = cli("beacon", "verify", _write("chain.json", CHAIN))

    assert result.exit_code == 0, result.output
    assert "No --public-key pinned" in result.stderr
    assert "internal consistency only" in result.stderr
    assert "(embedded)" in result.stdout


@pytest.mark.parametrize("name", sorted(INVALID))
def test_every_invalid_vector_fails_with_its_code(cli: Invoke, name: str) -> None:
    case = INVALID[name]

    result = cli(
        "beacon", "verify", _write("bad.json", _tampered_chain(case)), "--public-key", PUBLIC_KEY
    )

    assert result.exit_code == 1, result.output
    assert "NOT verified" in result.stdout
    assert case["expected_error"] in result.stdout


def test_json_report(cli: Invoke) -> None:
    chain = _tampered_chain(INVALID["tampered_summary"])

    result = cli(
        "beacon", "verify", _write("bad.json", chain), "--public-key", PUBLIC_KEY, "--json"
    )

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["pinned"] is True
    assert report["checked"] == 5
    assert {"position": 2, "index": 2, "code": "bad_summary_hash"} in report["errors"]


def test_wrong_pinned_key(cli: Invoke) -> None:
    other = "ab" * 32

    result = cli("beacon", "verify", _write("chain.json", CHAIN), "--public-key", other)

    assert result.exit_code == 1
    assert "wrong_public_key" in result.stdout


def test_single_pulse_file(cli: Invoke) -> None:
    result = cli("beacon", "verify", _write("one.json", CHAIN[3]), "--public-key", PUBLIC_KEY)

    assert result.exit_code == 0, result.output
    assert "1 pulse(s), index 3-3" in result.stdout


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "not a valid JSON document"),
        ('[{"version": 1}]', "pulse #0 is malformed"),
        ('[{"index": 1, "index": 2}]', "duplicate JSON key"),
        ("[]", "contains no pulses"),
    ],
)
def test_malformed_files_do_not_verify(cli: Invoke, content: str, message: str) -> None:
    Path("x.json").write_text(content)

    result = cli("beacon", "verify", "x.json")

    assert result.exit_code == 1
    assert message in result.stderr


def test_missing_file_and_bad_key(cli: Invoke) -> None:
    assert cli("beacon", "verify", "missing.json").exit_code == 10
    result = cli("beacon", "verify", _write("c.json", CHAIN), "--public-key", "xyz")
    assert result.exit_code == 2
    assert "64 hex characters" in result.stderr


# -------------------------------------------------------------------- HTTP


class _Beacon(BaseHTTPRequestHandler):
    """Serves DESIGN.md 7.1 beacon routes from a list of pulses; pages are capped at 2."""

    pulses: ClassVar[list[dict[str, Any]]] = CHAIN
    requests: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        query = dict(urllib.parse.parse_qsl(url.query))
        type(self).requests.append(self.path)
        by_index = {p["index"]: p for p in self.pulses}
        tail = url.path.rsplit("/", 1)[-1]
        body: object
        if url.path == "/api/beacon/pulses":
            start, limit = int(query.get("start", 0)), min(int(query.get("limit", 50)), 2)
            body = [p for p in self.pulses if p["index"] >= start][:limit]
        elif url.path == "/api/beacon/latest" and self.pulses:
            body = self.pulses[-1]
        elif url.path.startswith("/api/beacon/pulse/") and tail.isdigit() and int(tail) in by_index:
            body = by_index[int(tail)]
        else:
            self.send_error(404, "not found")
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:  # keep test output clean
        pass


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, type[_Beacon]]]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    handler = type("Handler", (_Beacon,), {"pulses": list(CHAIN), "requests": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, args=(0.02,), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_verify_whole_chain_from_a_server(cli: Invoke, server: tuple[str, type[_Beacon]]) -> None:
    base, handler = server

    result = cli("beacon", "verify", base, "--public-key", PUBLIC_KEY)

    assert result.exit_code == 0, result.output
    assert "✓ verified 5 pulse(s)" in result.stdout
    pages = [r for r in handler.requests if r.startswith("/api/beacon/pulses")]
    assert len(pages) == 4  # 2 + 2 + 1 + the empty page that ends the chain


def test_max_pulses_truncates_with_a_note(cli: Invoke, server: tuple[str, type[_Beacon]]) -> None:
    base, _ = server

    result = cli("beacon", "verify", base, "--public-key", PUBLIC_KEY, "--max-pulses", "3")

    assert result.exit_code == 0, result.output
    assert "3 pulse(s), index 0-2" in result.stdout
    assert "only the first 3 pulses were checked" in result.stderr


def test_verify_a_tampered_server(cli: Invoke, server: tuple[str, type[_Beacon]]) -> None:
    base, handler = server
    handler.pulses = _tampered_chain(INVALID["tampered_prev_output"])

    result = cli("beacon", "verify", base, "--public-key", PUBLIC_KEY)

    assert result.exit_code == 1
    assert "broken_link" in result.stdout


def test_verify_a_single_pulse_url(cli: Invoke, server: tuple[str, type[_Beacon]]) -> None:
    base, _ = server

    result = cli("beacon", "verify", f"{base}/api/beacon/pulse/4", "--public-key", PUBLIC_KEY)

    assert result.exit_code == 0, result.output
    assert "index 4-4" in result.stdout


def test_latest_verifies_the_link_to_its_predecessor(
    cli: Invoke, server: tuple[str, type[_Beacon]]
) -> None:
    base, _ = server

    result = cli("beacon", "latest", "--url", base, "--public-key", PUBLIC_KEY)

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "Beacon pulse 4" in out
    assert "linked to pulse 3" in out
    assert CHAIN[4]["output"][:64] in out
    assert "never use it as a secret key" in out


def test_latest_json_and_tampering(cli: Invoke, server: tuple[str, type[_Beacon]]) -> None:
    base, handler = server
    handler.pulses = [*CHAIN[:4], {**CHAIN[4], "summary": CHAIN[3]["summary"]}]

    result = cli("beacon", "latest", "--url", base, "--json")

    assert result.exit_code == 1
    document = json.loads(result.stdout)
    assert document["ok"] is False
    assert "bad_summary_hash" in document["errors"]
    assert document["link_checked"] is True
    assert document["pinned"] is False
    assert "No --public-key pinned" in result.stderr


def test_latest_on_an_empty_beacon(cli: Invoke, server: tuple[str, type[_Beacon]]) -> None:
    base, handler = server
    handler.pulses = []

    result = cli("beacon", "latest", "--url", base)

    assert result.exit_code == 1
    assert "has not published a pulse yet" in result.stderr


def test_latest_defaults_to_the_configured_server(
    cli: Invoke, tank: Tank, server: tuple[str, type[_Beacon]]
) -> None:
    base, _ = server
    tank.config.port = int(base.rsplit(":", 1)[1])

    result = cli("beacon", "latest", "--public-key", PUBLIC_KEY)

    assert result.exit_code == 0, result.output


def test_unreachable_server_and_bad_scheme(cli: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # closed again: nothing listens there

    result = cli("beacon", "latest", "--url", f"http://127.0.0.1:{port}")
    assert result.exit_code == 10
    assert "cannot reach" in result.stderr

    result = cli("beacon", "latest", "--url", "file:///etc/passwd")
    assert result.exit_code == 2
    assert "not an http(s) URL" in result.stderr
