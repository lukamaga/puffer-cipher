"""Serving the dashboard: caching, reserved paths, CSP and security headers."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from puffer.api.security import content_security_policy, inline_script_hashes
from puffer.api.static import IMMUTABLE, REVALIDATE, cache_control_for

from .conftest import FAST, ClientFactory

THEME_SCRIPT = "\n      try { document.documentElement.dataset.theme = 'dark'; } catch {}\n    "
REPO_DIST = Path(__file__).resolve().parents[2] / "dashboard" / "dist"


def _sha256(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'"


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(
        "<!doctype html><html><head>"
        f"<script>{THEME_SCRIPT}</script>"
        '<script type="module" crossorigin src="/assets/index-BQJl5FOn.js"></script>'
        '<link rel="stylesheet" href="/assets/index-KtZQbutl.css">'
        '</head><body><div id="root"></div></body></html>',
        encoding="utf-8",
    )
    (root / "assets" / "index-BQJl5FOn.js").write_text("console.log('fish')")
    (root / "assets" / "index-KtZQbutl.css").write_text("body{}")
    (root / "favicon.svg").write_text("<svg/>")
    (root / "api").mkdir()
    (root / "api" / "status").write_text("a file that must never shadow the API")
    return root


def test_index_is_served_with_revalidation_and_a_hash_pinned_csp(
    make_client: ClientFactory, dist: Path
) -> None:
    client = make_client(static_dir=dist)
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == REVALIDATE
    csp = response.headers["content-security-policy"]
    assert f"script-src 'self' {_sha256(THEME_SCRIPT)};" in csp
    assert "style-src 'self';" in csp
    assert "'unsafe-inline'" not in csp
    assert "connect-src 'self' ws: wss:" in csp
    assert "img-src 'self' data: blob:" in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"
    assert client.get("/index.html").status_code == 200


def test_hashed_assets_are_immutable_and_others_revalidate(
    make_client: ClientFactory, dist: Path
) -> None:
    client = make_client(static_dir=dist)
    js = client.get("/assets/index-BQJl5FOn.js")
    assert js.status_code == 200
    assert js.headers["cache-control"] == IMMUTABLE
    assert "javascript" in js.headers["content-type"]
    assert client.get("/assets/index-KtZQbutl.css").headers["cache-control"] == IMMUTABLE
    assert client.get("/favicon.svg").headers["cache-control"] == REVALIDATE


def test_revalidation_returns_304_with_the_same_policy(
    make_client: ClientFactory, dist: Path
) -> None:
    client = make_client(static_dir=dist)
    etag = client.get("/").headers["etag"]
    cached = client.get("/", headers={"if-none-match": etag})
    assert cached.status_code == 304
    assert cached.headers["cache-control"] == REVALIDATE


def test_api_routes_win_and_reserved_paths_never_hit_the_disk(
    make_client: ClientFactory, dist: Path
) -> None:
    client = make_client(static_dir=dist)
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["source"] == "synthetic simulator"
    missing = client.get("/api/does-not-exist")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Not Found"}
    assert missing.headers["cache-control"] == "no-store"
    with client.websocket_connect("/ws/live") as ws:  # the mount does not shadow the socket
        assert ws.receive_json()["type"] == "hello"


@pytest.mark.parametrize("path", ["/nope.js", "/../pyproject.toml", "/%2e%2e/pyproject.toml"])
def test_missing_files_and_traversal_are_json_404(
    make_client: ClientFactory, dist: Path, path: str
) -> None:
    response = make_client(static_dir=dist).get(path)
    assert response.status_code == 404
    assert "fish" not in response.text


def test_static_files_are_read_only(make_client: ClientFactory, dist: Path) -> None:
    assert make_client(static_dir=dist).post("/").status_code == 405


@pytest.mark.parametrize("static_dir", [None, "missing"])
def test_without_a_build_the_root_returns_a_json_hint(
    make_client: ClientFactory, tmp_path: Path, static_dir: str | None
) -> None:
    client = make_client(static_dir=tmp_path / static_dir if static_dir else None)
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "npm run build" in body["hint"]
    assert body["api"] == "/api/status"
    assert "script-src 'self';" in response.headers["content-security-policy"]


def test_inline_styles_can_be_allowed_explicitly(make_client: ClientFactory, dist: Path) -> None:
    client = make_client(static_dir=dist, settings=replace(FAST, csp_inline_styles=True))
    csp = client.get("/").headers["content-security-policy"]
    assert "style-src 'self' 'unsafe-inline';" in csp
    assert "script-src 'self' 'sha256-" in csp


def test_inline_script_hashes_skip_external_scripts() -> None:
    html = (
        '<script src="/a.js"></script><SCRIPT type="module">one()</SCRIPT>'
        "<script>two()</script><script>one()</script>"
    )
    assert inline_script_hashes(html) == [_sha256("one()"), _sha256("two()")]


def test_policy_shape() -> None:
    policy = content_security_policy(["'sha256-x'"])
    directives = dict(d.split(" ", 1) for d in policy.split("; "))
    assert directives["default-src"] == "'self'"
    assert directives["script-src"] == "'self' 'sha256-x'"
    assert directives["object-src"] == "'none'"
    assert directives["base-uri"] == "'none'"


@pytest.mark.parametrize(
    ("path", "policy"),
    [
        ("index.html", REVALIDATE),
        ("assets/index-BQJl5FOn.js", IMMUTABLE),
        ("assets/index-BQJl5FOn.js.map", REVALIDATE),
        ("assets/logo.svg", REVALIDATE),
        ("favicon.svg", REVALIDATE),
    ],
)
def test_cache_policy_by_path(path: str, policy: str) -> None:
    assert cache_control_for(path) == policy


@pytest.mark.skipif(not (REPO_DIST / "index.html").is_file(), reason="dashboard/dist not built")
def test_the_real_dashboard_build_gets_its_theme_script_pinned(make_client: ClientFactory) -> None:
    client = make_client(static_dir=REPO_DIST)
    index = client.get("/")
    assert index.status_code == 200
    (expected,) = inline_script_hashes(index.text)
    assert expected in index.headers["content-security-policy"]
    for asset in REPO_DIST.joinpath("assets").glob("index-*.js"):
        response = client.get(f"/assets/{asset.name}")
        assert response.headers["cache-control"] == IMMUTABLE
