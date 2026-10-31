"""Beacon signing key: persistence, RFC 8032 conformance, tamper checks, races."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from nacl.signing import SigningKey, VerifyKey

from puffer.crypto import (
    FormatError,
    InsecurePermissionsWarning,
    RandomBytes,
    b64url_encode,
    load_or_create_signing_key,
    public_key_hex,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")

# RFC 8032 section 7.1, TEST 1.
RFC8032_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
RFC8032_PUBLIC = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
RFC8032_SIGNATURE = (
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
    "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)


def fixed(seed: bytes) -> RandomBytes:
    return lambda n: seed[:n]


def test_create_then_reload_is_the_same_key(tmp_path: Path, det_random: RandomBytes) -> None:
    path = tmp_path / "beacon.key"
    created = load_or_create_signing_key(path, random_bytes=det_random)
    reloaded = load_or_create_signing_key(path, random_bytes=det_random)
    assert public_key_hex(reloaded) == public_key_hex(created)
    assert bytes(reloaded) == bytes(created)
    assert reloaded.sign(b"pulse").signature == created.sign(b"pulse").signature


def test_rfc8032_vector(tmp_path: Path) -> None:
    key = load_or_create_signing_key(tmp_path / "k", random_bytes=fixed(RFC8032_SEED))
    assert public_key_hex(key) == RFC8032_PUBLIC
    assert key.sign(b"").signature.hex() == RFC8032_SIGNATURE
    VerifyKey(bytes.fromhex(RFC8032_PUBLIC)).verify(b"", bytes.fromhex(RFC8032_SIGNATURE))


def test_on_disk_format(tmp_path: Path) -> None:
    path = tmp_path / "beacon.key"
    load_or_create_signing_key(path, random_bytes=fixed(RFC8032_SEED))
    assert json.loads(path.read_text("ascii")) == {
        "format": "puffer-signing-key",
        "version": 1,
        "seed": b64url_encode(RFC8032_SEED),
        "public_key": RFC8032_PUBLIC,
    }


@posix_only
def test_file_is_private_from_creation(tmp_path: Path) -> None:
    previous = os.umask(0)
    try:
        load_or_create_signing_key(tmp_path / "beacon.key")
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "beacon.key").stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["beacon.key"]


def test_public_key_hex_accepts_both_key_types() -> None:
    key = SigningKey(RFC8032_SEED)
    assert public_key_hex(key) == public_key_hex(key.verify_key) == RFC8032_PUBLIC


def test_default_rng_creates_distinct_keys(tmp_path: Path) -> None:
    a = load_or_create_signing_key(tmp_path / "a")
    b = load_or_create_signing_key(tmp_path / "b")
    assert public_key_hex(a) != public_key_hex(b)


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc))
    os.chmod(path, 0o600)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("public_key", "00" * 32, "does not match its seed"),
        ("seed", b64url_encode(bytes(32)), "does not match its seed"),
        ("seed", b64url_encode(bytes(31)), "32 bytes"),
        ("seed", "%%%", "base64url"),
        ("seed", 7, "base64url string"),
        ("public_key", RFC8032_PUBLIC.upper(), "lowercase hex"),
        ("public_key", RFC8032_PUBLIC[:-2], "lowercase hex"),
        ("format", "puffer-key", "not a puffer signing key"),
        ("version", 2, "version"),
        ("version", True, "version"),
        ("extra", 1, "exactly the fields"),
    ],
)
def test_rejects_tampered_files(tmp_path: Path, field: str, value: Any, message: str) -> None:
    path = tmp_path / "beacon.key"
    load_or_create_signing_key(path, random_bytes=fixed(RFC8032_SEED))
    doc = json.loads(path.read_text())
    doc[field] = value
    _write(path, doc)
    with pytest.raises(FormatError, match=message):
        load_or_create_signing_key(path)


def test_missing_field_is_rejected_not_regenerated(tmp_path: Path) -> None:
    path = tmp_path / "beacon.key"
    load_or_create_signing_key(path, random_bytes=fixed(RFC8032_SEED))
    doc = json.loads(path.read_text())
    del doc["seed"]
    _write(path, doc)
    with pytest.raises(FormatError, match="exactly the fields"):
        load_or_create_signing_key(path)
    assert "seed" not in json.loads(path.read_text())


def test_garbage_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "beacon.key"
    path.write_text("not json")
    os.chmod(path, 0o600)
    with pytest.raises(FormatError, match="not valid JSON"):
        load_or_create_signing_key(path)
    assert path.read_text() == "not json"


def test_concurrent_creator_wins(tmp_path: Path) -> None:
    """If another process creates the key first, adopt its identity instead of clobbering."""
    path = tmp_path / "beacon.key"
    winner_seed = bytes(range(32))

    def racing_rng(n: int) -> bytes:
        load_or_create_signing_key(path, random_bytes=fixed(winner_seed))
        return RFC8032_SEED[:n]

    key = load_or_create_signing_key(path, random_bytes=racing_rng)
    assert bytes(key) == winner_seed
    assert public_key_hex(load_or_create_signing_key(path)) == public_key_hex(key)
    assert [p.name for p in tmp_path.iterdir()] == ["beacon.key"]


def test_misbehaving_rng_creates_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="returned 16 bytes"):
        load_or_create_signing_key(tmp_path / "k", random_bytes=lambda n: bytes(16))
    assert list(tmp_path.iterdir()) == []


@posix_only
def test_warns_about_exposed_signing_key(tmp_path: Path) -> None:
    path = tmp_path / "beacon.key"
    load_or_create_signing_key(path)
    os.chmod(path, 0o640)
    with pytest.warns(InsecurePermissionsWarning) as record:
        load_or_create_signing_key(path)
    assert record[0].filename == __file__
