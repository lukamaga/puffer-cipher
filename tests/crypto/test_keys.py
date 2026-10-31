"""Key ids and key files: format, permissions, overwrite protection, tamper checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import warnings
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from puffer.crypto import (
    KEY_BYTES,
    FormatError,
    InsecurePermissionsWarning,
    KeyFile,
    b64url_encode,
    key_id,
    load_key,
    save_key,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")

RECEIPT = {
    "receipt_version": 1,
    "purpose": "key",
    "bytes": 32,
    "channels": {"fish": {"credited_bits": 512.5, "share": 0.39, "health": "ok"}},
    "fish": [{"fish_id": "f3", "name": "Nemo", "bits": 120.5, "share": 0.31}],
}


@pytest.fixture
def umask_zero() -> Iterator[None]:
    """Most permissive umask: proves 0600 comes from creation, not from the umask."""
    previous = os.umask(0)
    try:
        yield
    finally:
        os.umask(previous)


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- key ids -----------------------------------------------------------------


def test_key_id_follows_design_formula(key: bytes) -> None:
    expected = hashlib.blake2b(key, digest_size=8, person=b"puffer-key-id").hexdigest()
    assert key_id(key) == expected
    assert re.fullmatch(r"[0-9a-f]{16}", key_id(key))


def test_key_id_known_value() -> None:
    # Pinned: key ids are stored in every .puf header, so they must never change.
    assert key_id(bytes(range(32))) == "406f03dada41df19"


def test_key_id_distinguishes_keys(key: bytes, other_key: bytes) -> None:
    assert key_id(key) != key_id(other_key)


def test_key_id_rejects_bad_keys() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        key_id(bytes(16))
    with pytest.raises(TypeError, match="bytes"):
        key_id(bytearray(32))  # type: ignore[arg-type]


# --- save / load -------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path, key: bytes) -> None:
    path = tmp_path / "fish.key"
    saved = save_key(path, key, receipt=RECEIPT)
    loaded = load_key(path)
    assert loaded == saved
    assert loaded.key == key
    assert loaded.key_id == key_id(key)
    assert loaded.receipt == RECEIPT
    assert loaded.created.tzinfo is not None
    assert loaded.created.utcoffset() == timedelta(0)
    assert abs(datetime.now(UTC) - loaded.created) < timedelta(minutes=1)


def test_on_disk_format(tmp_path: Path, key: bytes) -> None:
    path = tmp_path / "fish.key"
    save_key(path, key, receipt=RECEIPT)
    doc = json.loads(path.read_text("ascii"))
    assert list(doc) == ["format", "version", "key_id", "created", "key", "receipt"]
    assert doc["format"] == "puffer-key"
    assert doc["version"] == 1
    assert doc["key_id"] == key_id(key)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", doc["created"])
    assert doc["key"] == b64url_encode(key)
    assert "=" not in doc["key"]
    assert doc["receipt"] == RECEIPT


def test_receipt_is_optional(tmp_path: Path, key: bytes) -> None:
    path = tmp_path / "plain.key"
    save_key(path, key)
    assert "receipt" not in json.loads(path.read_text())
    assert load_key(path).receipt is None


def test_explicit_null_receipt_is_accepted(tmp_path: Path, key: bytes) -> None:
    path = tmp_path / "null.key"
    save_key(path, key)
    doc = json.loads(path.read_text())
    doc["receipt"] = None
    path.write_text(json.dumps(doc))
    os.chmod(path, 0o600)
    assert load_key(path).receipt is None


def test_offset_timestamps_are_normalised_to_utc(tmp_path: Path, key: bytes) -> None:
    path = tmp_path / "tz.key"
    save_key(path, key)
    doc = json.loads(path.read_text())
    doc["created"] = "2026-10-31T18:00:00+03:00"
    path.write_text(json.dumps(doc))
    os.chmod(path, 0o600)
    assert load_key(path).created == datetime(2026, 10, 31, 15, 0, tzinfo=UTC)


@posix_only
@pytest.mark.usefixtures("umask_zero")
def test_key_file_is_private_from_creation(tmp_path: Path, key: bytes) -> None:
    path = tmp_path / "fish.key"
    save_key(path, key)
    assert mode_of(path) == 0o600
    save_key(path, key, overwrite=True)
    assert mode_of(path) == 0o600


def test_refuses_to_overwrite(tmp_path: Path, key: bytes, other_key: bytes) -> None:
    path = tmp_path / "fish.key"
    save_key(path, key)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_key(path, other_key)
    assert path.read_bytes() == before
    save_key(path, other_key, overwrite=True)
    assert load_key(path).key == other_key


def test_no_temporary_files_left_behind(tmp_path: Path, key: bytes) -> None:
    save_key(tmp_path / "a.key", key)
    with pytest.raises(FileExistsError):
        save_key(tmp_path / "a.key", key)
    with pytest.raises(ValueError, match="JSON compliant"):
        save_key(tmp_path / "b.key", key, receipt={"bits": float("nan")})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.key"]


def test_save_rejects_bad_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        save_key(tmp_path / "x.key", bytes(31))
    assert not (tmp_path / "x.key").exists()


# --- validation --------------------------------------------------------------

Mutation = Callable[[dict[str, Any]], None]


def _set(field: str, value: Any) -> Mutation:
    def mutate(doc: dict[str, Any]) -> None:
        doc[field] = value

    return mutate


def _drop(field: str) -> Mutation:
    def mutate(doc: dict[str, Any]) -> None:
        del doc[field]

    return mutate


def _tamper_key_id(doc: dict[str, Any]) -> None:
    flipped = "0" if doc["key_id"][0] != "0" else "1"
    doc["key_id"] = flipped + doc["key_id"][1:]


def _swap_key(doc: dict[str, Any]) -> None:
    """A different, well-formed key under the original key_id."""
    doc["key"] = b64url_encode(bytes(KEY_BYTES))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_tamper_key_id, "does not match"),
        (_swap_key, "does not match"),
        (_set("key_id", "ABCDEF0123456789"), "lowercase hex"),
        (_set("key_id", "abc"), "lowercase hex"),
        (_set("key_id", 12), "lowercase hex"),
        (_set("key", b64url_encode(bytes(31))), "32 bytes"),
        (_set("key", b64url_encode(bytes(33))), "32 bytes"),
        (_set("key", "not base64!"), "base64url"),
        (_set("key", 42), "base64url string"),
        (_set("format", "age-key"), "not a puffer key file"),
        (_set("version", 2), "version"),
        (_set("version", True), "version"),
        (_set("version", "1"), "version"),
        (_set("created", "yesterday"), "ISO 8601"),
        (_set("created", "2026-10-31T15:00:00"), "UTC offset"),
        (_set("created", 1793454000), "must be a string"),
        (_set("receipt", [1, 2]), "JSON object"),
        (_set("comment", "hi"), "unknown fields"),
        (_drop("key"), "missing fields"),
        (_drop("key_id"), "missing fields"),
        (_drop("created"), "missing fields"),
        (_drop("format"), "missing fields"),
    ],
)
def test_load_rejects_invalid_documents(
    tmp_path: Path, key: bytes, mutate: Mutation, message: str
) -> None:
    path = tmp_path / "fish.key"
    save_key(path, key, receipt=RECEIPT)
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc))
    os.chmod(path, 0o600)
    with pytest.raises(FormatError, match=message):
        load_key(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "not valid JSON"),
        ("{", "not valid JSON"),
        ("[]", "JSON object"),
        ('"puffer-key"', "JSON object"),
        ('{"format": "puffer-key", "format": "puffer-key"}', "duplicate"),
        ('{"version": NaN}', "NaN"),
        ('{"version": Infinity}', "Infinity"),
    ],
)
def test_load_rejects_malformed_json(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.key"
    path.write_text(content)
    os.chmod(path, 0o600)
    with pytest.raises(FormatError, match=message):
        load_key(path)


@pytest.mark.parametrize(
    ("receipt", "message"),
    [
        ("[" * 100_000 + "]" * 100_000, "not valid JSON"),  # nesting: RecursionError
        ("1" * 5000, "not valid JSON"),  # beyond int()'s digit limit: plain ValueError
        ("1e999", "overflows"),  # Infinity in disguise
        ("-1e999", "overflows"),
    ],
    ids=["deep-nesting", "huge-int", "inf", "minus-inf"],
)
def test_load_rejects_pathological_json(
    tmp_path: Path, key: bytes, receipt: str, message: str
) -> None:
    """Hostile but small key files must fail as FormatError, never as another exception."""
    path = tmp_path / "evil.key"
    save_key(path, key)
    text = path.read_text().rstrip().removesuffix("}") + f', "receipt": {{"x": {receipt}}}}}'
    path.write_text(text)
    os.chmod(path, 0o600)
    with pytest.raises(FormatError, match=message):
        load_key(path)


def test_load_rejects_duplicate_key_members(tmp_path: Path, key: bytes, other_key: bytes) -> None:
    """Two "key" members could be resolved differently by different JSON parsers."""
    path = tmp_path / "dup.key"
    save_key(path, key)
    text = path.read_text().replace('"key":', f'"key": "{b64url_encode(other_key)}",\n  "key":')
    path.write_text(text)
    os.chmod(path, 0o600)
    with pytest.raises(FormatError, match="duplicate JSON member 'key'"):
        load_key(path)


def test_load_rejects_oversized_and_binary_files(tmp_path: Path) -> None:
    big = tmp_path / "big.key"
    big.write_bytes(b" " * ((1 << 20) + 1))
    os.chmod(big, 0o600)
    with pytest.raises(FormatError, match="larger than"):
        load_key(big)
    binary = tmp_path / "bin.key"
    binary.write_bytes(b"\xff\xfe{}")
    os.chmod(binary, 0o600)
    with pytest.raises(FormatError, match="not valid JSON"):
        load_key(binary)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_key(tmp_path / "absent.key")


@posix_only
def test_warns_about_exposed_key_files(tmp_path: Path, key: bytes) -> None:
    path = tmp_path / "fish.key"
    save_key(path, key)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_key(path)  # 0600: silent
    os.chmod(path, 0o644)
    with pytest.warns(InsecurePermissionsWarning, match="chmod 600") as record:
        assert load_key(path).key == key
    assert record[0].filename == __file__


# --- KeyFile -----------------------------------------------------------------


def test_repr_redacts_the_key(key: bytes) -> None:
    kf = KeyFile(key=key, key_id=key_id(key), created=datetime.now(UTC), receipt=RECEIPT)
    for text in (repr(kf), str(kf), f"{kf}", repr([kf])):
        assert "redacted" in text
        assert key_id(key) in text
        assert key.hex() not in text
        assert b64url_encode(key) not in text
        assert repr(key) not in text


def test_keyfile_validates_consistency(key: bytes, other_key: bytes) -> None:
    with pytest.raises(ValueError, match="key_id does not match"):
        KeyFile(key=key, key_id=key_id(other_key), created=datetime.now(UTC))
    with pytest.raises(ValueError, match="timezone-aware"):
        KeyFile(key=key, key_id=key_id(key), created=datetime(2026, 10, 31))
    with pytest.raises(ValueError, match="32 bytes"):
        KeyFile(key=bytes(8), key_id="0" * 16, created=datetime.now(UTC))


def test_keyfile_is_immutable_and_hashable(key: bytes) -> None:
    kf = KeyFile(key=key, key_id=key_id(key), created=datetime.now(UTC), receipt=RECEIPT)
    with pytest.raises(AttributeError):
        kf.key = bytes(32)  # type: ignore[misc]
    assert hash(kf) == hash(KeyFile(kf.key, kf.key_id, kf.created, receipt=None))
