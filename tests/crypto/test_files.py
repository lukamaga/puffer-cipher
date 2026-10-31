"""encrypt_file / decrypt_file: atomic publication, overwrite protection, no leftovers."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pytest

from puffer.crypto import (
    AuthenticationError,
    RandomBytes,
    TruncatedError,
    WrongKeyError,
    _files,
    decrypt_file,
    encrypt_file,
    fileformat,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
PASSWORD = "puffer"


def listing(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


@pytest.fixture
def secret_file(tmp_path: Path, rng: np.random.Generator) -> tuple[Path, bytes]:
    data = rng.bytes(5000)
    path = tmp_path / "secret.bin"
    path.write_bytes(data)
    return path, data


def test_file_round_trip_key(tmp_path: Path, secret_file: tuple[Path, bytes], key: bytes) -> None:
    src, data = secret_file
    header = encrypt_file(src, tmp_path / "secret.puf", key=key, chunk_size=1024)
    assert header.chunk_size == 1024
    returned = decrypt_file(tmp_path / "secret.puf", tmp_path / "plain.bin", key=key)
    assert returned == header
    assert (tmp_path / "plain.bin").read_bytes() == data
    assert listing(tmp_path) == ["plain.bin", "secret.bin", "secret.puf"]


def test_file_round_trip_password(
    tmp_path: Path, secret_file: tuple[Path, bytes], fast_kdf: dict[str, Any]
) -> None:
    src, data = secret_file
    encrypt_file(str(src), str(tmp_path / "s.puf"), password=PASSWORD, **fast_kdf)
    decrypt_file(str(tmp_path / "s.puf"), str(tmp_path / "s.out"), password=PASSWORD)
    assert (tmp_path / "s.out").read_bytes() == data


@posix_only
def test_outputs_are_private(tmp_path: Path, secret_file: tuple[Path, bytes], key: bytes) -> None:
    src, _ = secret_file
    previous = os.umask(0)
    try:
        encrypt_file(src, tmp_path / "s.puf", key=key)
        decrypt_file(tmp_path / "s.puf", tmp_path / "s.out", key=key)
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "s.puf").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "s.out").stat().st_mode) == 0o600


def test_encrypt_refuses_to_overwrite(
    tmp_path: Path, secret_file: tuple[Path, bytes], key: bytes
) -> None:
    src, _ = secret_file
    dst = tmp_path / "s.puf"
    dst.write_bytes(b"precious")
    with pytest.raises(FileExistsError):
        encrypt_file(src, dst, key=key)
    assert dst.read_bytes() == b"precious"
    assert listing(tmp_path) == ["s.puf", "secret.bin"]
    encrypt_file(src, dst, key=key, overwrite=True)
    assert dst.read_bytes().startswith(b"PUFR")


def test_decrypt_refuses_to_overwrite(
    tmp_path: Path, secret_file: tuple[Path, bytes], key: bytes
) -> None:
    src, data = secret_file
    encrypt_file(src, tmp_path / "s.puf", key=key)
    with pytest.raises(FileExistsError):
        decrypt_file(tmp_path / "s.puf", src, key=key)
    decrypt_file(tmp_path / "s.puf", src, key=key, overwrite=True)
    assert src.read_bytes() == data


def test_in_place_encrypt_and_decrypt(
    tmp_path: Path, secret_file: tuple[Path, bytes], key: bytes
) -> None:
    """Reading src while writing a separate temp file makes src == dst safe."""
    src, data = secret_file
    encrypt_file(src, src, key=key, overwrite=True)
    assert src.read_bytes()[:4] == b"PUFR"
    decrypt_file(src, src, key=key, overwrite=True)
    assert src.read_bytes() == data
    assert listing(tmp_path) == ["secret.bin"]


def _corrupt_last_byte(path: Path) -> None:
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(bytes(blob))


def _drop_final_chunk(path: Path) -> None:
    blob = path.read_bytes()
    path.write_bytes(blob[: -(5000 % 1024 + 17)])


@pytest.mark.parametrize(
    ("damage", "error"),
    [(_corrupt_last_byte, AuthenticationError), (_drop_final_chunk, TruncatedError)],
    ids=["tampered-final-chunk", "dropped-final-chunk"],
)
@pytest.mark.parametrize("existing", [False, True], ids=["fresh", "overwrite"])
def test_failed_decryption_leaves_no_plaintext(
    tmp_path: Path,
    secret_file: tuple[Path, bytes],
    key: bytes,
    damage: Callable[[Path], None],
    error: type[Exception],
    existing: bool,
) -> None:
    """Most chunks are authentic, but nothing may be published unless all of them are."""
    src, _ = secret_file
    puf, out = tmp_path / "s.puf", tmp_path / "out.bin"
    encrypt_file(src, puf, key=key, chunk_size=1024)
    damage(puf)
    if existing:
        out.write_bytes(b"previous contents")
    before = listing(tmp_path)
    with pytest.raises(error):
        decrypt_file(puf, out, key=key, overwrite=existing)
    assert listing(tmp_path) == before
    if existing:
        assert out.read_bytes() == b"previous contents"


def test_wrong_credentials_leave_nothing(
    tmp_path: Path,
    secret_file: tuple[Path, bytes],
    key: bytes,
    other_key: bytes,
    fast_kdf: dict[str, Any],
) -> None:
    src, _ = secret_file
    encrypt_file(src, tmp_path / "k.puf", key=key)
    encrypt_file(src, tmp_path / "p.puf", password=PASSWORD, **fast_kdf)
    before = listing(tmp_path)
    with pytest.raises(WrongKeyError):
        decrypt_file(tmp_path / "k.puf", tmp_path / "out", key=other_key)
    with pytest.raises(AuthenticationError):
        decrypt_file(tmp_path / "p.puf", tmp_path / "out", password="wrong")
    assert listing(tmp_path) == before


def test_interrupt_mid_decryption_cleans_up(
    tmp_path: Path,
    secret_file: tuple[Path, bytes],
    key: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src, _ = secret_file
    encrypt_file(src, tmp_path / "s.puf", key=key)
    before = listing(tmp_path)

    def interrupted(src_stream: BinaryIO, dst: BinaryIO, **kwargs: Any) -> None:
        dst.write(b"partial plaintext")
        raise KeyboardInterrupt

    monkeypatch.setattr(fileformat, "decrypt_stream", interrupted)
    with pytest.raises(KeyboardInterrupt):
        decrypt_file(tmp_path / "s.puf", tmp_path / "out", key=key)
    assert listing(tmp_path) == before


def test_missing_source_creates_nothing(tmp_path: Path, key: bytes) -> None:
    with pytest.raises(FileNotFoundError):
        encrypt_file(tmp_path / "absent", tmp_path / "x.puf", key=key)
    with pytest.raises(FileNotFoundError):
        decrypt_file(tmp_path / "absent.puf", tmp_path / "x", key=key)
    assert listing(tmp_path) == []


def test_missing_destination_directory(
    tmp_path: Path, secret_file: tuple[Path, bytes], key: bytes
) -> None:
    src, _ = secret_file
    with pytest.raises(FileNotFoundError):
        encrypt_file(src, tmp_path / "no" / "such" / "dir.puf", key=key)


def _racing_rng(dst: Path, inner: RandomBytes) -> RandomBytes:
    """Create ``dst`` while encryption is in progress, as a concurrent process might."""

    def rng(n: int) -> bytes:
        dst.write_bytes(b"created by someone else")
        return inner(n)

    return rng


def test_destination_created_during_encryption_is_not_clobbered(
    tmp_path: Path, secret_file: tuple[Path, bytes], fast_kdf: dict[str, Any]
) -> None:
    src, _ = secret_file
    dst = tmp_path / "s.puf"
    kdf = {**fast_kdf, "random_bytes": _racing_rng(dst, fast_kdf["random_bytes"])}
    with pytest.raises(FileExistsError):
        encrypt_file(src, dst, password=PASSWORD, **kdf)
    assert dst.read_bytes() == b"created by someone else"
    assert listing(tmp_path) == ["s.puf", "secret.bin"]


def _no_hard_links(src: Any, dst: Any) -> None:
    raise PermissionError(1, "Operation not permitted")


def test_filesystems_without_hard_links(
    tmp_path: Path,
    secret_file: tuple[Path, bytes],
    key: bytes,
    fast_kdf: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_files.os, "link", _no_hard_links)
    src, data = secret_file
    encrypt_file(src, tmp_path / "s.puf", key=key)
    decrypt_file(tmp_path / "s.puf", tmp_path / "s.out", key=key)
    assert (tmp_path / "s.out").read_bytes() == data
    with pytest.raises(FileExistsError):
        encrypt_file(src, tmp_path / "s.puf", key=key)

    raced = tmp_path / "raced.puf"
    kdf = {**fast_kdf, "random_bytes": _racing_rng(raced, fast_kdf["random_bytes"])}
    with pytest.raises(FileExistsError):
        encrypt_file(src, raced, password=PASSWORD, **kdf)
    assert raced.read_bytes() == b"created by someone else"
    assert listing(tmp_path) == ["raced.puf", "s.out", "s.puf", "secret.bin"]
