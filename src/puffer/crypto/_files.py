"""Filesystem plumbing shared by key files and encrypted files.

Two properties matter for secrets on disk, and both are easy to get subtly
wrong with ``open(path, "w")``:

1. **Never a half-written or wrongly-permissioned file.** Output goes to a
   temporary file created by :func:`tempfile.mkstemp` in the destination
   directory. ``mkstemp`` creates it ``O_EXCL`` with mode ``0600``, so the
   secret is private from the very first byte (a later ``chmod`` would
   leave a window, and the umask can only remove bits). The data is
   fsync'ed and only then given its final name, so readers see either the
   old file, no file, or the complete new file. Any failure - including
   ``KeyboardInterrupt`` - deletes the temporary file.
2. **No silent clobbering.** Without ``overwrite`` the final name is
   claimed with ``os.link`` (fails atomically if the name exists, the
   technique used by the ``atomicwrites`` package), not with an
   exists-check followed by ``os.replace``, which would race. Filesystems
   without hard links (FAT, some FUSE mounts) fall back to check+replace.

Reading is equally strict: key files are small JSON objects, so the reader
caps the size, rejects duplicate keys (two ``"key"`` members could be
resolved differently by different parsers) and non-standard constants such
as ``NaN``.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import stat
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, NoReturn

from puffer.crypto.errors import FormatError, InsecurePermissionsWarning

StrPath = str | os.PathLike[str]

MAX_JSON_FILE_BYTES = 1 << 20
"""Upper bound for key-file size; real files are a few hundred bytes plus a receipt."""


def _exists_error(path: Path) -> FileExistsError:
    return FileExistsError(errno.EEXIST, "refusing to overwrite existing file", str(path))


def _fsync_directory(directory: Path) -> None:
    """Persist a rename on POSIX; best effort because some filesystems reject it."""
    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _publish(tmp: Path, path: Path, *, overwrite: bool) -> None:
    """Give the finished temporary file its final name."""
    if overwrite:
        os.replace(tmp, path)
        return
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise _exists_error(path) from None
    except OSError:
        # No hard links on this filesystem: fall back to a (racy) check.
        if os.path.lexists(path):
            raise _exists_error(path) from None
        os.replace(tmp, path)
        return
    tmp.unlink()


@contextlib.contextmanager
def atomic_output(path: StrPath, *, overwrite: bool = False) -> Iterator[BinaryIO]:
    """Yield a private (``0600``) binary file that becomes ``path`` only on success.

    If the ``with`` body raises, the temporary file is removed and ``path``
    is left exactly as it was. Without ``overwrite`` an existing ``path``
    raises :class:`FileExistsError`, both up front (before any expensive
    work) and at publication time (if another process created it meanwhile).
    """
    target = Path(path)
    if not overwrite and os.path.lexists(target):
        raise _exists_error(target)
    directory = target.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        _publish(tmp, target, overwrite=overwrite)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_directory(directory)


def write_private_file(path: StrPath, data: bytes, *, overwrite: bool = False) -> None:
    """Atomically write ``data`` to ``path`` with mode ``0600``."""
    with atomic_output(path, overwrite=overwrite) as handle:
        handle.write(data)


def warn_if_exposed(path: StrPath, *, stacklevel: int = 1) -> None:
    """Warn when a secret file grants any permission to group or others (POSIX only).

    ``stacklevel`` counts from the caller of this function, as in
    :func:`warnings.warn`, so the warning can point at user code.
    """
    if os.name != "posix":
        return
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o077:
        warnings.warn(
            f"{os.fspath(path)} has mode {mode:04o}; secret key files should be 0600 "
            f"(chmod 600 {os.fspath(path)})",
            InsecurePermissionsWarning,
            stacklevel=stacklevel + 1,
        )


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise FormatError(f"duplicate JSON member {name!r}")
        result[name] = value
    return result


def _reject_constant(name: str) -> NoReturn:
    raise FormatError(f"non-standard JSON constant {name}")


def read_json_object(path: StrPath, *, what: str) -> dict[str, Any]:
    """Read a small, strictly valid JSON object from ``path``.

    ``what`` names the file kind in error messages ("key file", ...).

    Raises:
        FormatError: oversized file, invalid UTF-8 or JSON, duplicate
            members, ``NaN``/``Infinity``, or a top level that is not an object.
        OSError: the file cannot be read.
    """
    with open(path, "rb") as handle:
        raw = handle.read(MAX_JSON_FILE_BYTES + 1)
    if len(raw) > MAX_JSON_FILE_BYTES:
        raise FormatError(f"{what} is larger than {MAX_JSON_FILE_BYTES} bytes")
    try:
        doc = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise FormatError(f"{what} must contain a JSON object")
    return doc


def dump_json(doc: dict[str, Any]) -> bytes:
    """Serialize ``doc`` as strict, human-readable JSON (no NaN, ASCII only)."""
    return (json.dumps(doc, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
