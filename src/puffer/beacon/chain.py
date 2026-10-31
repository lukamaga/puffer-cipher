"""Verification of whole pulse sequences, and the file helper behind ``puffer beacon verify``.

A beacon is only as trustworthy as its auditability: anyone holding an
exported list of pulses and the beacon's public key must be able to check,
offline, that every pulse is authentic and that the sequence is an unbroken
hash chain whose precommitments were all honoured. This module applies
:func:`~puffer.beacon.pulse.verify_pulse` along a sequence and reports *every*
failure with its position, rather than stopping at the first one, so an
auditor sees the full extent of a tampered export.

A sequence does not have to start at genesis: a segment such as the result
of ``GET /api/beacon/pulses?start=100`` verifies on its own, with its first
pulse checked without a predecessor.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puffer.beacon.pulse import BeaconFormatError, ErrorCode, Pulse, verify_pulse


@dataclass(frozen=True, slots=True)
class ChainError:
    """One failed check at one position of a verified sequence."""

    position: int  # 0-based position in the sequence that was verified
    index: int  # the ``index`` field of the offending pulse, as published
    code: ErrorCode

    def __str__(self) -> str:
        return f"pulse #{self.position} (index {self.index}): {self.code}"


@dataclass(frozen=True, slots=True)
class ChainReport:
    """Result of :func:`verify_chain`."""

    checked: int  # number of pulses examined
    errors: tuple[ChainError, ...] = ()

    @property
    def ok(self) -> bool:
        """``True`` when every pulse and every link verified."""
        return not self.errors

    @property
    def first_bad_index(self) -> int | None:
        """``index`` field of the first pulse that failed, or ``None``."""
        return self.errors[0].index if self.errors else None


def verify_chain(pulses: Iterable[Pulse], public_key: bytes) -> ChainReport:
    """Verify each pulse against ``public_key`` and against its predecessor in ``pulses``.

    Every pulse after the first is linked to the pulse *before it in the
    sequence*, even when that predecessor itself failed: a tampered output
    therefore also surfaces as ``broken_link`` on the next pulse, which is
    exactly what an auditor needs to see.

    Raises:
        ValueError: ``public_key`` is not 32 bytes.
    """
    errors: list[ChainError] = []
    prev: Pulse | None = None
    checked = 0
    for position, pulse in enumerate(pulses):
        result = verify_pulse(pulse, public_key=public_key, prev=prev)
        errors.extend(ChainError(position, pulse.index, code) for code in result.errors)
        prev = pulse
        checked += 1
    return ChainReport(checked=checked, errors=tuple(errors))


def verify_file(path: str | os.PathLike[str], public_key: bytes | None = None) -> ChainReport:
    """Verify a JSON file holding a list of pulses (the ``/api/beacon/pulses`` format).

    Pass the beacon's published ``public_key`` to verify authenticity. Without
    it the key embedded in the first pulse is pinned, which proves only that
    the file is internally consistent - anyone can produce a self-consistent
    chain under their own key - so callers should say so to the user.

    The file is parsed strictly: an object that repeats a key is refused,
    because JSON parsers disagree on which copy wins, and the verdict must be
    about the same bytes every reader sees.

    Raises:
        OSError: the file cannot be read.
        BeaconFormatError: the file is not a JSON list of well-formed pulses
            (the message names the offending position).
        ValueError: ``public_key`` is given but is not 32 bytes.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError) as exc:
        # ValueError covers JSONDecodeError, UnicodeDecodeError, duplicate keys
        # and integers beyond Python's digit limit; RecursionError runaway nesting.
        raise BeaconFormatError(f"{os.fspath(path)}: not a valid JSON document: {exc}") from exc
    if not isinstance(data, list):
        raise BeaconFormatError(f"{os.fspath(path)}: expected a JSON list of pulses")
    pulses: list[Pulse] = []
    for position, item in enumerate(data):
        try:
            pulses.append(Pulse.from_json(item))
        except BeaconFormatError as exc:
            raise BeaconFormatError(f"pulse #{position}: {exc}") from exc
    if not pulses:
        return ChainReport(checked=0)
    return verify_chain(pulses, public_key if public_key is not None else pulses[0].public_key)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that refuses a JSON object naming the same key twice."""
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise BeaconFormatError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj
