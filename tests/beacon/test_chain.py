"""Chain verification over sequences and exported files."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from puffer.beacon.chain import ChainError, ChainReport, verify_chain, verify_file
from puffer.beacon.pulse import BeaconFormatError, ErrorCode, Pulse
from puffer.beacon.vectors import FOREIGN_SEED, build_chain


@pytest.fixture
def chain(signing_key: SigningKey) -> list[Pulse]:
    return build_chain(signing_key, 8)


@pytest.fixture
def public_key(signing_key: SigningKey) -> bytes:
    return bytes(signing_key.verify_key)


def _codes(report: ChainReport) -> list[tuple[int, str]]:
    return [(error.position, str(error.code)) for error in report.errors]


def test_valid_chain(chain: list[Pulse], public_key: bytes) -> None:
    report = verify_chain(chain, public_key)
    assert report.ok
    assert report.checked == 8
    assert report.first_bad_index is None


def test_segment_without_genesis_verifies(chain: list[Pulse], public_key: bytes) -> None:
    assert verify_chain(chain[3:], public_key).ok


def test_empty_chain_is_vacuously_ok(public_key: bytes) -> None:
    report = verify_chain([], public_key)
    assert report.ok
    assert report.checked == 0


def test_tampered_output_also_breaks_the_next_link(chain: list[Pulse], public_key: bytes) -> None:
    chain[2] = dataclasses.replace(chain[2], output=bytes(64))
    report = verify_chain(chain, public_key)
    assert not report.ok
    assert report.first_bad_index == 2
    assert _codes(report) == [(2, "bad_output"), (3, "broken_link")]
    assert str(report.errors[0]) == "pulse #2 (index 2): bad_output"


def test_dropped_pulse_is_detected(chain: list[Pulse], public_key: bytes) -> None:
    del chain[4]
    report = verify_chain(chain, public_key)
    assert _codes(report) == [(4, "bad_index"), (4, "broken_link"), (4, "bad_precommitment")]
    assert report.first_bad_index == 5


def test_swapped_pulses_are_detected(chain: list[Pulse], public_key: bytes) -> None:
    chain[5], chain[6] = chain[6], chain[5]
    report = verify_chain(chain, public_key)
    assert {code for _, code in _codes(report)} >= {
        "bad_index",
        "bad_timestamp",
        "broken_link",
        "bad_precommitment",
    }
    assert report.first_bad_index == 6


def test_wrong_key_rejects_every_pulse(chain: list[Pulse]) -> None:
    foreign = bytes(SigningKey(FOREIGN_SEED).verify_key)
    report = verify_chain(chain, foreign)
    assert report.checked == 8
    assert {error.position for error in report.errors} == set(range(8))
    assert all(
        error.code in (ErrorCode.WRONG_PUBLIC_KEY, ErrorCode.BAD_SIGNATURE)
        for error in report.errors
    )


def test_chain_rejects_malformed_key(chain: list[Pulse]) -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        verify_chain(chain, b"short")


def test_chain_error_is_structured() -> None:
    error = ChainError(position=1, index=9, code=ErrorCode.BAD_OUTPUT)
    assert ChainReport(checked=2, errors=(error,)).first_bad_index == 9


# --------------------------------------------------------------------------
# verify_file
# --------------------------------------------------------------------------


def _write(path: Path, chain: list[Pulse]) -> Path:
    path.write_text(json.dumps([pulse.to_json() for pulse in chain]), encoding="utf-8")
    return path


def test_verify_file_with_pinned_key(tmp_path: Path, chain: list[Pulse], public_key: bytes) -> None:
    path = _write(tmp_path / "pulses.json", chain)
    assert verify_file(path, public_key).ok
    assert verify_file(str(path)).ok  # trust on first use: internal consistency only


def test_verify_file_reports_tampering(
    tmp_path: Path, chain: list[Pulse], public_key: bytes
) -> None:
    chain[1] = dataclasses.replace(chain[1], summary='{"fish_bits":999999}')
    report = verify_file(_write(tmp_path / "pulses.json", chain), public_key)
    assert _codes(report) == [(1, "bad_summary_hash")]


def test_verify_file_with_foreign_key_fails(tmp_path: Path, chain: list[Pulse]) -> None:
    path = _write(tmp_path / "pulses.json", chain)
    assert not verify_file(path, bytes(SigningKey(FOREIGN_SEED).verify_key)).ok


def test_verify_file_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    assert verify_file(path).checked == 0


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json", "not a valid JSON"),
        ('{"pulses": []}', "expected a JSON list"),
        ('[{"version": 1}]', "pulse #0"),
    ],
)
def test_verify_file_rejects_malformed_files(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(BeaconFormatError, match=match):
        verify_file(path)


def test_verify_file_names_the_malformed_position(tmp_path: Path, chain: list[Pulse]) -> None:
    items = [pulse.to_json() for pulse in chain[:3]]
    items[2]["output"] = "00"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    with pytest.raises(BeaconFormatError, match="pulse #2"):
        verify_file(path)


def test_verify_file_rejects_duplicate_keys(
    tmp_path: Path, chain: list[Pulse], public_key: bytes
) -> None:
    """A key given twice means different things to different JSON parsers: refuse it.

    Here a reader that keeps the first ``summary`` sees a forged claim, while
    the signature covers the last one, which Python's ``json`` keeps.
    """
    genuine = json.dumps(chain[1].to_json())
    forged = genuine.replace('"summary": ', '"summary": "{\\"fish_bits\\":999999}", "summary": ', 1)
    assert forged != genuine
    path = tmp_path / "duplicate.json"
    path.write_text(f"[{json.dumps(chain[0].to_json())}, {forged}]", encoding="utf-8")
    with pytest.raises(BeaconFormatError, match="duplicate JSON key 'summary'"):
        verify_file(path, public_key)


@pytest.mark.parametrize(
    "content",
    ["[" + "1" * 5000 + "]", "[" * 100_000 + "]" * 100_000],
    ids=["integer-beyond-the-digit-limit", "runaway-nesting"],
)
def test_verify_file_rejects_pathological_json(tmp_path: Path, content: str) -> None:
    path = tmp_path / "hostile.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(BeaconFormatError, match="not a valid JSON"):
        verify_file(path)
