"""The committed cross-language vector file: determinism and expected verdicts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from nacl.signing import SigningKey

from puffer.beacon.chain import verify_chain
from puffer.beacon.pulse import ErrorCode, Pulse, verify_pulse
from puffer.beacon.vectors import (
    VECTOR_CHAIN_LENGTH,
    VECTOR_SEED,
    build_vectors,
    main,
    render_vectors,
)

VECTOR_FILE = Path(__file__).resolve().parents[1] / "vectors" / "beacon_vectors.json"

# Full, exact verdict for every negative vector (the file only carries the
# primary code). Written out by hand as an independent statement of intent.
EXPECTED_ERRORS: dict[str, tuple[ErrorCode, ...]] = {
    "tampered_signature": (ErrorCode.BAD_SIGNATURE,),
    "tampered_summary": (ErrorCode.BAD_SUMMARY_HASH,),
    "tampered_summary_hash": (ErrorCode.BAD_SUMMARY_HASH,),
    "tampered_output": (ErrorCode.BAD_OUTPUT,),
    "tampered_prev_output": (ErrorCode.BROKEN_LINK,),
    "tampered_local_random": (ErrorCode.BAD_PRECOMMITMENT,),
    "index_gap": (ErrorCode.BAD_INDEX,),
    "timestamp_regression": (ErrorCode.BAD_TIMESTAMP,),
    "timestamp_repeated": (ErrorCode.BAD_TIMESTAMP,),
    "wrong_public_key": (ErrorCode.WRONG_PUBLIC_KEY, ErrorCode.BAD_SIGNATURE),
    "relabelled_public_key": (ErrorCode.WRONG_PUBLIC_KEY,),
    "bad_genesis": (ErrorCode.BAD_GENESIS,),
}


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(VECTOR_FILE.read_text(encoding="utf-8"))
    return data


def test_committed_file_is_byte_identical_to_a_fresh_build() -> None:
    assert VECTOR_FILE.read_text(encoding="utf-8") == render_vectors(), (
        "regenerate with: python -m puffer.beacon.vectors --out tests/vectors/beacon_vectors.json"
    )


def test_generation_is_deterministic() -> None:
    assert render_vectors() == render_vectors()
    assert build_vectors() == build_vectors()


def test_document_shape(document: dict[str, Any]) -> None:
    assert set(document) == {"public_key", "valid_chain", "invalid"}
    assert document["public_key"] == bytes(SigningKey(VECTOR_SEED).verify_key).hex()
    assert len(document["valid_chain"]) == VECTOR_CHAIN_LENGTH
    names = [vector["name"] for vector in document["invalid"]]
    assert len(names) == len(set(names))
    assert set(names) == set(EXPECTED_ERRORS)
    for vector in document["invalid"]:
        assert set(vector) == {"name", "pulse", "prev", "expected_error"}


def test_valid_chain_verifies(document: dict[str, Any]) -> None:
    public_key = bytes.fromhex(document["public_key"])
    chain = [Pulse.from_json(item) for item in document["valid_chain"]]
    assert [pulse.index for pulse in chain] == list(range(VECTOR_CHAIN_LENGTH))
    report = verify_chain(chain, public_key)
    assert report.ok, report.errors
    assert report.checked == VECTOR_CHAIN_LENGTH


def test_valid_chain_exercises_non_ascii_summary_escapes(document: dict[str, Any]) -> None:
    summaries = [item["summary"] for item in document["valid_chain"]]
    assert all(summary.isascii() for summary in summaries)
    assert any("\\u0414" in summary for summary in summaries)


def test_every_error_code_has_a_vector(document: dict[str, Any]) -> None:
    covered = {vector["expected_error"] for vector in document["invalid"]}
    assert covered == {str(code) for code in ErrorCode}


@pytest.mark.parametrize("name", sorted(EXPECTED_ERRORS))
def test_negative_vector_fails_with_expected_code(document: dict[str, Any], name: str) -> None:
    vector = next(v for v in document["invalid"] if v["name"] == name)
    public_key = bytes.fromhex(document["public_key"])
    pulse = Pulse.from_json(vector["pulse"])
    prev = None if vector["prev"] is None else Pulse.from_json(vector["prev"])

    result = verify_pulse(pulse, public_key=public_key, prev=prev)
    assert not result.ok
    assert result.errors[0] == vector["expected_error"]
    assert result.errors == EXPECTED_ERRORS[name]

    # Without the explicit key the verdict holds too: prev (or the pulse) pins it.
    assert vector["expected_error"] in verify_pulse(pulse, prev=prev).errors


def test_cli_writes_identical_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "vectors.json"
    assert main(["--out", str(out)]) == 0
    assert out.read_bytes() == VECTOR_FILE.read_bytes()


def test_cli_check_mode(tmp_path: Path) -> None:
    out = tmp_path / "vectors.json"
    assert main(["--check", "--out", str(out)]) == 1  # missing
    out.write_text(render_vectors(), encoding="utf-8")
    assert main(["--check", "--out", str(out)]) == 0
    out.write_text(render_vectors().replace('"index": 3', '"index": 4'), encoding="utf-8")
    assert main(["--check", "--out", str(out)]) == 1


def test_module_entry_point_runs_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "vectors.json"
    # -W error: importing the package must not pre-import the __main__ module.
    subprocess.run(
        [sys.executable, "-W", "error", "-m", "puffer.beacon.vectors", "--out", str(out)],
        check=True,
        capture_output=True,
    )
    assert out.read_bytes() == VECTOR_FILE.read_bytes()
