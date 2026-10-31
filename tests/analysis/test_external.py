"""Wrappers for the ``ent`` and ``dieharder`` binaries: parsers and real runs.

Parsers are tested on canned output captured from ent 1.x / dieharder 3.31.1;
real runs are skipped when the binary is missing. Real dieharder runs use a
1 MB deterministic file and small ``-t``/``-p`` so the module stays fast.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from puffer.analysis.external import (
    DieharderAssessment,
    ExternalToolError,
    ToolNotFoundError,
    ToolTimeoutError,
    dieharder_available,
    ent_available,
    parse_dieharder_output,
    parse_ent_terse,
    run_dieharder,
    run_ent,
)

needs_ent = pytest.mark.skipif(not ent_available(), reason="ent binary not installed")
needs_dieharder = pytest.mark.skipif(not dieharder_available(), reason="dieharder not installed")

DIEHARDER_OUTPUT = """\
#=============================================================================#
#            dieharder version 3.31.1 Copyright 2003 Robert G. Brown          #
#=============================================================================#
   rng_name    |           filename             |rands/second|
 file_input_raw|                       u100k.bin|  6.47e+07  |
#=============================================================================#
        test_name   |ntup| tsamples |psamples|  p-value |Assessment
#=============================================================================#
      diehard_operm5|   0|      1000|      10|0.98502494|  PASSED
# The file file_input_raw was rewound 553 times
   diehard_birthdays|   0|       100|     100|0.05098460|  PASSED
# The file file_input_raw was rewound 4553 times
          sts_serial|   1|      1000|       5|0.99999991|  FAILED
          sts_serial|   2|      1000|       5|0.99978116|   WEAK
"""


@pytest.fixture
def raw_file(tmp_path, rng):
    path = tmp_path / "random.bin"
    path.write_bytes(rng.bytes(1_000_000))
    return path


# ent -------------------------------------------------------------------------


def test_parse_ent_terse_byte_mode():
    stats = parse_ent_terse(
        "0,File-bytes,Entropy,Chi-square,Mean,Monte-Carlo-Pi,Serial-Correlation\n"
        "1,100000,7.998276,238.310400,127.394360,3.130925,-0.002172\n"
    )
    assert stats.size == 100_000
    assert not stats.bit_mode
    assert stats.entropy == 7.998276
    assert stats.chi_square_percent == pytest.approx(76.61, abs=0.005)  # ent prints 76.61
    assert stats.compression_percent == pytest.approx(100 * (8 - 7.998276) / 8)
    assert stats.monte_carlo_error_percent == pytest.approx(0.34, abs=0.005)
    assert stats.serial_correlation == -0.002172


def test_parse_ent_terse_bit_mode_and_sentinels():
    stats = parse_ent_terse(
        "0,File-bits,Entropy,Chi-square,Mean,Monte-Carlo-Pi,Serial-Correlation\n"
        "1,40,0.983708,0.900000,0.425000,-nan,-100000.000000\n"
    )
    assert stats.bit_mode
    assert stats.size == 40
    assert stats.monte_carlo_pi is None
    assert stats.monte_carlo_error_percent is None
    assert stats.serial_correlation is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "garbage",
        "0,File-bytes,Entropy,Chi-square,Mean,Monte-Carlo-Pi,Serial-Correlation\n1,10,x,1,1,1,1",
        "0,File-words,Entropy,Chi-square,Mean,Monte-Carlo-Pi,Serial-Correlation\n1,1,1,1,1,1,1",
        "0,File-bytes,Entropy,Chi-square,Mean,Monte-Carlo-Pi,Serial-Correlation\n"
        "1,0,0.000000,-nan,-nan,-nan,-100000.000000",
    ],
)
def test_parse_ent_terse_rejects_malformed_output(text):
    with pytest.raises(ExternalToolError):
        parse_ent_terse(text)


@needs_ent
def test_run_ent_rejects_missing_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_ent(tmp_path / "missing.bin")
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ExternalToolError, match="empty"):
        run_ent(empty)


def test_missing_tools_raise_tool_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert not ent_available()
    assert not dieharder_available()
    path = tmp_path / "x.bin"
    path.write_bytes(b"\x00" * 16)
    with pytest.raises(ToolNotFoundError):
        run_ent(path)
    with pytest.raises(ToolNotFoundError):
        run_dieharder(path, tests=[0], timeout_s=1.0)


# dieharder parsing ---------------------------------------------------------------


def test_parse_dieharder_rows_and_sticky_rewind_flags():
    rows = parse_dieharder_output(DIEHARDER_OUTPUT)
    assert [(r.test_name, r.ntup) for r in rows] == [
        ("diehard_operm5", 0),
        ("diehard_birthdays", 0),
        ("sts_serial", 1),
        ("sts_serial", 2),
    ]
    operm5, birthdays, serial1, serial2 = rows
    assert (operm5.tsamples, operm5.psamples, operm5.p_value) == (1000, 10, 0.98502494)
    assert operm5.assessment is DieharderAssessment.PASSED
    assert not operm5.rewound
    assert operm5.rewinds == 0
    assert birthdays.rewound
    assert birthdays.rewinds == 553
    assert serial1.assessment is DieharderAssessment.FAILED
    assert serial1.rewinds == 4553
    assert serial2.assessment is DieharderAssessment.WEAK
    assert serial2.rewound
    assert serial2.to_dict()["assessment"] == "WEAK"


def test_parse_dieharder_detects_error_reports():
    # dieharder prints this and exits with status 0.
    with pytest.raises(ExternalToolError, match="Cannot open"):
        parse_dieharder_output("# file_input_raw(): Error: Cannot open /nonexistent, exiting.\n")


def test_run_dieharder_validates_arguments(tmp_path):
    with pytest.raises(ValueError, match="timeout_s"):
        run_dieharder(tmp_path / "x.bin", tests=[0], timeout_s=0)
    if dieharder_available():
        with pytest.raises(FileNotFoundError):
            run_dieharder(tmp_path / "missing.bin", tests=[0], timeout_s=5.0)


# dieharder real runs -------------------------------------------------------------


@needs_dieharder
def test_real_run_without_rewind(raw_file):
    (result,) = run_dieharder(raw_file, tests=[1], tsamples=1000, psamples=10, timeout_s=30.0)
    assert result.test_name == "diehard_operm5"
    assert (result.tsamples, result.psamples) == (1000, 10)
    assert 0.0 <= result.p_value <= 1.0
    assert isinstance(result.assessment, DieharderAssessment)
    assert not result.rewound


@needs_dieharder
def test_real_run_that_exhausts_the_file_is_flagged_rewound(raw_file):
    # Birthdays draws ~2 KB per t-sample: 100 x 10 samples need ~2 MB of a 1 MB file.
    (result,) = run_dieharder(raw_file, tests=[0], tsamples=100, psamples=10, timeout_s=30.0)
    assert result.test_name == "diehard_birthdays"
    assert result.rewound
    assert result.rewinds >= 1


@needs_dieharder
def test_real_multi_row_test(raw_file):
    rows = run_dieharder(raw_file, tests=[102], tsamples=1000, psamples=5, timeout_s=30.0)
    assert {r.test_name for r in rows} == {"sts_serial"}
    assert sorted({r.ntup for r in rows}) == list(range(1, 17))


@needs_dieharder
def test_timeout_kills_the_run_and_keeps_finished_results(raw_file):
    # operm5 at these sizes takes milliseconds, birthdays about a second.
    start = time.monotonic()
    with pytest.raises(ToolTimeoutError) as info:
        run_dieharder(raw_file, tests=[1, 0], tsamples=1000, psamples=10, timeout_s=0.5)
    assert time.monotonic() - start < 5.0
    assert [r.test_name for r in info.value.partial] == ["diehard_operm5"]


@needs_dieharder
def test_full_battery_honours_the_timeout(raw_file):
    start = time.monotonic()
    with pytest.raises(ToolTimeoutError):
        run_dieharder(raw_file, timeout_s=0.3)
    assert time.monotonic() - start < 5.0


@needs_dieharder
def test_deterministic_input_gives_deterministic_p_values(raw_file):
    first = run_dieharder(raw_file, tests=[100], tsamples=10_000, psamples=10, timeout_s=30.0)
    second = run_dieharder(raw_file, tests=[100], tsamples=10_000, psamples=10, timeout_s=30.0)
    assert first == second
    assert np.isfinite(first[0].p_value)
