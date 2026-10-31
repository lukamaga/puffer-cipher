"""ent-equivalent statistics: hand-checkable cases and cross-validation with ``ent``.

When ``/usr/bin/ent`` (or any ``ent`` on PATH) is available, our rendered
report must be *byte-identical* to ent's own output for random, biased and
text inputs in byte and bit mode - every figure at ent's printed precision.
"""

from __future__ import annotations

import math
import subprocess

import numpy as np
import pytest

from puffer.analysis.ent import ent_stats, ent_stats_file
from puffer.analysis.external import ent_available, run_ent, tool_path

needs_ent = pytest.mark.skipif(not ent_available(), reason="ent binary not installed")

TEXT = (
    b"It is a truth universally acknowledged, that a single fish in possession "
    b"of a good aquarium, must be in want of an entropy pool. " * 40
)


def sample_inputs(rng: np.random.Generator) -> dict[str, bytes]:
    return {
        "random": rng.bytes(200_003),  # not a multiple of the 6-byte Monte Carlo group
        "biased": rng.binomial(255, 0.3, size=50_000).astype(np.uint8).tobytes(),
        "text": TEXT,
        "short": b"abcdefg",
    }


def test_all_byte_values_once_is_ideal():
    s = ent_stats(bytes(range(256)))
    assert s.size == 256
    assert s.entropy == pytest.approx(8.0)
    assert s.compression_percent == pytest.approx(0.0, abs=1e-12)
    assert s.chi_square == 0.0
    assert s.chi_square_percent == pytest.approx(100.0)
    assert s.mean == 127.5
    assert "more than 99.99 percent" in s.report()


def test_constant_input():
    s = ent_stats(b"\x07" * 600)
    assert s.entropy == 0.0
    assert s.compression_percent == 100.0
    assert s.mean == 7.0
    assert s.chi_square == pytest.approx(255 * 600)
    assert s.serial_correlation is None
    assert "undefined (all values equal!)" in s.report()
    assert "Entropy = 0.000000 bits per byte." in s.report()


def test_monte_carlo_uses_6_byte_groups_as_24_bit_coordinates():
    # (0, 0) is inside the circle, (2^24-1, 2^24-1) outside; the trailing
    # partial group is ignored.
    s = ent_stats(bytes(6) + b"\xff" * 6 + b"\x01\x02")
    assert s.monte_carlo_pi == 2.0
    assert s.monte_carlo_error_percent == pytest.approx(100 * (math.pi - 2) / math.pi)
    tiny = ent_stats(b"abcde")
    assert tiny.monte_carlo_pi is None
    assert tiny.monte_carlo_error_percent is None
    assert "undefined (fewer than 6 bytes)" in tiny.report()


def test_serial_correlation_is_circular():
    assert ent_stats(b"\x00\xff" * 50).serial_correlation == pytest.approx(-1.0)
    # 0, 1, 2 then wrap to 0: sum x_i x_(i+1) = 0 + 2 + 0.
    expected = (3 * 2 - 3**2) / (3 * 5 - 3**2)
    assert ent_stats(b"\x00\x01\x02").serial_correlation == pytest.approx(expected)


def test_bit_mode_counts_bits_msb_first():
    s = ent_stats(b"\xf0\x0f", bit_mode=True)
    assert s.bit_mode
    assert s.size == 16
    assert s.entropy == pytest.approx(1.0)
    assert s.mean == 0.5
    assert s.chi_square == 0.0
    # bits 1111 0000 0000 1111: 3 + 3 adjacent pairs of ones, plus the wrap-around pair.
    expected = (16 * 7 - 8**2) / (16 * 8 - 8**2)
    assert s.serial_correlation == pytest.approx(expected)
    assert "bits per bit" in s.report()


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one byte"):
        ent_stats(b"")


def test_streaming_matches_in_memory_across_chunk_boundaries(rng, tmp_path):
    data = rng.bytes(3 * 1024 * 1024 + 5)
    path = tmp_path / "data.bin"
    path.write_bytes(data)
    for bit_mode in (False, True):
        assert ent_stats_file(path, bit_mode=bit_mode) == ent_stats(data, bit_mode=bit_mode)


def test_to_dict_round_trips_fields():
    d = ent_stats(b"hello world").to_dict()
    assert d["size"] == 11
    assert d["bit_mode"] is False
    assert set(d) >= {"entropy", "chi_square_percent", "monte_carlo_pi", "serial_correlation"}


@needs_ent
@pytest.mark.parametrize("bit_mode", [False, True], ids=["bytes", "bits"])
def test_report_is_identical_to_ent_output(rng, tmp_path, bit_mode):
    exe = tool_path("ent")
    assert exe is not None
    for name, data in sample_inputs(rng).items():
        path = tmp_path / f"{name}.bin"
        path.write_bytes(data)
        real = subprocess.run(
            [exe, *(["-b"] if bit_mode else []), str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert ent_stats(data, bit_mode=bit_mode).report() == real, name


@needs_ent
@pytest.mark.parametrize("bit_mode", [False, True], ids=["bytes", "bits"])
def test_terse_output_matches_our_statistics(rng, tmp_path, bit_mode):
    for name, data in sample_inputs(rng).items():
        path = tmp_path / f"{name}.bin"
        path.write_bytes(data)
        theirs = run_ent(path, bit_mode=bit_mode)
        ours = ent_stats(data, bit_mode=bit_mode)
        assert (theirs.size, theirs.bit_mode) == (ours.size, ours.bit_mode), name
        for field in ("entropy", "chi_square", "mean", "monte_carlo_pi", "serial_correlation"):
            assert getattr(theirs, field) == pytest.approx(getattr(ours, field), abs=1e-6), (
                name,
                field,
            )
        # Derived figures inherit the 6-decimal rounding of ent -t.
        assert theirs.chi_square_percent == pytest.approx(ours.chi_square_percent, abs=1e-3)
