import math

import numpy as np

from puffer.entropy.estimators import most_common_value


def test_mcv_uniform_bytes_close_to_8_bits(rng):
    r = most_common_value(rng.integers(0, 256, size=200_000))
    assert 7.3 < r.min_entropy <= 8.0


def test_mcv_biased_binary_is_conservative(rng):
    s = (rng.random(100_000) < 0.75).astype(np.uint8)
    r = most_common_value(s)
    assert r.min_entropy <= -math.log2(0.75)
    assert r.min_entropy > -math.log2(0.76)


def test_mcv_constant_is_zero():
    assert most_common_value(np.zeros(1000, dtype=np.uint8)).min_entropy == 0.0
