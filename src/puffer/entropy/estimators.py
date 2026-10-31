"""NIST SP 800-90B min-entropy estimators (non-IID track, section 6.3).

All estimators take a 1-D numpy array of integer symbols and return an
:class:`EstimatorResult` with the estimated min-entropy *per sample* in bits.
They are statistical tools only - no cryptography here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

Z_99 = 2.576  # z_{1-0.005}: the 99% upper confidence bound used throughout 800-90B


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    name: str
    min_entropy: float  # bits per sample
    p_upper: float  # upper bound on the probability of the most likely outcome / prediction
    details: dict[str, Any] = field(default_factory=dict)


def most_common_value(samples: npt.ArrayLike) -> EstimatorResult:
    """SP 800-90B section 6.3.1 Most Common Value estimate.

    p_hat = max_count / L
    p_u   = min(1, p_hat + 2.576 * sqrt(p_hat * (1 - p_hat) / (L - 1)))
    H     = -log2(p_u)
    """
    s = np.asarray(samples).ravel()
    n = int(s.size)
    if n < 2:
        raise ValueError("most_common_value needs at least 2 samples")
    _, counts = np.unique(s, return_counts=True)
    p_hat = float(np.max(counts)) / n
    p_u = min(1.0, p_hat + Z_99 * math.sqrt(p_hat * (1.0 - p_hat) / (n - 1)))
    h = -math.log2(p_u) if p_u > 0 else 0.0
    return EstimatorResult("most_common_value", max(0.0, h), p_u, {"p_hat": p_hat, "n": n})
