"""Box geometry and gated linear assignment shared by the tracker and re-ID.

Boxes follow :data:`puffer.types.BBox` - continuous pixel coordinates
``(x1, y1, x2, y2)``. ByteTrack's reference code uses the legacy "+1 pixel"
convention from Pascal VOC (via ``cython_bbox``); we use the continuous one
because detections here come from sub-pixel blob statistics and Kalman
estimates, not integer annotations.

:func:`linear_assignment` reproduces ``lap.lapjv(cost, extend_cost=True,
cost_limit=thresh)`` - the solver ByteTrack actually calls - on top of
:func:`scipy.optimize.linear_sum_assignment`. Simply running the Hungarian
algorithm and discarding pairs above the threshold afterwards is *not*
equivalent: the optimum may then contain a forbidden pair whose removal
leaves two objects unmatched that could each have been matched elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from puffer.types import BBox

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


def bbox_to_xyah(bbox: BBox) -> FloatArray:
    """Convert ``(x1, y1, x2, y2)`` to the Kalman measurement ``(cx, cy, a, h)``.

    ``a = w / h`` is the aspect ratio. ByteTrack/DeepSORT measure aspect and
    height rather than width and height because the aspect of a rigid object
    changes slowly, which keeps the constant-velocity model honest.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError(f"degenerate bounding box {bbox!r}")
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def xyah_to_bbox(xyah: npt.ArrayLike) -> BBox:
    """Inverse of :func:`bbox_to_xyah` (only the first four entries are used)."""
    cx, cy, a, h = (float(v) for v in np.asarray(xyah, dtype=np.float64)[:4])
    w = a * h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def boxes_to_array(boxes: Sequence[BBox]) -> FloatArray:
    """Stack boxes into an ``(N, 4)`` float array (``(0, 4)`` when empty)."""
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def iou_matrix(a: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """Pairwise intersection-over-union of ``(N, 4)`` and ``(M, 4)`` boxes.

    Returns an ``(N, M)`` matrix; empty inputs give an empty matrix of the
    right shape so callers never special-case "no tracks" or "no detections".
    """
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if aa.shape[0] == 0 or bb.shape[0] == 0:
        return np.zeros((aa.shape[0], bb.shape[0]), dtype=np.float64)
    ix1 = np.maximum(aa[:, None, 0], bb[None, :, 0])
    iy1 = np.maximum(aa[:, None, 1], bb[None, :, 1])
    ix2 = np.minimum(aa[:, None, 2], bb[None, :, 2])
    iy2 = np.minimum(aa[:, None, 3], bb[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area_a = (aa[:, 2] - aa[:, 0]) * (aa[:, 3] - aa[:, 1])
    area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    out: FloatArray = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return out


def linear_assignment(
    cost: npt.ArrayLike, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost matching in which no matched pair may cost more than ``thresh``.

    Implements the ``lap.lapjv(..., cost_limit=thresh)`` extension: the
    ``N x M`` cost matrix is embedded in an ``(N+M) x (N+M)`` square matrix
    where leaving a row or a column unmatched costs ``thresh / 2`` each, so a
    pair is only worth matching when it is cheaper than leaving both sides
    alone. Entries above ``thresh`` (and NaN / inf) can therefore never be
    selected.

    Returns ``(matches, unmatched_rows, unmatched_cols)`` with ``matches`` as
    ``(row, col)`` pairs sorted by row.
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    n, m = c.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    # Forbidden pairs get a cost that can never beat "leave both unmatched".
    forbidden = thresh + 1.0
    c = np.where(np.isfinite(c) & (c <= thresh), c, forbidden)
    size = n + m
    ext = np.full((size, size), thresh / 2.0, dtype=np.float64)
    ext[:n, :m] = c
    ext[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(ext)
    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if r < n and col < m and c[r, col] <= thresh:
            matches.append((r, col))
            matched_rows.add(r)
            matched_cols.add(col)
    unmatched_rows = [i for i in range(n) if i not in matched_rows]
    unmatched_cols = [j for j in range(m) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
