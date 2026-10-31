"""Box conversions, IoU and the lapjv-equivalent gated assignment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from puffer.vision.geometry import (
    bbox_to_xyah,
    boxes_to_array,
    iou_matrix,
    linear_assignment,
    xyah_to_bbox,
)


def test_xyah_round_trip() -> None:
    box = (10.0, 20.0, 50.0, 40.0)
    xyah = bbox_to_xyah(box)
    np.testing.assert_allclose(xyah, [30.0, 30.0, 2.0, 20.0])
    np.testing.assert_allclose(xyah_to_bbox(xyah), box)


@pytest.mark.parametrize("box", [(0.0, 0.0, 0.0, 5.0), (0.0, 5.0, 4.0, 5.0), (3.0, 0.0, 1.0, 2.0)])
def test_degenerate_box_rejected(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        bbox_to_xyah(box)


def test_boxes_to_array_empty() -> None:
    assert boxes_to_array([]).shape == (0, 4)


def test_iou_known_values() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50.0 / 150.0, 0.0]])


def test_iou_touching_boxes_is_zero() -> None:
    assert iou_matrix([[0, 0, 10, 10]], [[10, 0, 20, 10]])[0, 0] == 0.0


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 3), (2, 0)])
def test_iou_empty_shapes(n: int, m: int) -> None:
    a = np.tile([0.0, 0.0, 1.0, 1.0], (n, 1))
    b = np.tile([0.0, 0.0, 1.0, 1.0], (m, 1))
    assert iou_matrix(a, b).shape == (n, m)


def test_iou_symmetric(rng: np.random.Generator) -> None:
    xy = rng.uniform(0, 100, size=(6, 2))
    wh = rng.uniform(5, 30, size=(6, 2))
    boxes = np.hstack([xy, xy + wh])
    iou = iou_matrix(boxes, boxes)
    np.testing.assert_allclose(iou, iou.T)
    np.testing.assert_allclose(np.diag(iou), 1.0)
    assert np.all((iou >= 0.0) & (iou <= 1.0))


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (3, 0)])
def test_assignment_empty(n: int, m: int) -> None:
    matches, rows, cols = linear_assignment(np.zeros((n, m)), 0.5)
    assert matches == []
    assert rows == list(range(n))
    assert cols == list(range(m))


def test_assignment_picks_minimum_cost() -> None:
    cost = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.6]])
    matches, rows, cols = linear_assignment(cost, 0.7)
    assert matches == [(0, 1), (1, 0)]
    assert rows == []
    assert cols == [2]


def test_assignment_respects_threshold() -> None:
    matches, rows, cols = linear_assignment(np.array([[0.75, 0.9]]), 0.7)
    assert matches == []
    assert rows == [0]
    assert cols == [0, 1]


def test_assignment_matches_lapjv_cost_limit_semantics() -> None:
    """Leaving a pair unmatched costs ``thresh``: one cheap match beats two mediocre ones.

    A plain Hungarian solve would pick (0,1)+(1,0) (total 1.8, both under the
    threshold); lapjv with ``cost_limit=1`` - what ByteTrack uses - prefers
    (0,0) alone (0 + 0.5 + 0.5 = 1.0).
    """
    cost = np.array([[0.0, 0.9], [0.9, 50.0]])
    matches, rows, cols = linear_assignment(cost, 1.0)
    assert matches == [(0, 0)]
    assert rows == [1]
    assert cols == [1]


def test_assignment_ignores_non_finite() -> None:
    cost = np.array([[math.nan, 0.2], [math.inf, 0.1]])
    matches, rows, cols = linear_assignment(cost, 0.5)
    assert len(matches) == 1
    assert matches[0][1] == 1
    assert cols == [0]
    assert len(rows) == 1


def test_assignment_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="2-D"):
        linear_assignment(np.zeros(3), 0.5)
