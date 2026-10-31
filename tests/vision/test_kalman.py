"""Constant-velocity Kalman filter maths."""

from __future__ import annotations

import numpy as np
import pytest

from puffer.vision.kalman import CHI2INV95, KalmanFilter


@pytest.fixture
def kf() -> KalmanFilter:
    return KalmanFilter()


def _is_spd(m: np.ndarray) -> bool:
    return bool(np.allclose(m, m.T) and np.all(np.linalg.eigvalsh(m) > 0))


def test_initiate_shapes_and_prior(kf: KalmanFilter) -> None:
    mean, cov = kf.initiate([100.0, 50.0, 2.0, 20.0])
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    np.testing.assert_allclose(mean, [100.0, 50.0, 2.0, 20.0, 0, 0, 0, 0])
    assert _is_spd(cov)
    # position std is 2 * h / 20 = 2 px for h = 20
    assert cov[0, 0] == pytest.approx(4.0)


def test_matrices(kf: KalmanFilter) -> None:
    f = kf.motion_matrix
    assert f.shape == (8, 8)
    assert f[0, 4] == 1.0
    assert f[3, 7] == 1.0
    np.testing.assert_array_equal(kf.observation_matrix, np.eye(4, 8))


def test_predict_moves_by_velocity_and_inflates_covariance(kf: KalmanFilter) -> None:
    mean, cov = kf.initiate([100.0, 50.0, 2.0, 20.0])
    mean[4:6] = [3.0, -2.0]
    new_mean, new_cov = kf.predict(mean, cov)
    np.testing.assert_allclose(new_mean[:2], [103.0, 48.0])
    np.testing.assert_allclose(new_mean[4:6], [3.0, -2.0])
    assert np.trace(new_cov) > np.trace(cov)
    assert _is_spd(new_cov)


def test_multi_predict_matches_single(kf: KalmanFilter, rng: np.random.Generator) -> None:
    means, covs = [], []
    for _ in range(4):
        m, c = kf.initiate([*rng.uniform(0, 300, 2), rng.uniform(0.5, 3), rng.uniform(10, 50)])
        m[4:] = rng.normal(0, 1, 4)
        means.append(m)
        covs.append(c)
    batch_m, batch_c = kf.multi_predict(np.stack(means), np.stack(covs))
    for i in range(4):
        single_m, single_c = kf.predict(means[i], covs[i])
        np.testing.assert_allclose(batch_m[i], single_m)
        np.testing.assert_allclose(batch_c[i], single_c)


def test_project_shapes(kf: KalmanFilter) -> None:
    mean, cov = kf.initiate([10.0, 10.0, 1.0, 10.0])
    pm, pc = kf.project(mean, cov)
    assert pm.shape == (4,)
    assert pc.shape == (4, 4)
    assert _is_spd(pc)


def test_update_innovation_sign_and_shrinkage(kf: KalmanFilter) -> None:
    mean, cov = kf.initiate([100.0, 100.0, 1.0, 20.0])
    mean, cov = kf.predict(mean, cov)
    measured = np.array([104.0, 97.0, 1.0, 20.0])
    upd = kf.update(mean, cov, measured)
    # innovation = measured - predicted
    np.testing.assert_allclose(upd.innovation, [4.0, -3.0, 0.0, 0.0], atol=1e-12)
    # posterior lies strictly between prediction and measurement
    assert 100.0 < upd.mean[0] < 104.0
    assert 97.0 < upd.mean[1] < 100.0
    # velocity is pulled in the direction of the surprise
    assert upd.mean[4] > 0.0
    assert upd.mean[5] < 0.0
    assert np.trace(upd.covariance) < np.trace(cov)
    assert _is_spd(upd.covariance)


def test_converges_on_constant_velocity_target(kf: KalmanFilter, rng: np.random.Generator) -> None:
    true_v = np.array([2.0, -1.0])
    pos = np.array([50.0, 200.0])
    mean, cov = kf.initiate([*pos, 0.5, 30.0])
    innovations = []
    for _ in range(120):
        pos = pos + true_v
        mean, cov = kf.predict(mean, cov)
        z = np.array([*(pos + rng.normal(0, 0.5, 2)), 0.5, 30.0])
        upd = kf.update(mean, cov, z)
        mean, cov = upd.mean, upd.covariance
        innovations.append(upd.innovation[:2])
    np.testing.assert_allclose(mean[4:6], true_v, atol=0.1)
    np.testing.assert_allclose(mean[:2], pos, atol=1.0)
    tail = np.array(innovations[60:])
    # once converged the residual is unbiased and of the order of the noise
    np.testing.assert_allclose(tail.mean(axis=0), 0.0, atol=0.25)
    assert 0.3 < float(tail.std()) < 1.5


def test_gating_distance(kf: KalmanFilter) -> None:
    mean, cov = kf.initiate([100.0, 100.0, 1.0, 20.0])
    near = [100.0, 100.0, 1.0, 20.0]
    far = [160.0, 100.0, 1.0, 20.0]
    d = kf.gating_distance(mean, cov, [near, far])
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert d[1] > CHI2INV95[4]
    # only_position ignores a wildly wrong aspect ratio
    weird_aspect = [[100.0, 100.0, 9.0, 20.0]]
    assert kf.gating_distance(mean, cov, weird_aspect)[0] > CHI2INV95[4]
    assert kf.gating_distance(mean, cov, weird_aspect, only_position=True)[0] < CHI2INV95[2]
