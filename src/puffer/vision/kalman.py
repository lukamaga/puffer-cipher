"""Constant-velocity Kalman filter for bounding boxes, as in ByteTrack / DeepSORT.

State (8-dim): ``(cx, cy, a, h, vcx, vcy, va, vh)`` - box centre, aspect ratio
``w / h``, height and their per-frame velocities. Measurement (4-dim):
``(cx, cy, a, h)``. One predict step is one *frame*; callers convert
velocities to pixels per second with the measured frame period.

Process and measurement noise are proportional to the box height (the
``std_weight_*`` constants are ByteTrack's), so the filter behaves the same
for a fish close to the glass and a fish at the back of the tank. Aspect
ratio noise is a small constant because the aspect is dimensionless.

Why this matters for entropy: :meth:`KalmanFilter.update` returns the
*innovation* ``z - H x_pred`` - the part of the observed motion the motion
model could not predict. The fish entropy channel quantises exactly this
residual (``docs/DESIGN.md`` section 4.1); a smooth, predictable swimmer
yields small innovations and correspondingly little credited entropy.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.linalg

FloatArray = npt.NDArray[np.float64]

CHI2INV95: dict[int, float] = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}
"""0.95 quantile of the chi-square distribution by degrees of freedom.

Gate for :meth:`KalmanFilter.gating_distance`: use ``CHI2INV95[4]`` for the
full measurement and ``CHI2INV95[2]`` with ``only_position=True``.
"""

_NDIM = 4


class KalmanUpdate(NamedTuple):
    """Result of a measurement update."""

    mean: FloatArray  # (8,) posterior state
    covariance: FloatArray  # (8, 8) posterior covariance
    innovation: FloatArray  # (4,) measurement minus predicted measurement


class KalmanFilter:
    """Linear Kalman filter with a constant-velocity motion model in xyah space.

    The filter itself is stateless: every method takes and returns
    ``(mean, covariance)`` so a tracker can hold one filter for all tracks
    and batch predictions with :meth:`multi_predict`.
    """

    def __init__(
        self, std_weight_position: float = 1.0 / 20.0, std_weight_velocity: float = 1.0 / 160.0
    ) -> None:
        self.std_weight_position = float(std_weight_position)
        self.std_weight_velocity = float(std_weight_velocity)
        self._motion_mat = np.eye(2 * _NDIM, dtype=np.float64)
        for i in range(_NDIM):
            self._motion_mat[i, _NDIM + i] = 1.0  # dt = one frame
        self._update_mat = np.eye(_NDIM, 2 * _NDIM, dtype=np.float64)

    @property
    def motion_matrix(self) -> FloatArray:
        """State transition matrix ``F`` (a copy)."""
        return self._motion_mat.copy()

    @property
    def observation_matrix(self) -> FloatArray:
        """Measurement matrix ``H`` (a copy)."""
        return self._update_mat.copy()

    def initiate(self, measurement: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Create a track from an unassociated measurement ``(cx, cy, a, h)``.

        Velocities start at zero with a deliberately wide prior (10x the
        per-frame process noise) so the first few updates can learn them.
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(_NDIM)
        mean = np.concatenate([z, np.zeros(_NDIM)])
        h = z[3]
        wp, wv = self.std_weight_position, self.std_weight_velocity
        std = np.array(
            [2 * wp * h, 2 * wp * h, 1e-2, 2 * wp * h, 10 * wv * h, 10 * wv * h, 1e-5, 10 * wv * h]
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: FloatArray, covariance: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Propagate one track by one frame."""
        means, covs = self.multi_predict(mean[None, :], covariance[None, :, :])
        return means[0], covs[0]

    def multi_predict(
        self, means: FloatArray, covariances: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Propagate ``N`` tracks by one frame: ``(N, 8)``, ``(N, 8, 8)`` in and out."""
        m = np.asarray(means, dtype=np.float64).reshape(-1, 2 * _NDIM)
        p = np.asarray(covariances, dtype=np.float64).reshape(-1, 2 * _NDIM, 2 * _NDIM)
        h = m[:, 3]
        ones = np.ones_like(h)
        wp, wv = self.std_weight_position, self.std_weight_velocity
        std = np.stack(
            [wp * h, wp * h, 1e-2 * ones, wp * h, wv * h, wv * h, 1e-5 * ones, wv * h], axis=1
        )
        motion_cov = np.zeros_like(p)
        idx = np.arange(2 * _NDIM)
        motion_cov[:, idx, idx] = np.square(std)
        f = self._motion_mat
        new_means: FloatArray = m @ f.T
        new_covs: FloatArray = f @ p @ f.T + motion_cov
        return new_means, new_covs

    def project(self, mean: FloatArray, covariance: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Map a state distribution into measurement space (adds measurement noise)."""
        h = mean[3]
        wp = self.std_weight_position
        innovation_cov = np.diag(np.square([wp * h, wp * h, 1e-1, wp * h]))
        hm = self._update_mat
        projected_mean: FloatArray = hm @ mean
        projected_cov: FloatArray = hm @ covariance @ hm.T + innovation_cov
        return projected_mean, projected_cov

    def update(
        self, mean: FloatArray, covariance: FloatArray, measurement: npt.ArrayLike
    ) -> KalmanUpdate:
        """Correct a predicted state with measurement ``(cx, cy, a, h)``.

        Uses a Cholesky solve for the gain (the innovation covariance is
        symmetric positive definite), exactly like the reference trackers.
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(_NDIM)
        projected_mean, projected_cov = self.project(mean, covariance)
        factor = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        gain: FloatArray = scipy.linalg.cho_solve(
            factor, (covariance @ self._update_mat.T).T, check_finite=False
        ).T
        innovation: FloatArray = z - projected_mean
        new_mean: FloatArray = mean + gain @ innovation
        new_cov: FloatArray = covariance - gain @ projected_cov @ gain.T
        # Guard against round-off asymmetry accumulating over long tracks.
        new_cov = 0.5 * (new_cov + new_cov.T)
        return KalmanUpdate(new_mean, new_cov, innovation)

    def gating_distance(
        self,
        mean: FloatArray,
        covariance: FloatArray,
        measurements: npt.ArrayLike,
        only_position: bool = False,
    ) -> FloatArray:
        """Squared Mahalanobis distance between the state and ``(M, 4)`` measurements.

        Compare against :data:`CHI2INV95` (4 degrees of freedom, or 2 with
        ``only_position``) to reject physically implausible associations.
        """
        z = np.asarray(measurements, dtype=np.float64).reshape(-1, _NDIM)
        proj_mean, proj_cov = self.project(mean, covariance)
        if only_position:
            proj_mean, proj_cov, z = proj_mean[:2], proj_cov[:2, :2], z[:, :2]
        chol = np.linalg.cholesky(proj_cov)
        d = z - proj_mean
        whitened = scipy.linalg.solve_triangular(chol, d.T, lower=True, check_finite=False)
        out: FloatArray = np.sum(whitened * whitened, axis=0)
        return out
