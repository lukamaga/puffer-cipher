"""Camera sensor model: shot noise, read noise and optional JPEG.

The sensor entropy channel harvests temporal pixel noise, so the simulator
must produce noise with the right *structure*, not just "some" noise:

* **Shot noise** is Poisson: its variance grows linearly with the signal,
  ``var = gain * I`` (``gain`` in digital numbers per photo-electron). At
  the photon counts of a camera pixel the Gaussian approximation of the
  Poisson distribution is excellent, so it is folded into one Gaussian
  draw per pixel with signal-dependent standard deviation.
* **Read noise** is signal-independent Gaussian with sigma ``read_sigma``.

Hence per pixel ``out = round(I + sqrt(read_sigma^2 + gain * I) * z)``,
saturated to 0..255, with ``z`` standard normal. Night (IR) frames use a
higher read noise and gain (more analogue amplification), like a real camera
at high ISO.

Generating ~1 M Gaussian samples per frame with ``Generator.standard_normal``
costs ~12 ms, a large share of the 16 ms frame budget at 60 fps. Instead
``z`` comes from the inverse normal CDF evaluated at 65536 equiprobable
quantiles, indexed by 16-bit words of the generator's raw output: an exact
Gaussian up to 16-bit quantile resolution (tails truncated at 4.3 sigma, a
2e-5 probability), about six times faster, and still fully determined by the
seed. The sigma map is a per-intensity lookup table.
"""

from __future__ import annotations

import functools
import math
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from scipy.special import ndtri

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]


@functools.cache
def _normal_quantiles() -> F32:
    """65536 equiprobable standard-normal quantiles, rescaled to unit variance."""
    q = ndtri((np.arange(65536, dtype=np.float64) + 0.5) / 65536.0)
    q /= q.std()
    out = q.astype(np.float32)
    out.setflags(write=False)
    return out


class SensorModel:
    """Adds shot + read noise to clean renders, optionally JPEG round-trips.

    ``apply`` accepts a BGR image (day) or a single-channel image (night IR,
    where one draw per pixel keeps the replicated channels identical, exactly
    like a monochrome sensor).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        read_sigma: float,
        shot_gain: float,
        shot_noise: bool = True,
        jpeg_quality: int | None = None,
    ) -> None:
        if read_sigma < 0 or shot_gain < 0:
            raise ValueError("noise parameters must be >= 0")
        self._bitgen = rng.bit_generator
        self.read_sigma = read_sigma
        self.shot_gain = shot_gain if shot_noise else 0.0
        self.jpeg_quality = jpeg_quality
        levels = np.arange(256, dtype=np.float64)
        sigma = np.sqrt(read_sigma**2 + self.shot_gain * levels)
        self._sigma_lut = sigma.astype(np.float32).reshape(1, 256)
        self._quantiles = _normal_quantiles()

    def expected_sigma(self, level: float) -> float:
        """Noise standard deviation (before rounding) at intensity ``level``."""
        return math.sqrt(self.read_sigma**2 + self.shot_gain * level)

    def _standard_normal(self, shape: tuple[int, ...]) -> F32:
        n = math.prod(shape)
        words = self._bitgen.random_raw((n + 3) // 4)  # 4 x 16-bit indices per uint64
        idx = words.view(np.uint16)[:n]
        z = np.take(self._quantiles, idx, mode="clip")
        return z.reshape(shape)

    def apply(self, image: U8) -> U8:
        """Return a noisy (and optionally JPEG-compressed) copy of ``image``."""
        noisy = image
        if self.read_sigma > 0 or self.shot_gain > 0:
            z = self._standard_normal(image.shape)
            sigma = cv2.LUT(image, self._sigma_lut).reshape(image.shape)
            cv2.multiply(z, sigma, dst=z)
            noisy = cast(U8, cv2.add(image, z, dtype=cv2.CV_8U)).reshape(image.shape)
        if self.jpeg_quality is not None:
            ok, buf = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:  # pragma: no cover - OpenCV always encodes uint8 images
                raise RuntimeError("JPEG encoding failed")
            flag = cv2.IMREAD_GRAYSCALE if noisy.ndim == 2 else cv2.IMREAD_COLOR
            noisy = cast(U8, cv2.imdecode(buf, flag))
        return noisy
