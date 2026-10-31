"""Anti-replay liveness: a server-controlled LED proves the video is live.

Cloudflare's LavaRand trusts that the camera really films the lava lamps. Here
the server *asks* the tank a question nobody could have known in advance - a
fresh random on/off pattern on an LED inside the aquarium - and only credits
camera entropy while the video answers it, a fraction of a second later.

* :class:`ChallengeSchedule` - fresh random bits, Manchester-coded, as a
  function of time (:class:`puffer.types.LedSchedule`).
* :class:`LedDetector` / :func:`calibrate_roi` - find the LED and turn each
  frame into an on/off observation.
* :class:`LivenessVerifier` - per-frame :class:`puffer.types.LivenessStatus`
  with a rigorous chance-match bound (``confidence_bits``).
* :class:`LivenessProof` - commitments for the public beacon.
"""

from puffer.liveness.calibrate import (
    CalibrationAccumulator,
    CalibrationResult,
    calibrate_roi,
    commanded_states,
    lag_grid,
)
from puffer.liveness.detector import LedDetector, LedObservation, Roi, roi_brightness
from puffer.liveness.proof import LivenessProof, challenge_commitment, observation_commitment
from puffer.liveness.schedule import ChallengeSchedule, Chips, Encoding, ExpiredSymbolError
from puffer.liveness.verifier import LivenessVerifier, confidence_bits

__all__ = [
    "CalibrationAccumulator",
    "CalibrationResult",
    "ChallengeSchedule",
    "Chips",
    "Encoding",
    "ExpiredSymbolError",
    "LedDetector",
    "LedObservation",
    "LivenessProof",
    "LivenessVerifier",
    "Roi",
    "calibrate_roi",
    "challenge_commitment",
    "commanded_states",
    "confidence_bits",
    "lag_grid",
    "observation_commitment",
    "roi_brightness",
]
