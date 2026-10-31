"""Anti-replay liveness: does the camera see *our* fresh LED challenge right now?

Threat model
------------
The camera feed is the entropy source, so an attacker who can substitute it -
with a recording, a loop, or a frozen frame - would like the system to keep
crediting "fish entropy" that they know. Hashing cannot detect that; freshness
can. The server drives an LED inside the tank with bits drawn from
``os.urandom`` *as time passes* (:class:`~puffer.liveness.schedule.ChallengeSchedule`).
A recording made before a bit was drawn cannot contain it, so a replayed
video matches the challenge only by luck, and we can say exactly how much
luck: that is ``confidence_bits``.

What it does not prove: that the pixels show the real aquarium. An attacker
who watches the LED in real time and composites it into fake video passes;
keeping the LED *inside* the tank makes that require physical access. And by
DESIGN.md rule 2 a successful attack still cannot make outputs weaker than
``os.urandom``; liveness only decides whether fish entropy earns credit.

Algorithm (per frame, cheap)
----------------------------
1. **Locate the LED.** With no ROI, the first ``calibration_s`` seconds are
   used by :mod:`~puffer.liveness.calibrate` to find the region that follows
   the challenge. Those frames are then *discarded*: the bits used for the
   verdict must be independent of the data that picked the region, otherwise
   the region search itself (hundreds of candidates) could manufacture a match.
2. **Observe.** :class:`~puffer.liveness.detector.LedDetector` turns each frame
   into ``(t, on, confident)``; a short history is kept in a ring buffer.
3. **Estimate the camera lag** (periodically, every ``lag_search_interval_s``):
   for every candidate lag ``L`` on a grid over ``[0, max_lag_s]``, the
   fraction of recent confident frames whose state equals
   ``schedule.state_at(t - L)``. The score is flat over the lags that put no
   frame on the wrong side of a transition, so the *centre* of the best
   plateau is taken, and a distant lag must win by ``lag_switch_margin`` to
   replace the current one (stability).
4. **Decode bits** (every frame, ``O(window)``): at the chosen lag every frame
   falls into a chip (half-symbol). Frames within ``edge_guard_s`` of a chip
   edge are skipped (lag uncertainty), the rest vote on/off, and each chip gets
   the vote balance in ``[-1, 1]`` (0 for no votes). A Manchester bit is 1 if
   its first chip is brighter than its second, 0 if darker, and undecodable if
   they tie - a covered or stuck LED decodes to nothing. Comparing the two
   halves of the same bit is what makes this robust to drift, and treating an
   empty chip as neutral lets one good chip still decide the bit.
5. **Verdict** over the last ``window_bits`` complete bits: ``matches / n`` and

   ``confidence_bits = -log2(min(1, P[Binomial(n, 1/2) >= matches] * n_lags))``

   For video that is not causally downstream of the challenge, every decoded
   bit is independent of the fresh challenge bit and matches with probability
   1/2 (an undecodable bit never matches), whatever the attacker recorded.
   The lag was chosen from ``n_lags`` candidates using the same window, which
   the union bound pays for with the ``n_lags`` factor. So
   ``P[a replay reaches this many matches] <= 2^-confidence_bits``, a rigorous
   bound, not a heuristic. Bits are counted, not chips: the two chips of a
   Manchester bit are complementary and are not independent evidence.
   The bound is per window and the window slides by one symbol (0.2 s): at
   the defaults a pass needs >= 58 of 64 bits, i.e. ``confidence_bits >=
   31.7`` with 61 lags at 30 fps, so a non-stop replay is expected to slip
   through for a single window about once in twenty years.

``VERIFIED`` requires ``match_ratio >= min_match``, ``confidence_bits >=
min_confidence_bits`` and a lag within ``[0, max_lag_s]``; ``CALIBRATING``
means the ROI or the first full window is not available yet; anything else is
``FAILED``. After a long failure the ROI is re-learnt (the camera may have
moved), but only if it came from calibration in the first place.

``process`` must be called from one thread (the frame loop); ``status`` and
:meth:`LivenessVerifier.proof` return immutable snapshots and may be read
from any thread.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from puffer.liveness.calibrate import CalibrationAccumulator, CalibrationResult, lag_grid
from puffer.liveness.detector import LedDetector, Roi
from puffer.liveness.proof import LivenessProof
from puffer.liveness.schedule import ChallengeSchedule, ExpiredSymbolError
from puffer.types import Frame, LightMode, LivenessState, LivenessStatus

DISPLAY_BITS: Final = 32
"""Number of most recent bits shown in ``LivenessStatus.expected`` / ``observed``."""

_MIN_LAG_FRAMES: Final = 8  # frames needed before the frame interval / lag can be estimated
_LN2: Final = math.log(2.0)


def confidence_bits(matches: int, n: int, n_lags: int = 1) -> float:
    """``-log2(min(1, P[Binomial(n, 1/2) >= matches] * n_lags))``.

    The binomial tail is summed exactly in log space (``lgamma`` and a
    log-sum-exp), so it neither underflows for long windows nor loses
    precision deep in the tail.
    """
    if n < 0 or not 0 <= matches <= n:
        raise ValueError(f"need 0 <= matches <= n, got matches={matches}, n={n}")
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    return max(0.0, _neg_log2_tail(matches, n) - math.log2(n_lags))


@functools.lru_cache(maxsize=4096)
def _neg_log2_tail(matches: int, n: int) -> float:
    """``-log2 P[Binomial(n, 1/2) >= matches]``.

    Cached because the verifier asks for the same few ``(matches, n)`` pairs
    frame after frame.
    """
    if matches == 0:
        return 0.0
    k = np.arange(matches, n + 1, dtype=np.float64)
    log_terms = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    peak = float(np.max(log_terms))
    log_tail = peak + math.log(float(np.sum(np.exp(log_terms - peak)))) - n * _LN2
    return -log_tail / _LN2


@dataclass(frozen=True, slots=True)
class _Window:
    """One bit-level evaluation (immutable, so it can be shared across threads)."""

    first_symbol: int
    expected: npt.NDArray[np.uint8]
    observed: npt.NDArray[np.int8]  # 1 / 0 / -1 (undecodable)
    matches: int
    lag_s: float
    confidence_bits: float
    start: float
    end: float

    @property
    def n(self) -> int:
        return int(self.expected.size)


class _History:
    """Ring buffer of observations with O(1) amortised append and zero-copy views."""

    def __init__(self, capacity: int = 512) -> None:
        self._t = np.empty(capacity, dtype=np.float64)
        self._on = np.empty(capacity, dtype=np.bool_)
        self._confident = np.empty(capacity, dtype=np.bool_)
        self._lo = 0
        self._hi = 0

    def __len__(self) -> int:
        return self._hi - self._lo

    def append(self, t: float, on: bool, confident: bool) -> None:
        if self._hi == self._t.size:
            self._make_room()
        self._t[self._hi] = t
        self._on[self._hi] = on
        self._confident[self._hi] = confident
        self._hi += 1

    def drop_before(self, t_min: float) -> None:
        self._lo += int(np.searchsorted(self._t[self._lo : self._hi], t_min, side="left"))

    def clear(self) -> None:
        self._lo = self._hi = 0

    def view(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        s = slice(self._lo, self._hi)
        return self._t[s], self._on[s], self._confident[s]

    def _make_room(self) -> None:
        n = len(self)
        capacity = self._t.size * (2 if n > self._t.size // 2 else 1)
        for name in ("_t", "_on", "_confident"):
            old = getattr(self, name)
            new = np.empty(capacity, dtype=old.dtype)
            new[:n] = old[self._lo : self._hi]
            setattr(self, name, new)
        self._lo, self._hi = 0, n


class LivenessVerifier:
    """Decide per frame whether the video shows the live LED challenge.

    Parameters
    ----------
    schedule:
        The challenge the LED is driven with (shared with the LED controller).
    detector:
        An :class:`LedDetector`, an ROI ``(x, y, w, h)``, or ``None`` to locate
        the LED automatically.
    window_bits:
        Number of most recent complete bits the verdict is based on.
    min_match, min_confidence_bits:
        Thresholds for ``VERIFIED`` (match ratio and chance-match bound).
    max_lag_s:
        Largest camera latency accepted (capture + encode + transport).
    lag_step_s:
        Spacing of candidate lags. Default: the measured frame interval divided
        by ``lag_oversample``; finer than a frame keeps enough frames clear of
        chip edges at low frame rates, at a cost of ``log2(lag_oversample)``
        bits of confidence.
    lag_search_interval_s:
        How often (frame time) the lag is re-estimated; decoding runs every frame.
    lag_switch_margin:
        Score advantage a lag more than one frame away from the current one
        needs before the estimate jumps to it.
    edge_guard_s:
        Frames this close to a chip edge do not vote. Default: half a lag step.
    auto_calibrate, calibration_s, calibration_grid, min_calibration_score:
        Automatic ROI search: duration of each attempt, coarse cell size and the
        correlation an ROI needs to be accepted.
    recalibrate_after_s:
        Re-learn a calibrated ROI after this long in ``FAILED`` (``None``: never).
    """

    def __init__(
        self,
        schedule: ChallengeSchedule,
        detector: LedDetector | Roi | None = None,
        *,
        window_bits: int = 64,
        min_match: float = 0.9,
        min_confidence_bits: float = 24.0,
        max_lag_s: float = 0.5,
        lag_step_s: float | None = None,
        lag_oversample: int = 4,
        lag_search_interval_s: float = 0.5,
        lag_switch_margin: float = 0.05,
        edge_guard_s: float | None = None,
        auto_calibrate: bool = True,
        calibration_s: float = 4.0,
        calibration_grid: int = 16,
        min_calibration_score: float = 0.6,
        recalibrate_after_s: float | None = 30.0,
    ) -> None:
        if window_bits < 1:
            raise ValueError("window_bits must be >= 1")
        if not 0.0 < min_match <= 1.0:
            raise ValueError("min_match must be in (0, 1]")
        if min_confidence_bits < 0:
            raise ValueError("min_confidence_bits must be >= 0")
        if not (math.isfinite(max_lag_s) and max_lag_s >= 0):
            raise ValueError("max_lag_s must be a non-negative finite number")
        if lag_step_s is not None and not lag_step_s > 0:
            raise ValueError("lag_step_s must be positive")
        if lag_oversample < 1:
            raise ValueError("lag_oversample must be >= 1")
        if lag_search_interval_s <= 0 or calibration_s <= 0:
            raise ValueError("lag_search_interval_s and calibration_s must be positive")
        if edge_guard_s is not None and not 0 <= edge_guard_s < schedule.chip_s / 2:
            raise ValueError("edge_guard_s must be in [0, chip_s / 2)")
        if recalibrate_after_s is not None and recalibrate_after_s <= 0:
            raise ValueError("recalibrate_after_s must be positive or None")
        span_symbols = window_bits + math.ceil(max_lag_s / schedule.symbol_s) + 2
        if span_symbols > schedule.retention_symbols:
            raise ValueError("window_bits + max_lag_s exceed the schedule's retention")

        self._schedule = schedule
        self._detector = detector if isinstance(detector, LedDetector) else LedDetector(detector)
        if self._detector.roi is None and not auto_calibrate:
            raise ValueError("no ROI given and auto_calibrate is disabled")
        self.window_bits = window_bits
        self.min_match = min_match
        self.min_confidence_bits = min_confidence_bits
        self.max_lag_s = max_lag_s
        self.lag_step_s = lag_step_s
        self.lag_oversample = lag_oversample
        self.lag_search_interval_s = lag_search_interval_s
        self.lag_switch_margin = lag_switch_margin
        self.edge_guard_s = edge_guard_s
        self.auto_calibrate = auto_calibrate
        self.calibration_s = calibration_s
        self.calibration_grid = calibration_grid
        self.min_calibration_score = min_calibration_score
        self.recalibrate_after_s = recalibrate_after_s

        self._window_s = window_bits * schedule.symbol_s
        self._history_span_s = self._window_s + max_lag_s + 2 * schedule.symbol_s
        self._history = _History()
        self._calibration: CalibrationResult | None = None
        self._roi_from_calibration = False
        self.reset()

    # ------------------------------------------------------------- public API

    @property
    def schedule(self) -> ChallengeSchedule:
        return self._schedule

    @property
    def detector(self) -> LedDetector:
        return self._detector

    @property
    def roi(self) -> Roi | None:
        """Current LED region (``None`` while calibrating); exclude it from entropy sampling."""
        return self._detector.roi

    @property
    def calibration(self) -> CalibrationResult | None:
        """Result of the latest calibration attempt, successful or not."""
        return self._calibration

    @property
    def status(self) -> LivenessStatus:
        return self._status

    @property
    def n_lags(self) -> int:
        """Number of lag hypotheses tested (0 until the lag grid is known)."""
        return 0 if self._lags is None else int(self._lags.size)

    def reset(self) -> None:
        """Start over: forget history and lag; re-calibrate unless the ROI was configured."""
        if self._roi_from_calibration:
            self._detector.roi = None
            self._roi_from_calibration = False
        else:
            self._detector.reset()
        self._lags: npt.NDArray[np.float64] | None = None
        self._near_steps = 1
        self._guard_s = 0.0
        self._last_t: float | None = None
        self._mode: LightMode | None = None
        self._forget_observations()
        self._status = LivenessStatus(LivenessState.CALIBRATING)

    def _forget_observations(self) -> None:
        """Drop everything learnt from frames except the ROI and the lag grid."""
        self._calib: CalibrationAccumulator | None = None
        self._calibration_failed = False
        self._next_calibration_try = -math.inf
        self._history.clear()
        self._origin: float | None = None
        self._lag_hint = 0.0
        self._lag_idx: int | None = None
        self._next_lag_search = -math.inf
        self._failed_since: float | None = None
        self._window: _Window | None = None
        self._proof: tuple[_Window, LivenessProof] | None = None

    def process(self, frame: Frame) -> LivenessStatus:
        """Ingest one frame and return the updated status.

        Frames whose timestamp does not increase are ignored (the previous
        status is returned): they carry no new timing information.
        """
        t = float(frame.timestamp)
        if not math.isfinite(t) or (self._last_t is not None and t <= self._last_t):
            return self._status
        self._last_t = t
        if self._mode is not None and frame.mode != self._mode:
            self._detector.reset()  # day/night switch: brightness levels change abruptly
        self._mode = frame.mode
        try:
            if self._detector.roi is None:
                status = self._calibrate(frame, t)
            else:
                status = self._observe(frame, t)
        except ExpiredSymbolError:
            # Timestamps older than the schedule remembers cannot show a fresh challenge.
            status = LivenessStatus(LivenessState.FAILED)
        self._status = status
        self._watch_for_recalibration(status, t)
        return status

    def proof(self) -> LivenessProof | None:
        """Proof for the latest fully evaluated window (``None`` before the first).

        Returned whatever the verdict was: the proof carries the match count
        and confidence, and the beacon publishes the state alongside it.
        """
        window = self._window
        if window is None:
            return None
        cached = self._proof
        if cached is not None and cached[0] is window:
            return cached[1]
        proof = LivenessProof.build(
            window_start=window.start,
            window_end=window.end,
            first_symbol=window.first_symbol,
            challenge=window.expected,
            observed=window.observed,
            lag_s=window.lag_s,
            confidence_bits=window.confidence_bits,
        )
        self._proof = (window, proof)
        return proof

    # ----------------------------------------------------------- calibration

    def _calibrate(self, frame: Frame, t: float) -> LivenessStatus:
        if self._calib is None:
            # Keep a little more than one attempt's worth: with dropped or
            # jittered frames the window would otherwise never span exactly
            # ``calibration_s``; re-tries then slide over the latest frames.
            self._calib = CalibrationAccumulator(
                cell=self.calibration_grid, keep_s=1.5 * self.calibration_s
            )
        acc = self._calib
        acc.add(frame)
        waiting = LivenessStatus(
            LivenessState.FAILED if self._calibration_failed else LivenessState.CALIBRATING
        )
        if acc.span < self.calibration_s or t < self._next_calibration_try:
            return waiting

        # Whatever happens, the next attempt waits: a solve is far too costly per frame.
        self._next_calibration_try = t + self.calibration_s / 2
        try:
            result = acc.solve(self._schedule, max_lag_s=self.max_lag_s, lag_step_s=self.lag_step_s)
        except ExpiredSymbolError:
            self._calibration_failed = True
            raise
        self._calibration = result
        if result.score < self.min_calibration_score:
            self._calibration_failed = True
            return LivenessStatus(LivenessState.FAILED)

        self._detector.roi = result.roi
        times = acc.times
        for ti, bi in zip(times, acc.roi_series(result.roi), strict=True):
            self._detector.update(float(ti), float(bi))  # warm up the levels, no votes
        if self._lags is None:
            self._init_lags(float(np.median(np.diff(times))))
        self._roi_from_calibration = True
        self._start_evaluation(t, lag_hint=result.lag_s)
        return LivenessStatus(LivenessState.CALIBRATING)

    def _watch_for_recalibration(self, status: LivenessStatus, t: float) -> None:
        if status.state is not LivenessState.FAILED:
            self._failed_since = None
            return
        if self._failed_since is None:
            self._failed_since = t
        elif (
            self.recalibrate_after_s is not None
            and self.auto_calibrate
            and self._roi_from_calibration
            and t - self._failed_since >= self.recalibrate_after_s
        ):
            self._detector.roi = None
            self._roi_from_calibration = False
            self._forget_observations()

    # ------------------------------------------------------------ observation

    def _start_evaluation(self, t: float, lag_hint: float = 0.0) -> float:
        """Judge only symbols that start after ``t`` (camera time); return ``t``.

        ``lag_hint`` seeds the lag until the first search succeeds, so that a
        window is evaluated - and fails - even if the LED never shows up.
        """
        self._forget_observations()
        self._origin = t
        self._lag_hint = lag_hint
        return t

    def _init_lags(self, frame_interval: float) -> npt.NDArray[np.float64]:
        """Fix the lag grid once, so that ``n_lags`` (the union bound) never changes.

        ``frame_interval`` is a median of positive timestamp steps (``process``
        drops non-increasing timestamps), hence positive.
        """
        step = (
            self.lag_step_s if self.lag_step_s is not None else frame_interval / self.lag_oversample
        )
        self._lags = lag_grid(self.max_lag_s, step)
        self._near_steps = max(1, round(frame_interval / step))
        chip_s = self._schedule.chip_s
        self._guard_s = (
            self.edge_guard_s if self.edge_guard_s is not None else min(step / 2, chip_s / 4)
        )
        return self._lags

    def _observe(self, frame: Frame, t: float) -> LivenessStatus:
        origin = self._origin if self._origin is not None else self._start_evaluation(t)
        obs = self._detector.observe(frame)
        self._history.append(t, obs.on, obs.confident)
        self._history.drop_before(t - self._history_span_s)
        lags = self._lags
        if lags is None:
            times = self._history.view()[0]
            if times.size < _MIN_LAG_FRAMES:
                return LivenessStatus(LivenessState.CALIBRATING)
            lags = self._init_lags(float(np.median(np.diff(times))))
        if self._lag_idx is None:
            self._lag_idx = int(np.argmin(np.abs(lags - self._lag_hint)))
        if t >= self._next_lag_search:
            self._next_lag_search = t + self.lag_search_interval_s
            self._lag_idx = self._search_lag(t, lags, self._lag_idx)
        window = self._evaluate(t, float(lags[self._lag_idx]), origin)
        if window is not None and window.n >= self.window_bits:
            self._window = window
        return self._judge(window)

    def _search_lag(self, t: float, lags: npt.NDArray[np.float64], current: int) -> int:
        """Return the lag index to use from now on (``current`` if undecided)."""
        times, on, confident = self._history.view()
        use = confident & (times >= t - self._window_s)
        n = int(np.count_nonzero(use))
        if n < _MIN_LAG_FRAMES // 2:
            return current
        ts, ons = times[use], on[use]
        commanded = self._schedule.states_at(ts[:, None] - lags[None, :])
        scores = np.count_nonzero(commanded == ons[:, None], axis=0) / n
        best = int(np.argmax(scores))
        floor = scores[best] - 1.0 / n  # within one frame of the best counts as tied
        lo, hi = best, best
        while lo > 0 and scores[lo - 1] >= floor:
            lo -= 1
        while hi < scores.size - 1 and scores[hi + 1] >= floor:
            hi += 1
        candidate = (lo + hi) // 2
        if (
            lo <= current <= hi
            or abs(candidate - current) <= self._near_steps
            or scores[candidate] - scores[current] > self.lag_switch_margin
        ):
            return candidate
        return current

    def _evaluate(self, t: float, lag: float, origin: float) -> _Window | None:
        """Decode and compare the last ``window_bits`` complete bits at ``lag``."""
        sch = self._schedule
        i_end = sch.symbol_index(t - lag)  # the symbol in progress is not complete yet
        i_first = max(0, sch.symbol_index(origin - lag) + 1)
        i0 = max(i_end - self.window_bits, i_first)
        n = i_end - i0
        if n <= 0:
            return None

        cps = sch.chips_per_symbol
        n_chips = n * cps
        times, on, confident = self._history.view()
        rel = ((times - lag) - sch.t0) / sch.chip_s
        chip = np.floor(rel)
        frac = rel - chip
        idx = chip.astype(np.int64) - i0 * cps
        guard = self._guard_s / sch.chip_s
        use = confident & (idx >= 0) & (idx < n_chips) & (frac >= guard) & (frac <= 1.0 - guard)
        votes = np.where(on[use], 1.0, -1.0)
        sums = np.bincount(idx[use], weights=votes, minlength=n_chips)
        counts = np.bincount(idx[use], minlength=n_chips)
        balance = np.zeros(n_chips, dtype=np.float64)
        np.divide(sums, counts, out=balance, where=counts > 0)
        diff = balance[0::2] - balance[1::2] if cps == 2 else balance
        observed = np.where(diff > 0, 1, np.where(diff < 0, 0, -1)).astype(np.int8)

        expected = sch.bits(i0, i_end)
        matches = int(np.count_nonzero(observed == expected))
        return _Window(
            first_symbol=i0,
            expected=expected,
            observed=observed,
            matches=matches,
            lag_s=lag,
            confidence_bits=confidence_bits(matches, n, self.n_lags),
            start=sch.symbol_start(i0) + lag,
            end=sch.symbol_start(i_end) + lag,
        )

    def _judge(self, window: _Window | None) -> LivenessStatus:
        if window is None:
            return LivenessStatus(LivenessState.CALIBRATING)
        ratio = window.matches / window.n
        if window.n < self.window_bits:
            state = LivenessState.CALIBRATING
        else:
            verified = (
                ratio >= self.min_match
                and window.confidence_bits >= self.min_confidence_bits
                and 0.0 <= window.lag_s <= self.max_lag_s
            )
            state = LivenessState.VERIFIED if verified else LivenessState.FAILED
        shown = slice(-DISPLAY_BITS, None)
        return LivenessStatus(
            state=state,
            match_ratio=ratio,
            lag_s=window.lag_s,
            confidence_bits=window.confidence_bits,
            expected="".join("1" if b else "0" for b in window.expected[shown]),
            observed="".join("?" if o < 0 else str(int(o)) for o in window.observed[shown]),
            window_start=window.start,
            window_end=window.end,
        )
