"""Fish re-identification: stable identities across track breaks.

ByteTrack ids are short-lived: a fish that hides behind a rock for longer
than the lost buffer, or leaves the field of view, comes back with a new
track id. The dashboard (fish cards, leaderboard) and the entropy receipts
("Nemo contributed 31% of this key") need *who*, not *which track*, so
:class:`FishReID` keeps a persistent gallery of :class:`FishIdentity`
records and binds live tracks to them.

Appearance model
    A 2-D hue-saturation histogram of the pixels inside the detection,
    restricted to an inscribed ellipse (fish are elongated blobs; the box
    corners are water) and, when available, to the detector's foreground
    mask. Value (brightness) is ignored on purpose: it is what lighting
    flicker and day/night drift change most. The hue axis is smoothed
    circularly so a small colour-temperature shift moves mass into a
    neighbouring bin instead of off the histogram. Each track keeps an
    exponential moving average; each identity keeps a slower one, updated
    only on confident matches so one bad crop cannot poison the gallery.

Assignment
    Every frame, confirmed tracks are matched to identities with the
    Hungarian algorithm on the Bhattacharyya distance (one identity per
    track and vice versa). The current binding gets a ``stickiness`` bonus
    (hysteresis), so two similar fish do not flip-flop, yet a genuine
    tracker id swap after a crossing is corrected within a few frames once
    the track histograms have drifted to the other fish.

Enrolment
    A confirmed track that stays unmatched for ``enroll_after`` frames is a
    new fish: it is enrolled with the next unused name from
    :data:`FISH_NAMES` (or the caller's list).

Night / IR frames
    Grayscale frames carry no hue, so appearance is neither learned nor
    matched; existing track bindings simply persist until colour returns.

Per-individual YOLO
    With ``trust_labels=True`` a detection label other than ``"fish"`` *is*
    the identity name (a model trained with one class per fish).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from puffer.types import BBox, Detection
from puffer.vision.detector import is_grayscale
from puffer.vision.geometry import linear_assignment

Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

GALLERY_FORMAT = "puffer-fish-gallery"
GALLERY_VERSION = 1
GENERIC_LABEL = "fish"

FISH_NAMES: tuple[str, ...] = (
    "Nemo",
    "Dory",
    "Bubbles",
    "Gill",
    "Wanda",
    "Goldie",
    "Captain Fin",
    "Sushi",
    "Neptune",
    "Squirt",
    "Pearl",
    "Splash",
    "Mango",
    "Pickles",
    "Biscuit",
    "Zigzag",
    "Noodle",
    "Pixel",
    "Turbo",
    "Blip",
    "Marble",
    "Comet",
    "Wasabi",
    "Admiral Bubbles",
    "Nacho",
    "Pebble",
    "Rocket",
    "Tofu",
    "Jellybean",
    "Professor Puff",
    "Ziggy",
    "Kraken",
)
"""Built-in display names for auto-enrolled fish, used in order."""

_FISH_ID_RE = re.compile(r"^f(\d+)$")
_DISPLAY_VALUE = 230  # HSV value used to render the dominant colour (readable on dark UI)


class ReIDTrack(Protocol):
    """What re-identification needs from a track (satisfied by ``bytetrack.Track``)."""

    @property
    def track_id(self) -> int: ...

    @property
    def is_confirmed(self) -> bool: ...

    @property
    def time_since_update(self) -> int: ...

    @property
    def detection(self) -> Detection: ...


@dataclass(slots=True)
class FishIdentity:
    """One known fish in the gallery.

    ``histogram`` is the L1-normalised, flattened hue x saturation
    histogram (long-term appearance); ``n_obs`` counts the observations
    that contributed to it. ``color_hex`` is the dominant colour, for UI.
    """

    fish_id: str
    name: str
    color_hex: str
    histogram: Float32Array
    n_obs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fish_id": self.fish_id,
            "name": self.name,
            "color_hex": self.color_hex,
            "n_obs": self.n_obs,
            "histogram": [round(float(v), 7) for v in self.histogram],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FishIdentity:
        hist = np.asarray(data["histogram"], dtype=np.float32)
        return cls(
            fish_id=str(data["fish_id"]),
            name=str(data["name"]),
            color_hex=str(data["color_hex"]),
            histogram=_normalise(hist),
            n_obs=int(data.get("n_obs", 0)),
        )


@dataclass(slots=True)
class _TrackAppearance:
    last_frame: int
    hist: Float32Array | None = None
    n_obs: int = 0
    fish_id: str | None = None
    unbound_frames: int = 0
    change: float | None = None  # distance of the latest crop from the model before it
    latest: Float32Array | None = None  # histogram of the latest crop (this frame only)


# ---------------------------------------------------------------------------
# appearance primitives
# ---------------------------------------------------------------------------


def _normalise(hist: Float32Array) -> Float32Array:
    total = float(hist.sum())
    if total <= 0.0:
        raise ValueError("histogram has no mass")
    out: Float32Array = (hist / total).astype(np.float32)
    return out


def appearance_histogram(
    image: npt.NDArray[np.uint8],
    bbox: BBox | None = None,
    foreground: BoolArray | None = None,
    bins: tuple[int, int] = (16, 8),
    min_pixels: int = 20,
) -> Float32Array | None:
    """Hue-saturation histogram of the fish inside ``bbox`` (whole image if ``None``).

    Pixels are restricted to the ellipse inscribed in the box and, if a
    full-frame ``foreground`` mask is given and leaves enough pixels, to the
    foreground too. Returns ``None`` when fewer than ``min_pixels`` remain.
    """
    height, width = image.shape[:2]
    if bbox is None:
        bbox = (0.0, 0.0, float(width), float(height))
    x1 = max(0, int(np.floor(bbox[0])))
    y1 = max(0, int(np.floor(bbox[1])))
    x2 = min(width, int(np.ceil(bbox[2])))
    y2 = min(height, int(np.ceil(bbox[3])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.ndim == 2:
        crop = np.asarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), np.uint8)
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 1, -1)
    if foreground is not None:
        refined = mask & foreground[y1:y2, x1:x2].astype(np.uint8)
        if int(refined.sum()) >= min_pixels:
            mask = refined
    if int(mask.sum()) < min_pixels:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, list(bins), [0, 180, 0, 256])
    # Circular smoothing along hue (axis 0): robust to small colour casts.
    hist = 0.5 * hist + 0.25 * (np.roll(hist, 1, axis=0) + np.roll(hist, -1, axis=0))
    return _normalise(hist.astype(np.float32).ravel())


def bhattacharyya_distance(p: npt.ArrayLike, q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Pairwise Bhattacharyya (Hellinger) distance between rows of ``p`` and ``q``.

    ``d = sqrt(1 - sum_i sqrt(p_i q_i))`` for L1-normalised histograms, the
    same quantity as OpenCV's ``HISTCMP_BHATTACHARYYA``: 0 for identical,
    1 for disjoint histograms. Returns an ``(N, M)`` matrix.
    """
    pp = np.sqrt(np.clip(np.atleast_2d(np.asarray(p, dtype=np.float64)), 0.0, None))
    qq = np.sqrt(np.clip(np.atleast_2d(np.asarray(q, dtype=np.float64)), 0.0, None))
    if pp.shape[0] == 0 or qq.shape[0] == 0:
        return np.zeros((pp.shape[0], qq.shape[0]), dtype=np.float64)
    coefficient = pp @ qq.T
    out: npt.NDArray[np.float64] = np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))
    return out


def dominant_color_hex(histogram: npt.ArrayLike, bins: tuple[int, int] = (16, 8)) -> str:
    """``#rrggbb`` of the most populated hue-saturation bin, rendered bright."""
    hist = np.asarray(histogram, dtype=np.float64).reshape(bins)
    h_bin, s_bin = np.unravel_index(int(np.argmax(hist)), hist.shape)
    hue = (int(h_bin) + 0.5) * 180.0 / bins[0]
    sat = (int(s_bin) + 0.5) * 256.0 / bins[1]
    hsv = np.array([[[round(hue), min(255, round(sat)), _DISPLAY_VALUE]]], dtype=np.uint8)
    b, g, r = (int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# re-identification
# ---------------------------------------------------------------------------


class FishReID:
    """Gallery-based fish re-identification.

    Parameters
    ----------
    threshold:
        Maximum Bhattacharyya distance for a track to match an identity.
    enroll_after:
        Consecutive frames a confirmed track must stay unmatched before it
        is enrolled as a new fish.
    names:
        Display names for auto-enrolment (default :data:`FISH_NAMES`); when
        exhausted, fish are called ``"Fish <n>"``.
    trust_labels:
        Use non-generic detection labels as identity names (per-fish YOLO).
    bins:
        Hue x saturation histogram bins.
    track_rate, gallery_rate:
        EMA rates of the per-track and per-identity appearance models.
    stickiness:
        Distance bonus for keeping the current track-identity binding.
    min_track_obs:
        Observations a track needs before it may be matched at all.
    forget_after:
        Frames after which per-track state of an unseen track is dropped.
    min_pixels:
        Minimum masked pixels for a usable histogram.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enroll_after: int = 15,
        names: Sequence[str] | None = None,
        trust_labels: bool = False,
        *,
        bins: tuple[int, int] = (16, 8),
        track_rate: float = 0.1,
        gallery_rate: float = 0.02,
        stickiness: float = 0.1,
        min_track_obs: int = 3,
        forget_after: int = 300,
        min_pixels: int = 20,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if enroll_after < 1:
            raise ValueError("enroll_after must be >= 1")
        self.threshold = float(threshold)
        self.enroll_after = int(enroll_after)
        self.trust_labels = trust_labels
        self.bins = (int(bins[0]), int(bins[1]))
        self.track_rate = float(track_rate)
        self.gallery_rate = float(gallery_rate)
        self.stickiness = float(stickiness)
        self.min_track_obs = int(min_track_obs)
        self.forget_after = int(forget_after)
        self.min_pixels = int(min_pixels)
        self._names = list(names) if names is not None else list(FISH_NAMES)
        self._gallery: dict[str, FishIdentity] = {}
        self._tracks: dict[int, _TrackAppearance] = {}
        self._frame = 0
        self._next_number = 1

    # ------------------------------------------------------------------ views

    @property
    def identities(self) -> dict[str, FishIdentity]:
        """Gallery by ``fish_id`` (a new dict; treat the records as read-only)."""
        return dict(self._gallery)

    def fish_id_for(self, track_id: int) -> str | None:
        """Identity currently bound to ``track_id`` (``None`` if unknown)."""
        state = self._tracks.get(track_id)
        return state.fish_id if state is not None else None

    def appearance_change(self, track_id: int) -> float | None:
        """How different the track's latest crop looked from its own appearance model.

        Bhattacharyya distance between this frame's histogram and the
        track's running histogram *before* this frame was folded in;
        ``None`` without a colour observation this frame (IR, tiny crop) or
        while the model is younger than ``min_track_obs``. A sustained large
        value means the track now follows a different object.
        """
        state = self._tracks.get(track_id)
        return state.change if state is not None else None

    def track_appearance(self, track_id: int, *, latest: bool = False) -> Float32Array | None:
        """The track's running colour histogram, or with ``latest`` this frame's crop alone.

        ``None`` before any colour observation (or, with ``latest``, without
        one this frame: IR, crop too small).
        """
        state = self._tracks.get(track_id)
        if state is None:
            return None
        hist = state.latest if latest else state.hist
        return None if hist is None else hist.copy()

    def forget_track(self, track_id: int) -> None:
        """Drop a track's appearance model and identity binding (it changed objects)."""
        self._tracks.pop(track_id, None)

    def closest_identity(
        self, track_id: int, exclude: Collection[str] = ()
    ) -> tuple[str, float] | None:
        """Best gallery match for a track's appearance so far, without binding it.

        Returns ``(fish_id, Bhattacharyya distance)`` of the closest identity
        not in ``exclude``, or ``None`` when the track has too little
        appearance evidence (fewer than ``min_track_obs`` colour
        observations, e.g. IR frames) or no identity is available. Used to
        recognise a known fish that reappears on a fresh track.
        """
        state = self._tracks.get(track_id)
        if state is None or state.hist is None or state.n_obs < self.min_track_obs:
            return None
        free = [ident for fid, ident in self._gallery.items() if fid not in exclude]
        if not free:
            return None
        dist = bhattacharyya_distance(state.hist, np.stack([ident.histogram for ident in free]))[0]
        best = int(np.argmin(dist))
        return free[best].fish_id, float(dist[best])

    # ----------------------------------------------------------------- update

    def update(
        self,
        image: npt.NDArray[np.uint8],
        tracks: Sequence[ReIDTrack],
        foreground: BoolArray | None = None,
    ) -> dict[int, str | None]:
        """Observe the current frame and return ``{track_id: fish_id | None}``.

        ``tracks`` are the tracker's active tracks; only those updated in
        this frame contribute appearance. ``foreground`` is an optional
        full-frame mask that restricts histograms to fish pixels.
        """
        self._frame += 1
        fresh = [tr for tr in tracks if tr.time_since_update == 0]
        gray = is_grayscale(image)
        frame_hists: dict[int, Float32Array] = {}
        for tr in fresh:
            state = self._tracks.get(tr.track_id)
            if state is None:
                state = self._tracks[tr.track_id] = _TrackAppearance(self._frame)
            state.last_frame = self._frame
            state.change = None
            state.latest = None
            if gray:
                continue
            hist = appearance_histogram(
                image, tr.detection.bbox, foreground, self.bins, self.min_pixels
            )
            if hist is None:
                continue
            frame_hists[tr.track_id] = hist
            state.latest = hist
            if state.hist is None:
                state.hist = hist.copy()
            else:
                if state.n_obs >= self.min_track_obs:
                    state.change = float(bhattacharyya_distance(hist, state.hist)[0, 0])
                rate = self.track_rate
                state.hist = _normalise((1.0 - rate) * state.hist + rate * hist)
            state.n_obs += 1
        self._forget_stale()

        if not gray:
            eligible = [tr for tr in fresh if tr.is_confirmed]
            claimed: set[str] = set()
            if self.trust_labels:
                eligible = self._assign_by_label(eligible, frame_hists, claimed)
            self._assign_by_appearance(eligible, frame_hists, claimed)
        return {tr.track_id: self.fish_id_for(tr.track_id) for tr in tracks}

    def _assign_by_label(
        self,
        eligible: list[ReIDTrack],
        frame_hists: Mapping[int, Float32Array],
        claimed: set[str],
    ) -> list[ReIDTrack]:
        """Bind labelled tracks by name; return the tracks left for appearance matching.

        If the detector puts one label on several tracks, the most confident
        detection wins and the others stay unidentified for this frame.
        """
        by_label: dict[str, list[ReIDTrack]] = {}
        rest: list[ReIDTrack] = []
        for tr in eligible:
            label = tr.detection.label
            if label and label != GENERIC_LABEL:
                by_label.setdefault(label, []).append(tr)
            else:
                rest.append(tr)
        for label, group in by_label.items():
            winner = max(group, key=lambda tr: tr.detection.score)
            for tr in group:
                self._tracks[tr.track_id].fish_id = None
            state = self._tracks[winner.track_id]
            identity = self._identity_by_name(label)
            if identity is None:
                if state.hist is None:
                    continue
                identity = self._enroll(state.hist, state.n_obs, name=label)
            state.fish_id = identity.fish_id
            state.unbound_frames = 0
            claimed.add(identity.fish_id)
            hist = frame_hists.get(winner.track_id)
            if hist is not None:
                self._learn(identity, hist)
        return rest

    def _assign_by_appearance(
        self,
        eligible: list[ReIDTrack],
        frame_hists: Mapping[int, Float32Array],
        claimed: set[str],
    ) -> None:
        candidates: list[tuple[ReIDTrack, _TrackAppearance, Float32Array]] = []
        for tr in eligible:
            state = self._tracks[tr.track_id]
            if state.hist is not None and state.n_obs >= self.min_track_obs:
                candidates.append((tr, state, state.hist))
            else:
                state.fish_id = None
        if not candidates:
            return
        free = [ident for fid, ident in self._gallery.items() if fid not in claimed]
        gallery = np.zeros((len(free), self.bins[0] * self.bins[1]), dtype=np.float32)
        for j, ident in enumerate(free):
            gallery[j] = ident.histogram
        raw = bhattacharyya_distance(np.stack([hist for _, _, hist in candidates]), gallery)
        cost = raw.copy()
        for i, (_, state, _) in enumerate(candidates):
            for j, ident in enumerate(free):
                if state.fish_id == ident.fish_id:
                    cost[i, j] -= self.stickiness
        matches, unmatched, _ = linear_assignment(cost, self.threshold)
        for i, j in matches:
            tr, state, _ = candidates[i]
            identity = free[j]
            state.fish_id = identity.fish_id
            state.unbound_frames = 0
            hist = frame_hists.get(tr.track_id)
            if hist is not None and raw[i, j] <= self.threshold:
                self._learn(identity, hist)
        for i in unmatched:
            _, state, hist = candidates[i]
            state.fish_id = None
            state.unbound_frames += 1
            if state.unbound_frames >= self.enroll_after:
                state.fish_id = self._enroll(hist, state.n_obs).fish_id
                state.unbound_frames = 0

    def _learn(self, identity: FishIdentity, hist: Float32Array) -> None:
        # Running mean while the model is young, then a slow EMA.
        rate = max(self.gallery_rate, 1.0 / (identity.n_obs + 1))
        identity.histogram = _normalise((1.0 - rate) * identity.histogram + rate * hist)
        identity.n_obs += 1
        identity.color_hex = dominant_color_hex(identity.histogram, self.bins)

    def _forget_stale(self) -> None:
        horizon = self._frame - self.forget_after
        for track_id in [tid for tid, st in self._tracks.items() if st.last_frame < horizon]:
            del self._tracks[track_id]

    # ---------------------------------------------------------------- gallery

    def _identity_by_name(self, name: str) -> FishIdentity | None:
        return next((ident for ident in self._gallery.values() if ident.name == name), None)

    def _new_fish_id(self) -> str:
        # Invariant (kept by register/load via _bump_number): no gallery id
        # "f<n>" has n >= _next_number, so the next number is always free.
        fish_id = f"f{self._next_number}"
        self._next_number += 1
        return fish_id

    def _next_name(self, fish_id: str) -> str:
        used = {ident.name for ident in self._gallery.values()}
        for name in self._names:
            if name not in used:
                return name
        return f"Fish {fish_id.removeprefix('f')}"

    def _enroll(self, hist: Float32Array, n_obs: int, name: str | None = None) -> FishIdentity:
        fish_id = self._new_fish_id()
        identity = FishIdentity(
            fish_id=fish_id,
            name=name if name is not None else self._next_name(fish_id),
            color_hex=dominant_color_hex(hist, self.bins),
            histogram=hist.copy(),
            n_obs=max(1, n_obs),
        )
        self._gallery[fish_id] = identity
        return identity

    def register(
        self,
        fish_id: str,
        name: str,
        references: npt.NDArray[np.uint8] | Sequence[npt.NDArray[np.uint8]],
        color_hex: str | None = None,
    ) -> FishIdentity:
        """Add or replace a known fish from reference BGR crops (one or several).

        Each crop should show the fish roughly centred and filling the
        image; the inscribed-ellipse mask discards the corners.
        """
        crops = [references] if isinstance(references, np.ndarray) else list(references)
        hists = [
            h
            for crop in crops
            if (h := appearance_histogram(crop, None, None, self.bins, self.min_pixels)) is not None
        ]
        if not hists:
            raise ValueError("no usable reference pixels for re-identification")
        hist = _normalise(np.mean(np.stack(hists), axis=0).astype(np.float32))
        identity = FishIdentity(
            fish_id=fish_id,
            name=name,
            color_hex=color_hex if color_hex is not None else dominant_color_hex(hist, self.bins),
            histogram=hist,
            n_obs=len(hists),
        )
        self._gallery[fish_id] = identity
        self._bump_number(fish_id)
        return identity

    def rename(self, fish_id: str, name: str) -> None:
        """Change a fish's display name (e.g. from ``fish_names`` config overrides)."""
        self._gallery[fish_id].name = name

    def _bump_number(self, fish_id: str) -> None:
        match = _FISH_ID_RE.match(fish_id)
        if match is not None:
            self._next_number = max(self._next_number, int(match.group(1)) + 1)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the gallery as JSON, atomically (temp file + rename)."""
        target = Path(path)
        payload = {
            "format": GALLERY_FORMAT,
            "version": GALLERY_VERSION,
            "bins": list(self.bins),
            "next_number": self._next_number,
            "identities": [ident.to_dict() for ident in self._gallery.values()],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def load(self, path: str | os.PathLike[str]) -> None:
        """Replace the gallery with one saved by :meth:`save`.

        Track bindings are reset: they refer to the previous gallery.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != GALLERY_FORMAT or data.get("version") != GALLERY_VERSION:
            raise ValueError(f"{path}: not a {GALLERY_FORMAT} v{GALLERY_VERSION} file")
        if tuple(data.get("bins", ())) != self.bins:
            raise ValueError(f"{path}: histogram bins {data.get('bins')} != {list(self.bins)}")
        n_bins = self.bins[0] * self.bins[1]
        gallery: dict[str, FishIdentity] = {}
        for item in data.get("identities", []):
            identity = FishIdentity.from_dict(item)
            if identity.histogram.size != n_bins:
                raise ValueError(f"{path}: identity {identity.fish_id} has a malformed histogram")
            gallery[identity.fish_id] = identity
        self._gallery = gallery
        self._tracks.clear()
        self._next_number = max(1, int(data.get("next_number", 1)))
        for fish_id in gallery:
            self._bump_number(fish_id)
