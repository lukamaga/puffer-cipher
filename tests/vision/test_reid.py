"""Appearance model and gallery re-identification."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from puffer.types import BBox, Detection
from puffer.vision.reid import (
    FISH_NAMES,
    FishIdentity,
    FishReID,
    appearance_histogram,
    bhattacharyya_distance,
    dominant_color_hex,
)

from .scenes import BLUE, ORANGE, YELLOW, FishSpec, Scene, linear

W, H = 320, 240


@dataclass
class FakeTrack:
    track_id: int
    detection: Detection
    is_confirmed: bool = True
    time_since_update: int = 0


def tank_image(fish: Sequence[tuple[tuple[int, int, int], BBox]], seed: int = 0) -> np.ndarray:
    """Background with fish drawn as filled ellipses inside the given boxes."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (150, 125, 60), dtype=np.uint8)
    for colour, (x1, y1, x2, y2) in fish:
        centre = (round((x1 + x2) / 2), round((y1 + y2) / 2))
        axes = (round((x2 - x1) / 2), round((y2 - y1) / 2))
        cv2.ellipse(img, centre, axes, 0, 0, 360, colour, -1, cv2.LINE_AA)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def box_at(cx: float, cy: float) -> BBox:
    return (cx - 15, cy - 6, cx + 15, cy + 6)


def observe(
    reid: FishReID, fish: Sequence[tuple[int, tuple[int, int, int], BBox]], frames: int, **kw: str
) -> dict[int, str | None]:
    """Show the same fish for ``frames`` frames; return the last assignment."""
    out: dict[int, str | None] = {}
    for i in range(frames):
        img = tank_image([(c, b) for _, c, b in fish], seed=i)
        tracks = [
            FakeTrack(tid, Detection(b, 0.9, 0, kw.get(str(tid), "fish"))) for tid, _, b in fish
        ]
        out = reid.update(img, tracks)
    return out


# ----------------------------------------------------------------- primitives


def test_histogram_is_normalised_and_masked() -> None:
    img = tank_image([(ORANGE, box_at(100, 100))])
    hist = appearance_histogram(img, box_at(100, 100))
    assert hist is not None
    assert hist.shape == (16 * 8,)
    assert float(hist.sum()) == pytest.approx(1.0)
    water = appearance_histogram(img, box_at(250, 50))
    assert water is not None
    assert bhattacharyya_distance(hist, water)[0, 0] > 0.7
    assert appearance_histogram(img, (10.0, 10.0, 11.0, 11.0)) is None


def test_histogram_uses_foreground_mask() -> None:
    img = tank_image([(ORANGE, box_at(100, 100))])
    fg = np.zeros((H, W), dtype=bool)
    fg[96:104, 92:108] = True  # the fish core only
    masked = appearance_histogram(img, (70.0, 80.0, 130.0, 120.0), fg)
    unmasked = appearance_histogram(img, (70.0, 80.0, 130.0, 120.0))
    reference = appearance_histogram(img, box_at(100, 100))
    assert masked is not None
    assert unmasked is not None
    assert reference is not None
    d_masked = bhattacharyya_distance(masked, reference)[0, 0]
    d_unmasked = bhattacharyya_distance(unmasked, reference)[0, 0]
    assert d_masked < d_unmasked


def test_bhattacharyya_bounds() -> None:
    p = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    d = bhattacharyya_distance(p, p)
    np.testing.assert_allclose(d, [[0.0, 1.0], [1.0, 0.0]], atol=1e-7)
    assert bhattacharyya_distance(np.zeros((0, 4)), p).shape == (0, 2)


def test_bhattacharyya_matches_opencv(rng: np.random.Generator) -> None:
    p = rng.dirichlet(np.ones(32)).astype(np.float32)
    q = rng.dirichlet(np.ones(32)).astype(np.float32)
    ours = bhattacharyya_distance(p, q)[0, 0]
    theirs = cv2.compareHist(p, q, cv2.HISTCMP_BHATTACHARYYA)
    assert ours == pytest.approx(theirs, abs=1e-5)


def test_dominant_colour() -> None:
    img = tank_image([(ORANGE, box_at(100, 100))])
    hist = appearance_histogram(img, box_at(100, 100))
    assert hist is not None
    colour = dominant_color_hex(hist)
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    assert colour.startswith("#")
    assert len(colour) == 7
    assert r > g > b  # orange


# ------------------------------------------------------------ re-identification


def test_enrolment_after_stable_frames() -> None:
    reid = FishReID(enroll_after=15, min_track_obs=3)
    fish = [(1, ORANGE, box_at(80, 80)), (2, BLUE, box_at(220, 160))]
    # min_track_obs - 1 frames to become a candidate, then 15 unmatched frames
    assert observe(reid, fish, 16) == {1: None, 2: None}
    ids = observe(reid, fish, 1)
    assert ids[1] is not None
    assert ids[2] is not None
    assert ids[1] != ids[2]
    names = sorted(ident.name for ident in reid.identities.values())
    assert names == sorted(FISH_NAMES[:2])
    assert reid.fish_id_for(1) == ids[1]
    assert reid.fish_id_for(99) is None


def test_identity_survives_track_loss_and_reentry() -> None:
    reid = FishReID()
    first = observe(reid, [(1, ORANGE, box_at(80, 80)), (2, BLUE, box_at(220, 160))], 30)
    # both fish vanish; later they come back with new track ids and swapped places
    later = observe(reid, [(7, BLUE, box_at(60, 60)), (8, ORANGE, box_at(260, 180))], 5)
    assert later[7] == first[2]
    assert later[8] == first[1]
    assert len(reid.identities) == 2


def test_tentative_tracks_are_not_identified() -> None:
    reid = FishReID(enroll_after=2)
    img = tank_image([(ORANGE, box_at(80, 80))])
    track = FakeTrack(1, Detection(box_at(80, 80), 0.9), is_confirmed=False)
    for _ in range(10):
        assert reid.update(img, [track]) == {1: None}
    assert reid.identities == {}


def test_two_tracks_cannot_share_an_identity() -> None:
    reid = FishReID(enroll_after=5)
    observe(reid, [(1, ORANGE, box_at(80, 80))], 20)
    # a second, identical-looking fish appears: it must get its own identity
    ids = observe(reid, [(1, ORANGE, box_at(80, 80)), (2, ORANGE, box_at(220, 160))], 12)
    assert ids[1] != ids[2]
    assert ids[2] is not None
    assert len(reid.identities) == 2


def test_track_id_swap_is_corrected_by_appearance() -> None:
    reid = FishReID()
    before = observe(reid, [(1, ORANGE, box_at(80, 80)), (2, BLUE, box_at(220, 160))], 30)
    # the tracker swapped ids during a crossing: track 1 now follows the blue fish
    after = observe(reid, [(1, BLUE, box_at(90, 90)), (2, ORANGE, box_at(210, 150))], 30)
    assert after[1] == before[2]
    assert after[2] == before[1]


def test_grayscale_frames_keep_bindings_and_learn_nothing() -> None:
    reid = FishReID()
    ids = observe(reid, [(1, ORANGE, box_at(80, 80))], 25)
    fish_id = ids[1]
    assert fish_id is not None
    hist_before = reid.identities[fish_id].histogram.copy()
    n_before = reid.identities[fish_id].n_obs
    luminance = tank_image([(ORANGE, box_at(80, 80))]).mean(axis=2).astype(np.uint8)
    gray = np.repeat(luminance[..., None], 3, axis=2)
    for _ in range(30):
        assert reid.update(gray, [FakeTrack(1, Detection(box_at(80, 80), 0.9))]) == {1: fish_id}
        # a brand-new track at night stays anonymous: no colour, no enrolment
        assert reid.update(gray, [FakeTrack(5, Detection(box_at(200, 80), 0.9))])[5] is None
    np.testing.assert_array_equal(reid.identities[fish_id].histogram, hist_before)
    assert reid.identities[fish_id].n_obs == n_before
    assert len(reid.identities) == 1


def test_trust_labels_uses_detector_names() -> None:
    reid = FishReID(trust_labels=True)
    fish = [(1, ORANGE, box_at(80, 80)), (2, BLUE, box_at(220, 160))]
    ids = observe(reid, fish, 1, **{"1": "Nemo", "2": "Dory"})
    assert {reid.identities[str(ids[1])].name, reid.identities[str(ids[2])].name} == {
        "Nemo",
        "Dory",
    }
    assert reid.identities[str(ids[1])].name == "Nemo"
    # the label wins even when the track id changes
    again = observe(reid, [(9, ORANGE, box_at(100, 90))], 1, **{"9": "Nemo"})
    assert again[9] == ids[1]


def test_trust_labels_conflict_goes_to_most_confident() -> None:
    reid = FishReID(trust_labels=True)
    img = tank_image([(ORANGE, box_at(80, 80)), (YELLOW, box_at(220, 160))])
    tracks = [
        FakeTrack(1, Detection(box_at(80, 80), 0.6, 3, "Nemo")),
        FakeTrack(2, Detection(box_at(220, 160), 0.95, 3, "Nemo")),
    ]
    ids = reid.update(img, tracks)
    assert ids[1] is None
    assert ids[2] is not None
    assert reid.identities[ids[2]].name == "Nemo"


def test_register_reference_and_match() -> None:
    reid = FishReID()
    crop = tank_image([(YELLOW, (0.0, 0.0, 30.0, 12.0))])[:12, :30]
    identity = reid.register("goldie-1", "Goldie", [crop, crop])
    assert identity.n_obs == 2
    assert reid.identities["goldie-1"].name == "Goldie"
    ids = observe(reid, [(4, YELLOW, box_at(150, 100))], 3)
    assert ids[4] == "goldie-1"
    with pytest.raises(ValueError, match="usable"):
        reid.register("x", "X", np.zeros((1, 1, 3), dtype=np.uint8))


def test_rename() -> None:
    reid = FishReID()
    ids = observe(reid, [(1, ORANGE, box_at(80, 80))], 20)
    reid.rename(str(ids[1]), "Sir Swims-a-lot")
    assert reid.identities[str(ids[1])].name == "Sir Swims-a-lot"


def test_custom_names_and_exhaustion() -> None:
    reid = FishReID(names=["Alpha"], enroll_after=3)
    observe(reid, [(1, ORANGE, box_at(60, 60))], 6)
    observe(reid, [(1, ORANGE, box_at(60, 60)), (2, BLUE, box_at(250, 180))], 6)
    assert sorted(i.name for i in reid.identities.values()) == ["Alpha", "Fish 2"]


def test_save_load_round_trip(tmp_path: Path) -> None:
    reid = FishReID()
    observe(reid, [(1, ORANGE, box_at(80, 80)), (2, BLUE, box_at(220, 160))], 25)
    path = tmp_path / "gallery" / "fish.json"
    reid.save(path)
    data = json.loads(path.read_text())
    assert data["format"] == "puffer-fish-gallery"
    assert len(data["identities"]) == 2
    assert list(tmp_path.joinpath("gallery").iterdir()) == [path]  # no temp files left

    restored = FishReID()
    restored.load(path)
    assert restored.identities.keys() == reid.identities.keys()
    for fish_id, ident in reid.identities.items():
        other = restored.identities[fish_id]
        assert (other.name, other.color_hex, other.n_obs) == (
            ident.name,
            ident.color_hex,
            ident.n_obs,
        )
        np.testing.assert_allclose(other.histogram, ident.histogram, atol=1e-6)
    # identities are recognised straight away and new fish get fresh ids
    ids = observe(restored, [(5, BLUE, box_at(100, 100)), (6, YELLOW, box_at(250, 60))], 20)
    assert ids[5] in reid.identities
    assert ids[6] not in reid.identities
    assert len(restored.identities) == 3


def test_load_rejects_foreign_files(tmp_path: Path) -> None:
    reid = FishReID()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something-else", "version": 1}))
    with pytest.raises(ValueError, match="not a"):
        reid.load(bad)
    other_bins = tmp_path / "bins.json"
    FishReID(bins=(8, 4)).save(other_bins)
    with pytest.raises(ValueError, match="bins"):
        reid.load(other_bins)
    malformed = tmp_path / "malformed.json"
    ident = FishIdentity("f1", "Nemo", "#ff8800", np.ones(5, dtype=np.float32), 3)
    payload = {
        "format": "puffer-fish-gallery",
        "version": 1,
        "bins": [16, 8],
        "identities": [ident.to_dict()],
    }
    malformed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="malformed"):
        reid.load(malformed)


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="threshold"):
        FishReID(threshold=0.0)
    with pytest.raises(ValueError, match="enroll_after"):
        FishReID(enroll_after=0)


def test_stale_track_state_is_forgotten() -> None:
    reid = FishReID(forget_after=5)
    observe(reid, [(1, ORANGE, box_at(80, 80))], 20)
    assert reid.fish_id_for(1) is not None
    observe(reid, [(2, BLUE, box_at(200, 150))], 6)
    assert reid.fish_id_for(1) is None


def test_scene_rendered_fish_are_told_apart() -> None:
    """Rendered (anti-aliased, noisy, moving) fish: one identity each, stable over time."""
    fish = [
        FishSpec("a", ORANGE, 30, 12, linear((40, 60), (20, 5))),
        FishSpec("b", BLUE, 30, 12, linear((250, 180), (-20, -5))),
    ]
    scene = Scene(fish, seed=3)
    reid = FishReID()
    history: dict[str, set[str | None]] = {"a": set(), "b": set()}
    for frame, truth in scene.frames(90):
        tracks = [FakeTrack(i + 1, Detection(gt.bbox, 0.9)) for i, gt in enumerate(truth)]
        ids = reid.update(frame.image, tracks)
        if frame.index >= 30:
            for i, gt in enumerate(truth):
                history[gt.name].add(ids[i + 1])
    assert len(history["a"]) == 1
    assert len(history["b"]) == 1
    assert history["a"] != history["b"]


def test_histogram_edge_cases() -> None:
    gray = tank_image([(ORANGE, box_at(100, 100))]).mean(axis=2).astype(np.uint8)
    hist = appearance_histogram(gray, box_at(100, 100))  # 2-D (IR) input is accepted
    assert hist is not None
    assert float(hist.reshape(16, 8)[:, 0].sum()) == pytest.approx(1.0)  # no saturation
    img = tank_image([(ORANGE, box_at(100, 100))])
    assert appearance_histogram(img, box_at(100, 100), min_pixels=10_000) is None


def test_unusable_crops_are_skipped() -> None:
    reid = FishReID(enroll_after=2, trust_labels=True)
    img = tank_image([(ORANGE, box_at(100, 100))])
    sliver = FakeTrack(1, Detection((100.0, 100.0, 101.0, 140.0), 0.9, 0, "Nemo"))
    for _ in range(10):
        assert reid.update(img, [sliver]) == {1: None}
    assert reid.identities == {}


def test_trust_labels_mixed_with_generic_detections() -> None:
    reid = FishReID(trust_labels=True, enroll_after=5)
    fish = [(1, ORANGE, box_at(80, 80)), (2, BLUE, box_at(220, 160))]
    ids = observe(reid, fish, 12, **{"1": "Nemo"})
    assert reid.identities[str(ids[1])].name == "Nemo"
    assert ids[2] is not None
    assert reid.identities[ids[2]].name == FISH_NAMES[1]  # FISH_NAMES[0] ("Nemo") is taken


def test_registered_ids_are_never_reused() -> None:
    reid = FishReID(enroll_after=3)
    crop = tank_image([(YELLOW, (0.0, 0.0, 30.0, 12.0))])[:12, :30]
    reid.register("f7", "Goldie", crop)
    ids = observe(reid, [(1, BLUE, box_at(200, 150))], 6)
    assert ids[1] == "f8"


def test_failed_save_keeps_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reid = FishReID()
    observe(reid, [(1, ORANGE, box_at(80, 80))], 20)
    path = tmp_path / "fish.json"
    reid.save(path)
    before = path.read_text()

    def explode(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", explode)
    with pytest.raises(OSError, match="disk full"):
        reid.save(path)
    assert path.read_text() == before
    assert list(tmp_path.iterdir()) == [path]


def test_identity_needs_a_non_empty_histogram() -> None:
    record = FishIdentity("f1", "Nemo", "#ff8800", np.ones(4, dtype=np.float32), 2).to_dict()
    assert FishIdentity.from_dict(record).histogram.sum() == pytest.approx(1.0)
    record["histogram"] = [0.0] * 4
    with pytest.raises(ValueError, match="no mass"):
        FishIdentity.from_dict(record)
