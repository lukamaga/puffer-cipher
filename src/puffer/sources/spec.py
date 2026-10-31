"""Parse source specs such as ``"synthetic:fish=8,seed=42"`` into sources.

One short string selects and configures any frame source, so the same
syntax works in the CLI (``--source``), the config file (``source = ...``)
and environment variables (``PUFFER_SOURCE``). Grammar::

    spec    := kind [":" args]
    args    := [positional ","] option ("," option)*
    option  := key "=" value

Conveniences: a bare integer (``"0"``) means ``camera:0`` and a bare path
with a video extension (``"tank.mp4"``) means ``video:tank.mp4``. Video
paths may contain commas; only trailing ``key=value`` items with known keys
are treated as options. Unknown kinds and keys get a "did you mean"
suggestion, and bad values name the offending key.
"""

from __future__ import annotations

import difflib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puffer.sources.camera import OpenCVCamera, Picamera2Camera
from puffer.sources.synthetic import SyntheticAquarium, SyntheticConfig
from puffer.sources.video import VideoFileSource
from puffer.types import FrameSource, LedSchedule

__all__ = ["SPEC_HELP", "SourceSpecError", "open_source"]

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mjpeg", ".mjpg"})


class SourceSpecError(ValueError):
    """The source spec string is malformed or has invalid values."""


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError("expected a boolean (1/0, true/false, yes/no, on/off)")


def _int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise ValueError("expected an integer") from None


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise ValueError("expected a number") from None


def _seed(value: str) -> int | None:
    return None if value.strip().lower() in ("random", "none") else _int(value)


def _mode(value: str) -> str:
    v = value.strip().lower()
    if v not in ("day", "night", "cycle"):
        raise ValueError("expected day, night or cycle")
    return v


def _start(value: str) -> float:
    return time.monotonic() if value.strip().lower() == "now" else _float(value)


def _point(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("expected X:Y")
    return _int(parts[0]), _int(parts[1])


def _jpeg(value: str) -> int | None:
    return None if value.strip().lower() in ("none", "off", "0") else _int(value)


@dataclass(frozen=True, slots=True)
class _Option:
    target: str  # keyword argument / config field
    parse: Callable[[str], Any]
    help: str


_SYNTHETIC: dict[str, _Option] = {
    "fish": _Option("n_fish", _int, "number of fish (0-32)"),
    "seed": _Option("seed", _seed, "integer, or 'random'"),
    "mode": _Option("mode", _mode, "day | night | cycle"),
    "latency": _Option("led_latency_s", _float, "LED camera latency, s"),
    "realtime": _Option("realtime", _bool, "pace frames to fps"),
    "width": _Option("width", _int, "frame width, px"),
    "height": _Option("height", _int, "frame height, px"),
    "fps": _Option("fps", _float, "frames per second"),
    "noise": _Option("noise_sigma_day", _float, "day read noise sigma"),
    "night_noise": _Option("noise_sigma_night", _float, "night read noise sigma"),
    "shot": _Option("shot_noise", _bool, "Poisson shot noise"),
    "jpeg": _Option("jpeg_quality", _jpeg, "JPEG quality 1-100, or none"),
    "bubbles": _Option("bubbles", _bool, "bubbler on/off"),
    "caustics": _Option("caustics", _bool, "caustic light on/off"),
    "day": _Option("cycle_day_s", _float, "cycle mode: day length, s"),
    "night": _Option("cycle_night_s", _float, "cycle mode: night length, s"),
    "start": _Option("start_time", _start, "clock start, s, or 'now' (time.monotonic)"),
    "led": _Option("led_xy", _point, "LED position X:Y"),
    "sway": _Option("plant_sway", _float, "water current strength, 0 = still plants"),
}
_VIDEO: dict[str, _Option] = {
    "loop": _Option("loop", _bool, "restart at the end"),
    "realtime": _Option("realtime", _bool, "pace frames to the file's fps"),
    "start": _Option("start_time", _start, "timestamp of the first frame, s"),
}
_CAMERA: dict[str, _Option] = {
    "ir": _Option("ir", _bool, "camera is in IR (night) mode"),
    "width": _Option("width", _int, "requested width, px"),
    "height": _Option("height", _int, "requested height, px"),
    "fps": _Option("fps", _float, "requested frame rate"),
}
_PICAMERA: dict[str, _Option] = {
    **_CAMERA,
    "denoise": _Option("denoise", _bool, "camera noise reduction (default off)"),
    "exposure": _Option("exposure_us", _int, "fixed exposure, microseconds"),
    "gain": _Option("analogue_gain", _float, "fixed analogue gain"),
}
_KINDS = ("synthetic", "video", "camera", "picamera")


def _options_help(options: dict[str, _Option]) -> str:
    return "\n".join(f"    {key:<12} {opt.help}" for key, opt in options.items())


SPEC_HELP = f"""Source specs:
  synthetic[:key=value,...]    aquarium simulator, e.g. synthetic:fish=8,seed=42,mode=night
{_options_help(_SYNTHETIC)}
  video:PATH[,key=value,...]   video file, e.g. video:tank.mp4,loop=1 (or just tank.mp4)
{_options_help(_VIDEO)}
  camera:INDEX[,key=value,...] OpenCV camera, e.g. camera:0,ir=1,width=1280,height=720 (or just 0)
{_options_help(_CAMERA)}
  picamera[:key=value,...]     Raspberry Pi camera (picamera2)
{_options_help(_PICAMERA)}
"""


def _suggest(word: str, choices: list[str] | tuple[str, ...]) -> str:
    close = difflib.get_close_matches(word, list(choices), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _parse_options(spec: str, items: list[str], table: dict[str, _Option]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        key = key.strip().lower()
        if not sep or not key:
            raise SourceSpecError(f"expected key=value, got {item!r} in source spec {spec!r}")
        option = table.get(key)
        if option is None:
            raise SourceSpecError(
                f"unknown option {key!r} in source spec {spec!r}{_suggest(key, list(table))}; "
                f"valid options: {', '.join(table)}"
            )
        try:
            kwargs[option.target] = option.parse(value.strip())
        except ValueError as exc:
            raise SourceSpecError(
                f"bad value {value!r} for {key!r} in source spec {spec!r}: {exc}"
            ) from None
    return kwargs


def _split_video(spec: str, rest: str) -> tuple[str, list[str]]:
    """Separate trailing known ``key=value`` options from a path that may contain commas."""
    parts = rest.split(",")
    options: list[str] = []
    while len(parts) > 1:
        key, sep, _ = parts[-1].partition("=")
        if not sep or key.strip().lower() not in _VIDEO:
            break
        options.insert(0, parts.pop())
    path = ",".join(parts).strip()
    if not path:
        raise SourceSpecError(f"missing video path in source spec {spec!r} (expected video:PATH)")
    return path, options


def _normalise(spec: str) -> tuple[str, str]:
    text = spec.strip()
    if not text:
        raise SourceSpecError(f"empty source spec; expected one of: {', '.join(_KINDS)}")
    kind, _, rest = text.partition(":")
    kind_l = kind.strip().lower()
    if kind_l in _KINDS:
        return kind_l, rest.strip()
    if text.isdigit():
        return "camera", text
    if Path(_split_video(spec, text)[0]).suffix.lower() in VIDEO_EXTENSIONS:
        return "video", text
    raise SourceSpecError(
        f"unknown source kind {kind.strip()!r} in {spec!r}{_suggest(kind_l, _KINDS)}; "
        f"expected one of: {', '.join(_KINDS)}"
    )


def open_source(spec: str, *, led_schedule: LedSchedule | None = None) -> FrameSource:
    """Open the frame source described by ``spec`` (see :data:`SPEC_HELP`).

    ``led_schedule`` is rendered by the simulator; real cameras film a
    physical LED driven by :class:`puffer.hardware.LedController`, and video
    files show whatever was recorded, so for those it is ignored.

    Raises :class:`SourceSpecError` for malformed specs,
    ``FileNotFoundError`` for a missing video file and
    :class:`puffer.sources.base.SourceError` when a device cannot be opened.
    """
    kind, rest = _normalise(spec)
    items = [p for p in rest.split(",") if p.strip()] if rest else []
    if kind == "synthetic":
        kwargs = _parse_options(spec, items, _SYNTHETIC)
        try:
            config = SyntheticConfig(**kwargs)
        except ValueError as exc:
            raise SourceSpecError(f"invalid simulator settings in {spec!r}: {exc}") from None
        return SyntheticAquarium(config, led_schedule=led_schedule)
    if kind == "video":
        path, option_items = _split_video(spec, rest)
        return VideoFileSource(path, **_parse_options(spec, option_items, _VIDEO))
    if kind == "camera":
        index: int | str = 0
        if items and "=" not in items[0]:
            first = items.pop(0).strip()
            index = int(first) if first.lstrip("-").isdigit() else first
        return OpenCVCamera(index, **_parse_options(spec, items, _CAMERA))
    return Picamera2Camera(**_parse_options(spec, items, _PICAMERA))
