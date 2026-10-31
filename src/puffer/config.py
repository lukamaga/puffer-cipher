"""Runtime configuration: one frozen dataclass, a TOML file and environment overrides.

:class:`PufferConfig` holds every knob of DESIGN.md section 9.1. It is loaded in
three layers, each overriding the previous one::

    dataclass defaults  <  TOML file ([puffer] table)  <  PUFFER_<FIELD> environment

so a Docker container can run the defaults, a Raspberry Pi can keep a
``puffer.toml`` next to the data directory, and a one-off experiment can flip a
single value with ``PUFFER_LIVENESS=advisory`` without editing anything.

Design decisions
----------------
* **Validated at construction.** ``__post_init__`` rejects every out-of-range
  value with a :class:`ConfigError` naming the field *and where the value came
  from* (TOML file or environment variable). A typo should stop the server at
  start-up, not surface an hour later as a channel that never earns credit.
* **Typed coercion, explicit table.** Environment variables are strings, TOML
  values are typed; :data:`FIELD_KINDS` says how each field is parsed (bool,
  int, float, path, ROI tuple ``"x,y,w,h"``, name mapping), so ``PUFFER_PORT=80x``
  fails with "expected an integer" instead of a ``TypeError`` deep inside
  uvicorn.
* **Relative paths in a TOML file are relative to that file.** A config file
  describes an installation, so ``data_dir = "data"`` in ``/srv/puffer/puffer.toml``
  means ``/srv/puffer/data`` whatever the working directory is. Defaults and
  environment values stay relative to the working directory.
* **No secrets.** Nothing secret lives in the configuration (keys live in
  ``data_dir`` with mode 0600). :meth:`PufferConfig.to_dict` is safe to show on
  the dashboard or in ``puffer info``; it still redacts credentials embedded in
  a stream URL (``camera:rtsp://user:pass@host/...``).
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Literal, cast, get_args

from puffer.beacon.pulse import MAX_PERIOD_MS

__all__ = [
    "ENV_CONFIG",
    "ENV_PREFIX",
    "FIELD_KINDS",
    "ConfigError",
    "LivenessPolicy",
    "PufferConfig",
]

logger = logging.getLogger(__name__)

ENV_PREFIX: Final = "PUFFER_"
"""Environment overrides are ``PUFFER_<FIELD NAME IN UPPER CASE>``."""

ENV_CONFIG: Final = "PUFFER_CONFIG"
"""Path of the TOML file used by :meth:`PufferConfig.load` when none is passed."""

TOML_TABLE: Final = "puffer"

LivenessPolicy = Literal["auto", "required", "advisory", "off"]
"""``required``: camera credit only while the LED challenge is VERIFIED;
``advisory``: verify and display, never gate credit; ``off``: no verifier;
``auto``: required for the simulator and live cameras, off for video files."""

_LIVENESS_POLICIES: Final = frozenset(get_args(LivenessPolicy))
_LED_WORDS: Final = frozenset({"auto", "none", "simulated"})
_LED_GPIO = re.compile(r"^gpio:(\d{1,2})$")
_MAX_GPIO_PIN: Final = 27  # BCM numbering on the 40-pin header
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")
_TRUE: Final = frozenset({"1", "true", "yes", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "off"})
_NONE_WORDS: Final = frozenset({"", "none", "auto", "null"})
_MAX_NAME_LEN: Final = 64

Roi = tuple[int, int, int, int]


class ConfigError(ValueError):
    """A configuration value is missing, malformed or out of range.

    ``field`` is the offending field (``None`` for file-level problems) and
    ``origin`` where the value came from (a TOML path or an environment
    variable name), so the message can point at the line the user must edit.
    """

    def __init__(self, message: str, *, field: str | None = None, origin: str | None = None):
        self.message = message
        self.field = field
        self.origin = origin
        where = ": ".join(part for part in (origin, field) if part)
        super().__init__(f"{where}: {message}" if where else message)

    def located(self, origin: str) -> ConfigError:
        """The same error, attributed to ``origin`` (keeps an origin already set)."""
        if self.origin is not None:
            return self
        return ConfigError(self.message, field=self.field, origin=origin)


# --------------------------------------------------------------------------
# Value parsers: (raw value, from_env) -> typed value. TOML values arrive typed,
# environment values as strings; each parser accepts both where that is sane.
# --------------------------------------------------------------------------


def _parse_str(value: Any, from_env: bool) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"expected a string, got {type(value).__name__}")
    return value.strip()


def _parse_bool(value: Any, from_env: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and from_env:
        word = value.strip().lower()
        if word in _TRUE:
            return True
        if word in _FALSE:
            return False
    raise ConfigError(f"expected a boolean (true/false, 1/0, yes/no, on/off), got {value!r}")


def _parse_int(value: Any, from_env: bool) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and from_env:
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ConfigError(f"expected an integer, got {value!r}")


def _parse_float(value: Any, from_env: bool) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str) and from_env:
        try:
            result = float(value.strip())
        except ValueError:
            raise ConfigError(f"expected a number, got {value!r}") from None
    else:
        raise ConfigError(f"expected a number, got {value!r}")
    if not math.isfinite(result):
        raise ConfigError(f"expected a finite number, got {value!r}")
    return result


def _parse_path(value: Any, from_env: bool) -> Path:
    text = _parse_str(value, from_env)
    if not text:
        raise ConfigError("expected a path, got an empty string")
    return Path(text).expanduser()


def _parse_roi(value: Any, from_env: bool) -> Roi | None:
    """``[x, y, w, h]`` (TOML), ``"x,y,w,h"`` (either), or ``"auto"``/``"none"`` for None."""
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _NONE_WORDS:
            return None
        parts: list[Any] = [p.strip() for p in text.split(",")]
        try:
            items = [int(p) for p in parts]
        except ValueError:
            raise ConfigError(f"expected 'x,y,w,h' integers, got {value!r}") from None
    elif isinstance(value, list | tuple):
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            raise ConfigError(f"expected four integers [x, y, w, h], got {value!r}")
        items = [int(v) for v in value]
    else:
        raise ConfigError(f"expected [x, y, w, h] or 'x,y,w,h', got {value!r}")
    if len(items) != 4:
        raise ConfigError(f"expected four integers x,y,w,h, got {len(items)}")
    return (items[0], items[1], items[2], items[3])


def _parse_names(value: Any, from_env: bool) -> dict[str, str]:
    """A TOML table, a JSON object, or ``"f1=Nemo,f2=Dory"``."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"invalid JSON object: {exc.msg}") from None
        else:
            pairs = {}
            for item in text.split(","):
                key, sep, name = item.partition("=")
                if not sep:
                    raise ConfigError(f"expected fish_id=name pairs, got {item.strip()!r}")
                pairs[key.strip()] = name.strip()
            value = pairs
    if not isinstance(value, Mapping):
        raise ConfigError(f"expected a mapping of fish_id to name, got {type(value).__name__}")
    out: dict[str, str] = {}
    for key, name in value.items():
        if not isinstance(key, str) or not isinstance(name, str):
            raise ConfigError(f"fish names must map strings to strings, got {key!r}: {name!r}")
        out[key] = name
    return out


FIELD_KINDS: Final[Mapping[str, Callable[[Any, bool], Any]]] = {
    "source": _parse_str,
    "detector": _parse_str,
    "liveness": _parse_str,
    "led": _parse_str,
    "led_roi": _parse_roi,
    "symbol_s": _parse_float,
    "liveness_window_bits": _parse_int,
    "sensor_sites": _parse_int,
    "safety_factor": _parse_float,
    "pool_capacity_bits": _parse_float,
    "sensor_h_assessed": _parse_float,
    "fish_h_assessed": _parse_float,
    "min_fish": _parse_int,
    "beacon_enabled": _parse_bool,
    "beacon_period_s": _parse_float,
    "data_dir": _parse_path,
    "fish_names": _parse_names,
    "vision_max_width": _parse_int,
    "stream_fps": _parse_float,
    "stream_width": _parse_int,
    "jpeg_quality": _parse_int,
    "host": _parse_str,
    "port": _parse_int,
    "static_dir": _parse_path,
}
"""How each field is parsed from TOML / environment values (one entry per field)."""

_PATH_FIELDS: Final = frozenset({"data_dir", "static_dir"})


@dataclass(frozen=True, slots=True)
class PufferConfig:
    """Every runtime setting (DESIGN.md section 9.1, plus ``vision_max_width``).

    Construct directly (tests), or with :meth:`load` (CLI and server). All
    fields are validated in ``__post_init__``; strings that name choices are
    normalised to lower case, paths to :class:`~pathlib.Path`.

    ``vision_max_width`` caps the width of the image the fish tracker sees:
    wider frames are downscaled for tracking (tracking cost grows with the
    pixel count, its accuracy does not) and boxes are scaled back to frame
    pixels. The sensor channel always samples the full-resolution frame.
    ``stream_fps``, ``stream_width`` and ``jpeg_quality`` shape the dashboard
    video stream; they never affect entropy.
    """

    source: str = "synthetic"
    detector: str = "motion"
    liveness: LivenessPolicy = "auto"
    led: str = "auto"
    led_roi: Roi | None = None
    symbol_s: float = 0.2
    liveness_window_bits: int = 64
    sensor_sites: int = 4096
    safety_factor: float = 0.5
    pool_capacity_bits: float = 4096.0
    sensor_h_assessed: float = 2.0
    fish_h_assessed: float = 2.0
    min_fish: int = 1
    beacon_enabled: bool = True
    beacon_period_s: float = 60.0
    data_dir: Path = Path("data")
    fish_names: Mapping[str, str] = field(default_factory=dict)
    vision_max_width: int = 640
    stream_fps: float = 10.0
    stream_width: int = 640
    jpeg_quality: int = 70
    host: str = "127.0.0.1"
    port: int = 8000
    static_dir: Path = Path("dashboard/dist")

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # frozen: normalise in place, once, before anyone sees it
        set_(self, "source", _require_text("source", self.source))
        set_(self, "detector", _normalise_detector(self.detector))
        set_(self, "liveness", _normalise_choice("liveness", self.liveness, _LIVENESS_POLICIES))
        set_(self, "led", _normalise_led(self.led))
        set_(self, "led_roi", _normalise_roi(self.led_roi))
        set_(self, "data_dir", Path(self.data_dir))
        set_(self, "static_dir", Path(self.static_dir))
        set_(self, "fish_names", _normalise_names(self.fish_names))
        set_(self, "host", _require_text("host", self.host))

        _check_range("symbol_s", self.symbol_s, 0.02, 10.0)
        # Below 32 bits even a perfect match cannot reach the verifier's
        # 24-bit chance-match bound once the ~61 lag hypotheses are paid for.
        _check_int("liveness_window_bits", self.liveness_window_bits, 32, 1024)
        _check_int("sensor_sites", self.sensor_sites, 16, 1 << 20)
        _check_range("safety_factor", self.safety_factor, 0.0, 1.0, low_open=True)
        _check_range("pool_capacity_bits", self.pool_capacity_bits, 1.0, float(1 << 24))
        # Health cutoffs are derived from H per uint8 symbol: (0, 8] bits.
        _check_range("sensor_h_assessed", self.sensor_h_assessed, 0.0, 8.0, low_open=True)
        _check_range("fish_h_assessed", self.fish_h_assessed, 0.0, 8.0, low_open=True)
        _check_int("min_fish", self.min_fish, 0, 64)
        if not isinstance(self.beacon_enabled, bool):
            raise ConfigError("expected a boolean", field="beacon_enabled")
        _check_range("beacon_period_s", self.beacon_period_s, 0.001, MAX_PERIOD_MS / 1000)
        _check_int("vision_max_width", self.vision_max_width, 64, 8192)
        _check_range("stream_fps", self.stream_fps, 0.0, 60.0, low_open=True)
        _check_int("stream_width", self.stream_width, 64, 4096)
        _check_int("jpeg_quality", self.jpeg_quality, 1, 100)
        _check_int("port", self.port, 0, 65535)

    def __repr__(self) -> str:
        # Built from to_dict() so credentials in a stream URL never reach a log line.
        items = ", ".join(f"{key}={value!r}" for key, value in self.to_dict().items())
        return f"{type(self).__name__}({items})"

    # ------------------------------------------------------------ derived

    @property
    def beacon_period_ms(self) -> int:
        """Beacon period in whole milliseconds (the pulse format's unit)."""
        return max(1, round(self.beacon_period_s * 1000))

    @property
    def led_gpio_pin(self) -> int | None:
        """BCM pin number for ``led = "gpio:<pin>"``, else ``None``."""
        match = _LED_GPIO.match(self.led)
        return int(match.group(1)) if match else None

    @property
    def yolo_weights(self) -> Path | None:
        """Weights path for ``detector = "yolo:<weights>"``, else ``None`` (motion detector)."""
        if self.detector.startswith("yolo:"):
            return Path(self.detector[len("yolo:") :]).expanduser()
        return None

    # ------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] = os.environ,
    ) -> PufferConfig:
        """Defaults, then the ``[puffer]`` table of ``path``, then ``PUFFER_*`` variables.

        ``path=None`` uses ``$PUFFER_CONFIG`` when set, else no file. Unknown
        TOML keys are errors (with a "did you mean" hint); unknown ``PUFFER_*``
        variables are only logged, because other tools may share the prefix.

        Raises:
            ConfigError: the file is missing or malformed, or any value is
                invalid; the message names the field and its origin.
        """
        if path is None:
            env_path = env.get(ENV_CONFIG, "").strip()
            path = env_path or None
        values: dict[str, Any] = {}
        origins: dict[str, str] = {}
        if path is not None:
            file_values = _read_toml(Path(path))
            values.update(file_values)
            origins.update(dict.fromkeys(file_values, os.fspath(path)))
        env_values, env_origins = _read_env(env)
        values.update(env_values)
        origins.update(env_origins)
        try:
            return cls(**values)
        except ConfigError as exc:
            origin = origins.get(exc.field or "")
            raise (exc.located(origin) if origin else exc) from None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view (paths as strings, ROI as a list); credentials in URLs redacted."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = os.fspath(value)
            elif isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, Mapping):
                value = dict(value)
            out[f.name] = value
        out["source"] = _URL_CREDENTIALS.sub(r"\g<scheme>***@", self.source)
        out["detector"] = _URL_CREDENTIALS.sub(r"\g<scheme>***@", self.detector)
        return out


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------


def _field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(PufferConfig))


def _unknown(name: str, *, origin: str) -> ConfigError:
    close = difflib.get_close_matches(name, _field_names(), n=1)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    return ConfigError(f"unknown setting {name!r}{hint}", origin=origin)


def _read_toml(path: Path) -> dict[str, Any]:
    origin = os.fspath(path)
    try:
        with path.open("rb") as fh:
            document = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError("config file not found", origin=origin) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML: {exc}", origin=origin) from None
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {exc.strerror}", origin=origin) from None
    table = document.get(TOML_TABLE)
    if table is None:
        stray = sorted(k for k in document if k in FIELD_KINDS)
        if stray:
            raise ConfigError(
                f"settings {', '.join(stray)} must be inside a [{TOML_TABLE}] table",
                origin=origin,
            )
        return {}
    if not isinstance(table, dict):
        raise ConfigError(f"[{TOML_TABLE}] must be a table", origin=origin)
    base = path.parent
    values: dict[str, Any] = {}
    for key, raw in table.items():
        parse = FIELD_KINDS.get(key)
        if parse is None:
            raise _unknown(key, origin=origin)
        try:
            value = parse(raw, False)
        except ConfigError as exc:
            raise ConfigError(exc.message, field=key, origin=origin) from None
        if key in _PATH_FIELDS and not value.is_absolute():
            value = base / value
        values[key] = value
    return values


def _read_env(env: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    values: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for var, raw in env.items():
        if not var.startswith(ENV_PREFIX) or var == ENV_CONFIG:
            continue
        name = var[len(ENV_PREFIX) :].lower()
        parse = FIELD_KINDS.get(name)
        if parse is None:
            logger.warning("ignoring %s", _unknown(name, origin=var))
            continue
        try:
            values[name] = parse(raw, True)
        except ConfigError as exc:
            raise ConfigError(exc.message, field=name, origin=var) from None
        origins[name] = var
    return values, origins


# --------------------------------------------------------------------------
# Validation helpers (raise ConfigError naming the field)
# --------------------------------------------------------------------------


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("must be a non-empty string", field=name)
    return value.strip()


def _normalise_choice(name: str, value: object, choices: frozenset[str]) -> Any:
    text = _require_text(name, value).lower()
    if text not in choices:
        close = difflib.get_close_matches(text, sorted(choices), n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        raise ConfigError(
            f"expected one of {', '.join(sorted(choices))}, got {value!r}{hint}", field=name
        )
    return text


def _normalise_detector(value: object) -> str:
    text = _require_text("detector", value)
    if text.lower() == "motion":
        return "motion"
    kind, sep, weights = text.partition(":")
    if kind.lower() == "yolo" and sep and weights.strip():
        return f"yolo:{weights.strip()}"
    raise ConfigError(f"expected 'motion' or 'yolo:<weights.pt>', got {value!r}", field="detector")


def _normalise_led(value: object) -> str:
    text = _require_text("led", value).lower().replace(" ", "")
    if text in _LED_WORDS:
        return text
    match = _LED_GPIO.match(text)
    if match and int(match.group(1)) <= _MAX_GPIO_PIN:
        return text
    raise ConfigError(
        f"expected auto, none, simulated or gpio:<pin 0-{_MAX_GPIO_PIN}>, got {value!r}",
        field="led",
    )


def _normalise_roi(value: object) -> Roi | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ConfigError(f"expected (x, y, w, h), got {value!r}", field="led_roi")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise ConfigError(f"expected four integers, got {value!r}", field="led_roi")
    x, y, w, h = (int(v) for v in value)
    if x < 0 or y < 0 or w < 1 or h < 1:
        raise ConfigError(f"need x, y >= 0 and w, h >= 1, got {value!r}", field="led_roi")
    return (x, y, w, h)


def _normalise_names(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ConfigError("expected a mapping of fish_id to name", field="fish_names")
    out: dict[str, str] = {}
    for key, name in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigError(
                f"fish ids must be non-empty strings, got {key!r}", field="fish_names"
            )
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"name for {key!r} must be a non-empty string", field="fish_names")
        clean = name.strip()
        if len(clean) > _MAX_NAME_LEN or not clean.isprintable():
            raise ConfigError(
                f"name for {key!r} must be printable and at most {_MAX_NAME_LEN} characters",
                field="fish_names",
            )
        out[key.strip()] = clean
    return out


def _check_range(
    name: str, value: object, low: float, high: float, *, low_open: bool = False
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"expected a number, got {value!r}", field=name)
    v = float(value)
    too_low = v <= low if low_open else v < low
    if not math.isfinite(v) or too_low or v > high:
        bracket = "(" if low_open else "["
        raise ConfigError(f"must be in {bracket}{low:g}, {high:g}], got {value!r}", field=name)


def _check_int(name: str, value: object, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"expected an integer, got {value!r}", field=name)
    if not low <= value <= high:
        raise ConfigError(f"must be in [{low}, {high}], got {value!r}", field=name)


def _assert_kinds_cover_fields() -> None:
    missing = set(_field_names()) ^ set(FIELD_KINDS)
    if missing:  # pragma: no cover - a developer error caught at import time
        raise RuntimeError(f"FIELD_KINDS out of sync with PufferConfig: {sorted(missing)}")


_assert_kinds_cover_fields()
