"""PufferConfig: DESIGN.md 9.1 defaults, TOML + environment layering, validation."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pytest

from puffer.config import FIELD_KINDS, ConfigError, PufferConfig

DESIGN_DEFAULTS: dict[str, object] = {
    "source": "synthetic",
    "detector": "motion",
    "liveness": "auto",
    "led": "auto",
    "led_roi": None,
    "symbol_s": 0.2,
    "liveness_window_bits": 64,
    "sensor_sites": 4096,
    "safety_factor": 0.5,
    "pool_capacity_bits": 4096,
    "sensor_h_assessed": 2.0,
    "fish_h_assessed": 2.0,
    "min_fish": 1,
    "beacon_enabled": True,
    "beacon_period_s": 60,
    "data_dir": Path("data"),
    "fish_names": {},
    "stream_fps": 10,
    "stream_width": 640,
    "jpeg_quality": 70,
    "host": "127.0.0.1",
    "port": 8000,
    "static_dir": Path("dashboard/dist"),
    "vision_max_width": 640,
}

TOML = """
[puffer]
source = "synthetic:fish=8,seed=42"
liveness = "advisory"
led_roi = [600, 20, 16, 16]
symbol_s = 0.25
beacon_enabled = false
data_dir = "state"
port = 9000

[puffer.fish_names]
f1 = "Nemo"
f2 = "Dory"

[other_tool]
ignored = true
"""


def test_defaults_match_the_design_table() -> None:
    cfg = PufferConfig()
    for name, value in DESIGN_DEFAULTS.items():
        assert getattr(cfg, name) == value, name
    assert set(DESIGN_DEFAULTS) == {f.name for f in dataclasses.fields(PufferConfig)}
    assert set(FIELD_KINDS) == set(DESIGN_DEFAULTS)
    assert cfg.beacon_period_ms == 60_000
    assert cfg.led_gpio_pin is None
    assert cfg.yolo_weights is None


def test_toml_then_environment(tmp_path: Path) -> None:
    path = tmp_path / "puffer.toml"
    path.write_text(TOML)
    env = {
        "PUFFER_PORT": "9100",
        "PUFFER_BEACON_ENABLED": "yes",
        "PUFFER_SAFETY_FACTOR": "0.25",
        "PUFFER_LED": "GPIO:18",
        "PUFFER_LED_ROI": " 1, 2, 3, 4 ",
        "PUFFER_FISH_NAMES": "f3=Bubbles,f4=Gill",
        "PUFFER_DETECTOR": "yolo:~/weights/fish.pt",
        "PUFFER_STATIC_DIR": "web",
        "HOME_UNRELATED": "x",
    }
    cfg = PufferConfig.load(path, env=env)
    assert cfg.source == "synthetic:fish=8,seed=42"  # from TOML
    assert cfg.liveness == "advisory"
    assert cfg.symbol_s == 0.25
    assert cfg.port == 9100
    assert cfg.beacon_enabled is True
    assert cfg.safety_factor == 0.25
    assert cfg.led == "gpio:18"
    assert cfg.led_gpio_pin == 18
    assert cfg.led_roi == (1, 2, 3, 4)
    assert cfg.fish_names == {"f3": "Bubbles", "f4": "Gill"}  # env replaces the table
    assert cfg.yolo_weights == Path("~/weights/fish.pt").expanduser()
    assert cfg.data_dir == tmp_path / "state"  # TOML paths are relative to the file
    assert cfg.static_dir == Path("web")  # environment paths are relative to the CWD


def test_toml_values_alone(tmp_path: Path) -> None:
    path = tmp_path / "puffer.toml"
    path.write_text(TOML)
    cfg = PufferConfig.load(path, env={})
    assert cfg.led_roi == (600, 20, 16, 16)
    assert cfg.fish_names == {"f1": "Nemo", "f2": "Dory"}
    assert cfg.beacon_enabled is False
    assert cfg.port == 9000


def test_environment_json_names_and_config_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text("[puffer]\nmin_fish = 3\n")
    env = {"PUFFER_CONFIG": str(path), "PUFFER_FISH_NAMES": '{"f9": "Kraken"}'}
    cfg = PufferConfig.load(env=env)
    assert cfg.min_fish == 3
    assert cfg.fish_names == {"f9": "Kraken"}
    assert PufferConfig.load(env={"PUFFER_LED_ROI": "auto"}).led_roi is None


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PUFFER_PORT": "80x"}, "PUFFER_PORT: port: expected an integer"),
        ({"PUFFER_PORT": "70000"}, "PUFFER_PORT: port: must be in [0, 65535]"),
        ({"PUFFER_BEACON_ENABLED": "maybe"}, "expected a boolean"),
        ({"PUFFER_SYMBOL_S": "nan"}, "expected a finite number"),
        ({"PUFFER_SAFETY_FACTOR": "0"}, "safety_factor: must be in (0, 1]"),
        ({"PUFFER_LIVENESS": "requird"}, "did you mean 'required'"),
        ({"PUFFER_LED": "gpio:99"}, "gpio:<pin 0-27>"),
        ({"PUFFER_LED_ROI": "1,2,3"}, "expected four integers"),
        ({"PUFFER_LED_ROI": "1,2,0,4"}, "w, h >= 1"),
        ({"PUFFER_DETECTOR": "yolo:"}, "yolo:<weights.pt>"),
        ({"PUFFER_FISH_NAMES": "f1"}, "fish_id=name"),
        ({"PUFFER_FISH_NAMES": '{"f1": 3}'}, "strings to strings"),
        ({"PUFFER_LIVENESS_WINDOW_BITS": "16"}, "liveness_window_bits: must be in [32, 1024]"),
        ({"PUFFER_SOURCE": "  "}, "source: must be a non-empty string"),
    ],
)
def test_environment_errors_name_the_variable(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError) as info:
        PufferConfig.load(env=env)
    assert message in str(info.value)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[puffer]\nprot = 1\n", "unknown setting 'prot' (did you mean 'port'?)"),
        ("[puffer]\nport = '8000'\n", "port: expected an integer"),
        ("[puffer]\nbeacon_enabled = 1\n", "expected a boolean"),
        ("[puffer]\nled_roi = [1, 2, 3.5, 4]\n", "expected four integers"),
        ("port = 8000\n", "must be inside a [puffer] table"),
        ("[puffer\n", "invalid TOML"),
        ("puffer = 3\n", "[puffer] must be a table"),
    ],
)
def test_toml_errors_name_the_file(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError) as info:
        PufferConfig.load(path, env={})
    assert message in str(info.value)
    assert str(path) in str(info.value)


def test_missing_file_and_unknown_environment(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        PufferConfig.load(tmp_path / "nope.toml", env={})
    with caplog.at_level(logging.WARNING, logger="puffer.config"):
        cfg = PufferConfig.load(env={"PUFFER_LIVENES": "off"})
    assert cfg.liveness == "auto"
    assert "did you mean 'liveness'" in caplog.text


def test_direct_construction_and_replace_validate() -> None:
    cfg = PufferConfig(liveness="REQUIRED", led=" none ", detector="Motion")  # type: ignore[arg-type]
    assert (cfg.liveness, cfg.led, cfg.detector) == ("required", "none", "motion")
    with pytest.raises(ConfigError, match="jpeg_quality"):
        dataclasses.replace(cfg, jpeg_quality=0)
    with pytest.raises(ConfigError, match="min_fish: expected an integer"):
        PufferConfig(min_fish=1.5)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="printable"):
        PufferConfig(fish_names={"f1": "bad\nname"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.port = 1  # type: ignore[misc]


def test_to_dict_is_json_safe_and_redacts_credentials() -> None:
    cfg = PufferConfig(
        source="camera:rtsp://admin:hunter2@10.0.0.5/stream",
        led_roi=(1, 2, 3, 4),
        fish_names={"f1": "Nemo"},
    )
    d = cfg.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["source"] == "camera:rtsp://***@10.0.0.5/stream"
    assert d["led_roi"] == [1, 2, 3, 4]
    assert d["data_dir"] == "data"
    assert "hunter2" not in repr(cfg)
    assert "hunter2" not in json.dumps(d)
    assert cfg.source.endswith("hunter2@10.0.0.5/stream")  # the engine still gets the real URL
