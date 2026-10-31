"""``puffer keygen`` with a scripted tank."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from puffer.crypto import load_key
from puffer.sources.spec import SourceSpecError
from puffer.types import LivenessState

from .conftest import Invoke, Tank


def test_keygen_writes_private_key_file_with_receipt(cli: Invoke, tank: Tank) -> None:
    result = cli("keygen", "-o", "fish.key")

    assert result.exit_code == 0, result.output
    path = Path("fish.key")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    key_file = load_key(path)
    receipt = key_file.receipt
    assert receipt is not None
    assert receipt["purpose"] == "key"
    assert receipt["channels"]["system"]["credited_bits"] == 256.0
    assert {f["name"] for f in receipt["fish"]} == {"Biscuit", "Captain Fin", "Professor Puff"}
    collection = receipt["collection"]
    assert collection["target_met"] is True
    assert collection["stopped"] == "target"
    assert collection["simulated"] is True
    assert collection["camera_bits"] >= 256
    assert collection["fish_bits"] >= 32
    assert tank.engine.closed
    assert ("generate", 32, "key") in tank.engine.rng.calls


def test_keygen_shows_panel_attribution_and_simulator_note(cli: Invoke, tank: Tank) -> None:
    result = cli("keygen")

    assert result.exit_code == 0
    assert "Collecting fish entropy" in result.stderr
    assert "✓ targets reached" in result.stderr
    assert "liveness" in result.stderr
    assert "verified" in result.stderr
    out = result.stdout
    assert f"Where the entropy of key {load_key('fish.key').key_id} came from" in out
    for name in ("Biscuit", "Captain Fin", "Professor Puff", "camera sensor", "os.urandom"):
        assert name in out
    assert "simulated aquarium - fish entropy is simulated; security comes from mixing" in out
    assert "Wrote fish.key" in out


def test_keygen_waits_for_the_fish_not_only_the_sensor(cli: Invoke, tank: Tank) -> None:
    # The sensor alone meets --min-bits in one frame; the fish need 4 frames for 40 bits.
    tank.options.update(sensor_bits=4096.0, fish_bits=10.0)

    result = cli("keygen", "--min-bits", "256", "--min-fish-bits", "40")

    assert result.exit_code == 0, result.output
    receipt = load_key("fish.key").receipt
    assert receipt is not None
    assert receipt["channels"]["fish"]["credited_bits"] == pytest.approx(40.0)
    assert receipt["collection"]["frames"] == 4


def test_keygen_json_is_machine_readable_and_never_contains_the_key(
    cli: Invoke, tank: Tank
) -> None:
    result = cli("keygen", "--json", "-o", "k.key")

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    key_file = load_key("k.key")
    assert document["key_id"] == key_file.key_id
    assert document["key_file"] == str(Path("k.key").resolve())
    assert document["target_met"] is True
    assert document["receipt"]["collection"]["target_met"] is True
    assert json.loads(Path("k.key").read_text())["key"] not in result.stdout


def test_keygen_refuses_to_overwrite_before_collecting(cli: Invoke, tank: Tank) -> None:
    Path("fish.key").write_text("precious")

    result = cli("keygen")

    assert result.exit_code == 3
    assert "refusing to overwrite fish.key" in result.stderr
    assert "--force" in result.stderr
    assert tank.engines == []  # failed fast: no entropy collection was wasted
    assert Path("fish.key").read_text() == "precious"


def test_keygen_force_replaces_the_key_file(cli: Invoke, tank: Tank) -> None:
    Path("fish.key").write_text("old")

    result = cli("keygen", "--force")

    assert result.exit_code == 0, result.output
    assert load_key("fish.key").key_id


def test_liveness_failure_still_yields_a_key_with_a_loud_warning(cli: Invoke, tank: Tank) -> None:
    tank.options.update(liveness=LivenessState.FAILED, frame_delay_s=0.001)

    result = cli("keygen", "--timeout", "0.3")

    assert result.exit_code == 0, result.output
    assert "WARNING: camera entropy target not met" in result.stderr
    assert "liveness never verified (state failed" in result.stderr
    assert "exactly as strong as os.urandom" in result.stderr
    assert "--strict" in result.stderr
    receipt = load_key("fish.key").receipt
    assert receipt is not None
    collection = receipt["collection"]
    assert collection["target_met"] is False
    assert collection["stopped"] == "timeout"
    assert collection["camera_bits"] == 0.0
    assert "os.urandom alone" in collection["warning"]
    assert any("liveness never verified" in reason for reason in collection["reasons"])
    assert receipt["channels"]["sensor"]["credited_bits"] == 0.0
    assert receipt["liveness"]["state"] == "failed"


def test_strict_refuses_to_write_a_key_without_fish_entropy(cli: Invoke, tank: Tank) -> None:
    tank.options.update(liveness=LivenessState.CALIBRATING, frame_delay_s=0.001)

    result = cli("keygen", "--timeout", "0.2", "--strict")

    assert result.exit_code == 8
    assert "camera entropy target not met" in result.stderr
    assert "no key was written" in result.stderr
    assert "liveness never verified (state calibrating" in result.stderr
    assert not Path("fish.key").exists()
    assert not any(call[0] == "generate" for call in tank.engine.rng.calls)
    assert tank.engine.closed


def test_source_that_ends_early_is_reported(cli: Invoke, tank: Tank) -> None:
    tank.options.update(frames=2, source_id="video:tank.mp4")
    tank.config.source = "video:tank.mp4"

    result = cli("keygen")

    assert result.exit_code == 0, result.output
    assert "the frame source ended after 2 frame(s)" in result.stderr
    receipt = load_key("fish.key").receipt
    assert receipt is not None
    assert receipt["collection"]["stopped"] == "source_ended"
    assert receipt["collection"]["simulated"] is False
    assert receipt["collection"]["source_kind"] == "video"
    assert "simulated aquarium" not in result.stdout


def test_zero_targets_draw_immediately(cli: Invoke, tank: Tank) -> None:
    result = cli("keygen", "--min-bits", "0", "--min-fish-bits", "0")

    assert result.exit_code == 0, result.output
    assert tank.engine.index == -1  # no frame was needed
    receipt = load_key("fish.key").receipt
    assert receipt is not None
    assert receipt["collection"]["target_met"] is True


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--min-bits", "9000", "--min-bits 9000 can never be reached"),
        ("--min-fish-bits", "5000", "--min-fish-bits 5000 can never be reached"),
    ],
)
def test_unreachable_targets_are_usage_errors(
    cli: Invoke, tank: Tank, option: str, value: str, message: str
) -> None:
    result = cli("keygen", option, value)

    assert result.exit_code == 2
    assert message in result.stderr
    assert tank.engines == []


def test_bad_source_spec_shows_the_spec_grammar(cli: Invoke, tank: Tank) -> None:
    tank.error = SourceSpecError("unknown source kind 'synthtic' (did you mean 'synthetic'?)")

    result = cli("keygen", "--source", "synthtic")

    assert result.exit_code == 2
    assert "did you mean 'synthetic'" in result.stderr
    assert "Source specs:" in result.stderr
    assert tank.configs[0].source == "synthtic"


def test_ctrl_c_during_collection_writes_nothing(cli: Invoke, tank: Tank) -> None:
    tank.options.update(interrupt_after=1, sensor_bits=1.0)

    result = cli("keygen")

    assert result.exit_code == 130
    assert "interrupted" in result.stderr
    assert not Path("fish.key").exists()
    assert tank.engine.closed


def test_global_options_override_the_configuration(cli: Invoke, tank: Tank) -> None:
    result = cli(
        "--source",
        "synthetic:seed=1",
        "--liveness",
        "off",
        "--data-dir",
        "elsewhere",
        "--config",
        "puffer.toml",
        "keygen",
    )
    assert result.exit_code == 2  # --config must exist
    Path("puffer.toml").write_text("[puffer]\n")

    result = cli(
        "--source", "synthetic:seed=1", "--liveness", "off", "--data-dir", "elsewhere",
        "--config", "puffer.toml", "keygen",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    config = tank.configs[-1]
    assert (config.source, config.liveness, config.data_dir) == (
        "synthetic:seed=1",
        "off",
        Path("elsewhere"),
    )
    assert tank.config_paths[-1] == Path("puffer.toml")


def test_command_source_beats_global_source(cli: Invoke, tank: Tank) -> None:
    result = cli("--source", "synthetic:seed=1", "keygen", "--source", "synthetic:seed=2")

    assert result.exit_code == 0, result.output
    assert tank.configs[-1].source == "synthetic:seed=2"


def test_unidentified_tracks_are_labelled_honestly(cli: Invoke, tank: Tank) -> None:
    tank.options.update(unidentified=True)

    result = cli("keygen")

    assert result.exit_code == 0, result.output
    assert "unidentified track-99" in result.stdout
