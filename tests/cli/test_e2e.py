"""End-to-end: the real engine (seeded simulator) behind the real CLI.

These tests use no fakes: ``puffer.runtime.build_engine`` wires the
simulator, tracker, liveness verifier, health tests, pool and PufferRNG.
They run with ``--liveness off`` for speed (the LED verifier needs ~13 s of
simulated time), except the replay test, which shows why liveness exists.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from puffer.crypto import FileHeader, load_key, read_header

from .conftest import Invoke


def _engine_unavailable() -> str | None:
    for module in ("puffer.config", "puffer.engine", "puffer.runtime"):
        try:
            importlib.import_module(module)
        except Exception as exc:  # any import failure means "not landed yet"
            return f"real engine not importable yet: {module} ({type(exc).__name__}: {exc})"
    return None


_REASON = _engine_unavailable()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")

SOURCE = "synthetic:seed=7"


def _header(path: Path) -> FileHeader:
    with path.open("rb") as handle:
        return read_header(handle)


@pytest.fixture
def run(cli: Invoke, tmp_path: Path) -> Invoke:
    """The CLI with the seeded simulator, no liveness gate, and a private data dir."""

    def invoke(*args: str, **kwargs: object) -> object:
        globals_ = ("--source", SOURCE, "--liveness", "off", "--data-dir", str(tmp_path / "data"))
        return cli(*globals_, *args, **kwargs)

    return invoke  # type: ignore[return-value]


@pytest.fixture
def plain() -> Path:
    path = Path("letter.txt")
    path.write_bytes(b"Dear fish, thank you for the entropy.\n" * 5000)
    return path


def test_keygen_encrypt_decrypt_round_trip(run: Invoke, plain: Path) -> None:
    result = run("keygen", "-o", "fish.key", "--min-fish-bits", "0")
    assert result.exit_code == 0, result.output
    key_file = load_key("fish.key")
    receipt = key_file.receipt
    assert receipt is not None
    assert receipt["collection"]["target_met"] is True
    assert receipt["channels"]["sensor"]["credited_bits"] >= 256
    assert receipt["collection"]["simulated"] is True
    assert "simulated aquarium - fish entropy is simulated" in " ".join(result.stdout.split())

    result = run("encrypt", str(plain), "-k", "fish.key")
    assert result.exit_code == 0, result.output
    assert _header(Path("letter.txt.puf")).key_id == key_file.key_id

    result = run("decrypt", "letter.txt.puf", "-k", "fish.key", "-o", "back.txt")
    assert result.exit_code == 0, result.output
    assert Path("back.txt").read_bytes() == plain.read_bytes()


def test_the_fish_really_contribute(run: Invoke, tmp_path: Path) -> None:
    """With --min-fish-bits the key waits for real tracked fish, then they top the leaderboard."""
    result = run("keygen", "--json", "--min-fish-bits", "8", "--timeout", "60")

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)["receipt"]
    assert receipt["channels"]["fish"]["credited_bits"] >= 8
    fish = receipt["fish"]
    assert len(fish) >= 2
    assert sum(f["share"] for f in fish) == pytest.approx(1.0, abs=0.01)
    assert all(f["bits"] > 0 for f in fish)

    result = run("fish", "--json")  # the engine flushed its statistics on close
    assert result.exit_code == 0, result.output
    board = json.loads(result.stdout)
    identified = {f["fish_id"] for f in fish if not f["fish_id"].startswith("track-")}
    assert {row["fish_id"] for row in board} >= identified


def test_encrypt_with_a_fresh_fish_key(run: Invoke, plain: Path) -> None:
    result = run("encrypt", str(plain), "--min-fish-bits", "0")

    assert result.exit_code == 0, result.output
    assert "Keep it separately from the encrypted file" in result.stdout
    result = run("decrypt", "letter.txt.puf", "-o", "back.txt")
    assert result.exit_code == 0, result.output
    assert "using key file letter.txt.puf.key" in result.stderr
    assert Path("back.txt").read_bytes() == plain.read_bytes()


def test_password_mode_via_environment(run: Invoke, plain: Path) -> None:
    env = {"PUFFER_PASSWORD": "fish are friends"}

    result = run("encrypt", str(plain), "--password", "--kdf", "interactive", env=env)
    assert result.exit_code == 0, result.output
    result = run("decrypt", "letter.txt.puf", "-o", "back.txt", env=env)
    assert result.exit_code == 0, result.output
    assert Path("back.txt").read_bytes() == plain.read_bytes()

    result = run("decrypt", "letter.txt.puf", "-o", "nope.txt", env={"PUFFER_PASSWORD": "sharks"})
    assert result.exit_code == 5
    assert not Path("nope.txt").exists()


def test_tampered_file_is_rejected(run: Invoke, plain: Path) -> None:
    assert run("keygen", "--min-fish-bits", "0").exit_code == 0
    assert run("encrypt", str(plain), "-k", "fish.key").exit_code == 0
    data = bytearray(Path("letter.txt.puf").read_bytes())
    data[-5] ^= 0x40
    Path("letter.txt.puf").write_bytes(bytes(data))

    result = run("decrypt", "letter.txt.puf", "-k", "fish.key", "-o", "out.txt")

    assert result.exit_code == 5
    assert "authentication failed: the file was modified or corrupted" in result.stderr
    assert not Path("out.txt").exists()


def test_random_json_receipt(run: Invoke) -> None:
    result = run("random", "-n", "32", "--json", "--min-bits", "256")

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert len(bytes.fromhex(document["data"])) == 32
    assert document["receipt"]["channels"]["sensor"]["credited_bits"] >= 256
    assert document["receipt"]["collection"]["source_kind"] == "synthetic"


def test_random_stream_into_a_pipe(tmp_path: Path) -> None:
    """``puffer random --stream | head -c N``: exact bytes, quiet exit when the reader leaves."""
    n = 3 * (1 << 20) + 12345
    env = {k: v for k, v in os.environ.items() if not k.startswith("PUFFER_")}
    process = subprocess.Popen(
        [
            sys.executable, "-m", "puffer.cli", "--source", SOURCE, "--liveness", "off",
            "--data-dir", str(tmp_path / "data"), "random", "--stream",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )  # fmt: skip
    assert process.stdout is not None
    assert process.stderr is not None
    data = b""
    while len(data) < n:
        chunk = process.stdout.read(n - len(data))
        if not chunk:
            break
        data += chunk
    process.stdout.close()  # the reader goes away: the writer sees EPIPE
    returncode = process.wait(timeout=60)
    stderr = process.stderr.read().decode()

    assert len(data) == n
    assert len(set(data[:4096])) > 200  # bytes, not a constant
    assert returncode == 0, stderr
    assert "Traceback" not in stderr
    assert "BrokenPipe" not in stderr


def test_info_json_with_the_real_configuration(run: Invoke, tmp_path: Path) -> None:
    result = run("info", "--json")

    assert result.exit_code == 0, result.output
    info = json.loads(result.stdout)
    assert info["components"]["engine"] is True
    assert info["config"]["source"] == SOURCE
    assert info["config"]["liveness"] == "off"
    assert info["data_dir"]["path"] == str((tmp_path / "data").resolve())
    assert info["libraries"]["opencv"]


def test_replayed_video_never_passes_liveness(cli: Invoke, tmp_path: Path) -> None:
    """A recording shows a stale LED challenge: required liveness credits nothing, loudly."""
    small = ("--seconds", "2", "--fps", "15", "--width", "320", "--height", "240", "--seed", "7")
    assert cli("simulate", "-o", "tank.mp4", *small).exit_code == 0
    replay = (
        "--source", "video:tank.mp4,loop=1", "--liveness", "required",
        "--data-dir", str(tmp_path / "data"),
    )  # fmt: skip

    result = cli(*replay, "keygen", "-o", "replay.key", "--timeout", "2")

    assert result.exit_code == 0, result.output
    assert "WARNING: camera entropy target not met" in result.stderr
    assert "liveness never verified" in result.stderr
    receipt = load_key("replay.key").receipt
    assert receipt is not None
    assert receipt["collection"]["target_met"] is False
    assert receipt["collection"]["camera_bits"] == 0.0
    assert receipt["liveness"]["state"] in ("failed", "calibrating")
    assert receipt["channels"]["sensor"]["credited_bits"] == 0.0

    result = cli(*replay, "keygen", "-o", "strict.key", "--timeout", "1", "--strict")
    assert result.exit_code == 8
    assert not Path("strict.key").exists()
