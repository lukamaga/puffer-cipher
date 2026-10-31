"""``puffer encrypt`` / ``puffer decrypt``: credentials, refusals and every failure class."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from puffer.cli.commands import keys as keys_cmd
from puffer.crypto import FileHeader, Kdf, load_key, read_header, save_key

from .conftest import Invoke, Tank

PLAINTEXT = bytes(range(256)) * 700  # ~175 KiB: three 64 KiB chunks
FAST_KDF = ("--kdf", "interactive")


def _header(path: Path) -> FileHeader:
    with path.open("rb") as handle:
        return read_header(handle)


@pytest.fixture
def plain() -> Path:
    path = Path("secret.txt")
    path.write_bytes(PLAINTEXT)
    return path


@pytest.fixture
def key_path() -> Path:
    path = Path("my.key")
    save_key(path, os.urandom(32))
    return path


def _encrypted_with_key(cli: Invoke, plain: Path, key_path: Path) -> Path:
    result = cli("encrypt", str(plain), "-k", str(key_path))
    assert result.exit_code == 0, result.output
    return Path("secret.txt.puf")


# -------------------------------------------------------------------- encrypt


def test_key_file_round_trip(cli: Invoke, plain: Path, key_path: Path) -> None:
    encrypted = _encrypted_with_key(cli, plain, key_path)
    key_id = load_key(key_path).key_id

    assert _header(encrypted).key_id == key_id
    result = cli("decrypt", str(encrypted), "-k", str(key_path), "-o", "back.txt")

    assert result.exit_code == 0, result.output
    assert Path("back.txt").read_bytes() == PLAINTEXT
    assert f"key {key_id}" in result.stdout
    assert "authenticated" in result.stdout


def test_encrypt_without_credential_makes_and_saves_a_fish_key(
    cli: Invoke, tank: Tank, plain: Path
) -> None:
    result = cli("encrypt", str(plain), "-o", "out.puf")

    assert result.exit_code == 0, result.output
    key_path = Path("out.puf.key")
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    key_file = load_key(key_path)
    assert key_file.receipt is not None
    assert key_file.receipt["collection"]["target_met"]
    assert _header(Path("out.puf")).key_id == key_file.key_id
    assert f"saved to {key_path}" in result.stdout
    assert "Keep it separately from the encrypted file" in result.stdout
    assert "Where the entropy of key" in result.stdout
    assert "Collecting fish entropy" in result.stderr

    # decrypt finds OUT.key next to the file on its own
    result = cli("decrypt", "out.puf", "-o", "back.txt")
    assert result.exit_code == 0, result.output
    assert "using key file out.puf.key" in result.stderr
    assert Path("back.txt").read_bytes() == PLAINTEXT


def test_failed_encryption_removes_the_orphan_key(
    cli: Invoke, tank: Tank, plain: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device", "secret.txt.puf")

    monkeypatch.setattr(keys_cmd, "encrypt_file", broken)

    result = cli("encrypt", str(plain))

    assert result.exit_code == 10
    assert "No space left on device" in result.stderr
    assert not Path("secret.txt.puf.key").exists()


def test_encrypt_refuses_existing_output_or_key_before_collecting(
    cli: Invoke, tank: Tank, plain: Path
) -> None:
    Path("secret.txt.puf.key").write_text("keep me")

    result = cli("encrypt", str(plain))

    assert result.exit_code == 3
    assert "secret.txt.puf.key" in result.stderr
    assert tank.engines == []
    assert Path("secret.txt.puf.key").read_text() == "keep me"


def test_encrypt_force_overwrites(cli: Invoke, plain: Path, key_path: Path) -> None:
    Path("secret.txt.puf").write_bytes(b"old")

    assert cli("encrypt", str(plain), "-k", str(key_path)).exit_code == 3
    result = cli("encrypt", str(plain), "-k", str(key_path), "--force")

    assert result.exit_code == 0, result.output
    assert Path("secret.txt.puf").read_bytes()[:4] == b"PUFR"


def test_key_and_password_are_exclusive(cli: Invoke, plain: Path, key_path: Path) -> None:
    result = cli("encrypt", str(plain), "-k", str(key_path), "--password")

    assert result.exit_code == 2
    assert "either -k/--key or --password" in result.stderr


def test_output_must_differ_from_input(cli: Invoke, plain: Path, key_path: Path) -> None:
    result = cli("encrypt", str(plain), "-k", str(key_path), "-o", str(plain), "--force")

    assert result.exit_code == 2
    assert Path(plain).read_bytes() == PLAINTEXT


def test_missing_input_is_a_usage_error(cli: Invoke) -> None:
    result = cli("encrypt", "nope.txt", "--password")

    assert result.exit_code == 2
    assert "does not exist" in result.output


# -------------------------------------------------------------------- passwords


def test_password_from_environment_round_trip(cli: Invoke, tank: Tank, plain: Path) -> None:
    env = {"PUFFER_PASSWORD": "correct horse battery staple"}

    result = cli("encrypt", str(plain), "--password", *FAST_KDF, env=env)

    assert result.exit_code == 0, result.output
    header = _header(Path("secret.txt.puf"))
    assert header.kdf == Kdf.ARGON2ID
    assert ("salt", 16, "salt") in tank.engine.rng.calls  # public salt: no credit consumed
    assert not any(call[0] == "generate" for call in tank.engine.rng.calls)
    assert tank.engine.collect_calls[0][2] == 10  # warm-up frames before the salt
    assert "password" in result.stdout

    result = cli("decrypt", "secret.txt.puf", "-o", "back.txt", env=env)
    assert result.exit_code == 0, result.output
    assert Path("back.txt").read_bytes() == PLAINTEXT


def test_salt_falls_back_to_generate_on_older_rngs(cli: Invoke, tank: Tank, plain: Path) -> None:
    tank.options.update(with_salt=False)

    result = cli("encrypt", str(plain), "--password", *FAST_KDF, env={"PUFFER_PASSWORD": "pw"})

    assert result.exit_code == 0, result.output
    assert ("generate", 16, "salt") in tank.engine.rng.calls


def test_password_prompt_asks_twice(cli: Invoke, tank: Tank, plain: Path) -> None:
    result = cli("encrypt", str(plain), "--password", *FAST_KDF, input="s3cret\ns3cret\n")

    assert result.exit_code == 0, result.output
    assert "Repeat for confirmation" in result.stderr
    assert "s3cret" not in result.output  # hidden input is never echoed

    result = cli("decrypt", "secret.txt.puf", "-o", "back.txt", input="s3cret\n")
    assert result.exit_code == 0, result.output
    assert Path("back.txt").read_bytes() == PLAINTEXT


def test_mismatched_password_prompts_abort(cli: Invoke, tank: Tank, plain: Path) -> None:
    result = cli("encrypt", str(plain), "--password", input="one\ntwo\n")

    assert result.exit_code != 0
    assert "do not match" in result.stderr
    assert not Path("secret.txt.puf").exists()


def test_empty_password_variable_is_rejected(cli: Invoke, tank: Tank, plain: Path) -> None:
    result = cli("encrypt", str(plain), "--password", env={"PUFFER_PASSWORD": ""})

    assert result.exit_code == 2
    assert "$PUFFER_PASSWORD is set but empty" in result.stderr


def test_wrong_password_fails_authentication(cli: Invoke, tank: Tank, plain: Path) -> None:
    assert (
        cli(
            "encrypt", str(plain), "--password", *FAST_KDF, env={"PUFFER_PASSWORD": "right"}
        ).exit_code
        == 0
    )

    result = cli("decrypt", "secret.txt.puf", "-o", "x", env={"PUFFER_PASSWORD": "wrong"})

    assert result.exit_code == 5
    assert "authentication failed" in result.stderr
    assert not Path("x").exists()


# -------------------------------------------------------------------- decrypt failures


def test_tampered_file_is_rejected_without_output(cli: Invoke, plain: Path, key_path: Path) -> None:
    encrypted = _encrypted_with_key(cli, plain, key_path)
    data = bytearray(encrypted.read_bytes())
    data[70_000] ^= 0x01  # inside chunk 1
    encrypted.write_bytes(bytes(data))

    result = cli("decrypt", str(encrypted), "-k", str(key_path), "-o", "out.txt")

    assert result.exit_code == 5
    assert "the file was modified or corrupted" in result.stderr
    assert "No output was written" in result.stderr
    assert not Path("out.txt").exists()
    assert not any(p.name.startswith(".out.txt") for p in Path().iterdir())  # no temp leftovers


def test_truncated_file_is_reported(cli: Invoke, plain: Path, key_path: Path) -> None:
    encrypted = _encrypted_with_key(cli, plain, key_path)
    encrypted.write_bytes(encrypted.read_bytes()[: 60 + 2 * (65536 + 17)])  # whole chunks only

    result = cli("decrypt", str(encrypted), "-k", str(key_path), "-o", "out.txt")

    assert result.exit_code == 6
    assert "truncated" in result.stderr
    assert not Path("out.txt").exists()


def test_not_a_puffer_file(cli: Invoke, plain: Path, key_path: Path) -> None:
    Path("fake.puf").write_bytes(b"GIF89a" + bytes(100))

    result = cli("decrypt", "fake.puf", "-k", str(key_path))

    assert result.exit_code == 7
    assert "not a valid puffer-cipher file" in result.stderr


def test_wrong_key_names_both_key_ids(cli: Invoke, plain: Path, key_path: Path) -> None:
    encrypted = _encrypted_with_key(cli, plain, key_path)
    other = Path("other.key")
    save_key(other, os.urandom(32))

    result = cli("decrypt", str(encrypted), "-k", str(other), "-o", "out.txt")

    assert result.exit_code == 4
    assert load_key(key_path).key_id in result.stderr
    assert load_key(other).key_id in result.stderr


def test_missing_key_file_is_explained(cli: Invoke, plain: Path, key_path: Path) -> None:
    encrypted = _encrypted_with_key(cli, plain, key_path)

    result = cli("decrypt", str(encrypted), "-o", "out.txt")

    assert result.exit_code == 4
    assert f"encrypted with key {load_key(key_path).key_id}" in result.stderr
    assert "pass -k KEYFILE (looked for secret.txt.puf.key)" in result.stderr


def test_credential_kind_mismatch_uses_cli_wording(
    cli: Invoke, tank: Tank, plain: Path, key_path: Path
) -> None:
    encrypted = _encrypted_with_key(cli, plain, key_path)
    result = cli("decrypt", str(encrypted), "--password", "-o", "x", env={"PUFFER_PASSWORD": "p"})
    assert result.exit_code == 4
    assert "not with a password" in result.stderr

    env = {"PUFFER_PASSWORD": "p"}
    assert (
        cli("encrypt", str(plain), "-o", "pw.puf", "--password", *FAST_KDF, env=env).exit_code == 0
    )
    result = cli("decrypt", "pw.puf", "-k", str(key_path), "-o", "y")
    assert result.exit_code == 4
    assert "password-protected" in result.stderr
    assert "use --password" in result.stderr


def test_default_output_strips_puf(cli: Invoke, plain: Path, key_path: Path) -> None:
    encrypted = _encrypted_with_key(cli, plain, key_path)
    plain.unlink()

    result = cli("decrypt", str(encrypted), "-k", str(key_path))

    assert result.exit_code == 0, result.output
    assert plain.read_bytes() == PLAINTEXT


def test_decrypt_refuses_to_overwrite_and_needs_a_derivable_name(
    cli: Invoke, plain: Path, key_path: Path
) -> None:
    encrypted = _encrypted_with_key(cli, plain, key_path)

    result = cli("decrypt", str(encrypted), "-k", str(key_path))
    assert result.exit_code == 3
    assert plain.read_bytes() == PLAINTEXT

    encrypted.rename("blob.bin")
    result = cli("decrypt", "blob.bin", "-k", str(key_path))
    assert result.exit_code == 2
    assert "pass -o/--out" in result.stderr


def test_world_readable_key_file_is_flagged(cli: Invoke, plain: Path, key_path: Path) -> None:
    key_path.chmod(0o644)

    result = cli("encrypt", str(plain), "-k", str(key_path))

    assert result.exit_code == 0, result.output
    assert "warning:" in result.stderr
    assert "chmod 600" in result.stderr


def test_corrupted_key_file_is_a_format_error(cli: Invoke, plain: Path) -> None:
    Path("bad.key").write_text('{"format": "puffer-key"}')

    result = cli("encrypt", str(plain), "-k", "bad.key")

    assert result.exit_code == 7
    assert "missing fields" in result.stderr
