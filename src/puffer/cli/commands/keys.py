"""``puffer keygen``, ``puffer encrypt`` and ``puffer decrypt`` (DESIGN.md 9.3, 3.1, 3.2).

All cryptography is delegated to :mod:`puffer.crypto` (libsodium). What
these commands add is the user-facing protocol around it:

* **Keys come from the fish.** ``keygen`` (and ``encrypt`` without a key)
  waits for camera entropy with a live panel, then shows which fish
  contributed how many bits. The key file stores that receipt.
* **Nothing is overwritten by accident.** Outputs are checked before the
  slow work and written atomically by the crypto layer (``--force`` to
  replace).
* **A fresh key is never separated from its ciphertext silently.**
  ``encrypt`` without ``-k`` saves the new key next to the output
  (``OUT.key``), says so loudly, and removes it again if encryption fails.
* **Failures say what happened.** Tampering, truncation, wrong keys and
  non-puffer files each have their own message and exit code
  (:mod:`puffer.cli.errors`).
"""

from __future__ import annotations

import enum
import os
import warnings
from pathlib import Path
from typing import Annotated, Any

import typer
from nacl.pwhash import argon2id
from rich.panel import Panel
from rich.text import Text

from puffer.cli.collect import FishKey, make_fish_key, public_random, warm_up
from puffer.cli.commands.common import (
    ForceOption,
    JsonOption,
    MinBitsOption,
    MinFishBitsOption,
    SourceOption,
    StrictOption,
    TimeoutOption,
    check_reachable,
    emit_json,
    open_engine,
    read_password,
    refuse_existing,
)
from puffer.cli.context import PASSWORD_ENV, CliState, get_state
from puffer.cli.errors import CliError, ExitCode
from puffer.cli.render import attribution, human_bytes
from puffer.crypto import (
    InsecurePermissionsWarning,
    Kdf,
    KeyFile,
    decrypt_file,
    encrypt_file,
    load_key,
    read_header,
    save_key,
)

__all__ = ["KdfStrength", "decrypt", "encrypt", "keygen"]

DEFAULT_MIN_BITS = 256.0
DEFAULT_MIN_FISH_BITS = 32.0
DEFAULT_TIMEOUT_S = 60.0
PUF_SUFFIX = ".puf"
KEY_SUFFIX = ".key"
_SALT_WARMUP_FRAMES = 10
_SALT_WARMUP_S = 2.0


class KdfStrength(enum.StrEnum):
    """Argon2id cost presets of libsodium for ``--password`` (``--kdf``)."""

    INTERACTIVE = "interactive"
    MODERATE = "moderate"
    SENSITIVE = "sensitive"

    def limits(self) -> tuple[int, int]:
        """``(opslimit, memlimit)`` of this preset."""
        presets = {
            KdfStrength.INTERACTIVE: (argon2id.OPSLIMIT_INTERACTIVE, argon2id.MEMLIMIT_INTERACTIVE),
            KdfStrength.MODERATE: (argon2id.OPSLIMIT_MODERATE, argon2id.MEMLIMIT_MODERATE),
            KdfStrength.SENSITIVE: (argon2id.OPSLIMIT_SENSITIVE, argon2id.MEMLIMIT_SENSITIVE),
        }
        return presets[self]


KeyOption = Annotated[
    Path | None,
    typer.Option(
        "-k",
        "--key",
        metavar="KEYFILE",
        exists=True,
        dir_okay=False,
        help="Key file (from puffer keygen).",
    ),
]
PasswordOption = Annotated[
    bool,
    typer.Option(
        "--password",
        help=f"Use a password (Argon2id) instead of a key file: prompted with hidden input, "
        f"or read from ${PASSWORD_ENV} for scripts.",
    ),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _load_key_file(state: CliState, path: Path) -> KeyFile:
    """Load a key file, turning the permission warning into a visible CLI warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InsecurePermissionsWarning)
        key_file = load_key(path)
    for warning in caught:
        if issubclass(warning.category, InsecurePermissionsWarning):
            state.err.print(Text(f"warning: {warning.message}", style="yellow"))
    return key_file


def _print_fish_key(state: CliState, fish_key: FishKey, key_id: str) -> None:
    state.out.print(
        attribution(
            fish_key.receipt,
            heading=f"Where the entropy of key {key_id} came from",
            fish=fish_key.fish,
            simulated=fish_key.source.simulated,
        )
    )


def _fish_key(
    state: CliState,
    *,
    source: str | None,
    min_bits: float,
    min_fish_bits: float,
    timeout: float,
    strict: bool,
) -> FishKey:
    config = state.config(source=source)
    reachable = check_reachable(min_bits, config, min_fish_bits)
    with open_engine(state, config) as engine:
        return make_fish_key(
            engine,
            target_bits=min_bits,
            fish_bits=min_fish_bits,
            timeout_s=timeout,
            reachable_bits=reachable,
            strict=strict,
            console=state.err,
            source_spec=config.source,
        )


def _key_saved_panel(path: Path, key_id: str) -> Panel:
    body = Text()
    body.append(f"Key {key_id} saved to {path} (mode 0600).\n", style="bold")
    body.append(
        "Keep it separately from the encrypted file: anyone holding both can decrypt it, "
        "and without the key the data cannot be recovered."
    )
    return Panel(body, title="Keep this key safe", border_style="yellow", expand=False)


# --------------------------------------------------------------------------
# keygen
# --------------------------------------------------------------------------


def keygen(
    ctx: typer.Context,
    out: Annotated[
        Path, typer.Option("-o", "--out", dir_okay=False, help="Key file to write.")
    ] = Path("fish.key"),
    source: SourceOption = None,
    min_bits: MinBitsOption = DEFAULT_MIN_BITS,
    min_fish_bits: MinFishBitsOption = DEFAULT_MIN_FISH_BITS,
    timeout: TimeoutOption = DEFAULT_TIMEOUT_S,
    strict: StrictOption = False,
    force: ForceOption = False,
    json_output: JsonOption = False,
) -> None:
    """Collect camera entropy from the fish and write a new key file.

    Waits (with a live panel) until the sensor-noise and fish-motion channels
    have earned --min-bits of credit, --min-fish-bits of it from the fish
    themselves, then draws a 256-bit key from PufferRNG
    (always keyed with os.urandom) and writes it with its entropy receipt to
    a 0600 key file. Prints which fish contributed how many bits.
    """
    state = get_state(ctx)
    refuse_existing(out, force=force)
    fish_key = _fish_key(
        state,
        source=source,
        min_bits=min_bits,
        min_fish_bits=min_fish_bits,
        timeout=timeout,
        strict=strict,
    )
    receipt = fish_key.receipt_document()
    key_file = save_key(out, fish_key.key, receipt=receipt, overwrite=force)
    if json_output:
        emit_json(
            {
                "key_file": os.fspath(out.resolve()),
                "key_id": key_file.key_id,
                "created": key_file.created.isoformat().replace("+00:00", "Z"),
                "target_met": fish_key.collection.met,
                "receipt": receipt,
            }
        )
        return
    _print_fish_key(state, fish_key, key_file.key_id)
    state.out.print(
        Text.assemble(
            ("Wrote ", ""),
            (os.fspath(out), "bold"),
            (f" (key id {key_file.key_id}, mode 0600).", ""),
        )
    )


# --------------------------------------------------------------------------
# encrypt
# --------------------------------------------------------------------------


def encrypt(
    ctx: typer.Context,
    file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="File to encrypt."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("-o", "--out", dir_okay=False, help="Encrypted output \\[default: FILE.puf]."),
    ] = None,
    key: KeyOption = None,
    password: PasswordOption = False,
    kdf: Annotated[
        KdfStrength,
        typer.Option(
            "--kdf",
            help="Argon2id cost for --password (interactive: 64 MiB, moderate: 256 MiB, "
            "sensitive: 1 GiB).",
        ),
    ] = KdfStrength.MODERATE,
    force: ForceOption = False,
    source: SourceOption = None,
    min_bits: MinBitsOption = DEFAULT_MIN_BITS,
    min_fish_bits: MinFishBitsOption = DEFAULT_MIN_FISH_BITS,
    timeout: TimeoutOption = DEFAULT_TIMEOUT_S,
    strict: StrictOption = False,
) -> None:
    """Encrypt FILE to a .puf file (XChaCha20-Poly1305 secretstream).

    With -k KEYFILE or --password the file is encrypted with that credential.
    With neither, a fresh fish key is generated (like puffer keygen) and saved
    to OUT.key: keep that key file separately from the encrypted file.
    """
    state = get_state(ctx)
    if key is not None and password:
        raise CliError("use either -k/--key or --password, not both", ExitCode.USAGE)
    target = out if out is not None else file.with_name(file.name + PUF_SUFFIX)
    if target.resolve() == file.resolve():
        raise CliError("the output must differ from the input file", ExitCode.USAGE)
    size = file.stat().st_size

    if key is not None:
        refuse_existing(target, force=force)
        key_file = _load_key_file(state, key)
        encrypt_file(file, target, key=key_file.key, overwrite=force)
        _report_encrypted(state, file, target, size, f"key {key_file.key_id}")
        return

    if password:
        refuse_existing(target, force=force)
        secret = read_password(confirm=True)
        opslimit, memlimit = kdf.limits()
        config = state.config(source=source)
        reachable = check_reachable(0.0, config)
        with open_engine(state, config) as engine:
            # The salt is public: drawn with consume=False so it costs no fish credit,
            # but after a few frames so it mixes the tank's current state.
            warm_up(
                engine,
                frames=_SALT_WARMUP_FRAMES,
                timeout_s=_SALT_WARMUP_S,
                reachable_bits=reachable,
            )
            salt = public_random(engine.rng, "salt")
            with state.err.status(f"Deriving the key with Argon2id ({kdf.value})..."):
                encrypt_file(
                    file,
                    target,
                    password=secret,
                    opslimit=opslimit,
                    memlimit=memlimit,
                    random_bytes=salt,
                    overwrite=force,
                )
        _report_encrypted(
            state, file, target, size, f"a password (Argon2id {kdf.value}, salt from PufferRNG)"
        )
        return

    key_path = target.with_name(target.name + KEY_SUFFIX)
    refuse_existing(target, key_path, force=force)
    fish_key = _fish_key(
        state,
        source=source,
        min_bits=min_bits,
        min_fish_bits=min_fish_bits,
        timeout=timeout,
        strict=strict,
    )
    key_file = save_key(
        key_path, fish_key.key, receipt=fish_key.receipt_document(), overwrite=force
    )
    try:
        encrypt_file(file, target, key=fish_key.key, overwrite=force)
    except BaseException:
        # No ciphertext was published (encrypt_file is atomic), so the new key
        # protects nothing: remove it rather than leave an orphan secret behind.
        key_path.unlink(missing_ok=True)
        raise
    _print_fish_key(state, fish_key, key_file.key_id)
    _report_encrypted(state, file, target, size, f"new fish key {key_file.key_id}")
    state.out.print(_key_saved_panel(key_path, key_file.key_id))


def _report_encrypted(state: CliState, src: Path, dst: Path, size: int, credential: str) -> None:
    state.out.print(
        Text.assemble(
            ("Encrypted ", ""),
            (os.fspath(src), "bold"),
            (f" ({human_bytes(size)}) -> ", ""),
            (os.fspath(dst), "bold"),
            (f" with {credential}.", ""),
        )
    )


# --------------------------------------------------------------------------
# decrypt
# --------------------------------------------------------------------------


def _default_plaintext_path(file: Path) -> Path:
    if file.suffix != PUF_SUFFIX or file.name == PUF_SUFFIX:
        raise CliError(
            f"cannot derive an output name for {file} (it does not end in {PUF_SUFFIX})",
            ExitCode.USAGE,
            hint="pass -o/--out",
        )
    return file.with_suffix("")


def decrypt(
    ctx: typer.Context,
    file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Encrypted .puf file."),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "-o", "--out", dir_okay=False, help="Plaintext output \\[default: FILE without .puf]."
        ),
    ] = None,
    key: KeyOption = None,
    password: PasswordOption = False,
    force: ForceOption = False,
) -> None:
    """Verify and decrypt a .puf file.

    The plaintext is written only after every chunk and the final tag have
    been authenticated, so a tampered or truncated file never produces
    output. Without -k or --password the credential is chosen from the file
    header: FILE.key if it exists for key files, a password prompt otherwise.
    """
    state = get_state(ctx)
    if key is not None and password:
        raise CliError("use either -k/--key or --password, not both", ExitCode.USAGE)
    target = out if out is not None else _default_plaintext_path(file)
    if target.resolve() == file.resolve():
        raise CliError("the output must differ from the input file", ExitCode.USAGE)
    refuse_existing(target, force=force)

    with file.open("rb") as handle:
        header = read_header(handle)  # cheap structural check before any prompt
    # Match the credential kind to the header here, so the user gets CLI wording
    # ("use --password") and is never prompted for a password a key file needs.
    credentials: dict[str, Any] = {}
    if header.kdf == Kdf.RAW_KEY:
        if password:
            raise CliError(
                f"{file} is encrypted with key {header.key_id}, not with a password",
                ExitCode.WRONG_KEY,
                hint="pass -k KEYFILE",
            )
        if key is None:
            candidate = file.with_name(file.name + KEY_SUFFIX)
            if not candidate.is_file():
                raise CliError(
                    f"{file} is encrypted with key {header.key_id}",
                    ExitCode.WRONG_KEY,
                    hint=f"pass -k KEYFILE (looked for {candidate})",
                )
            state.err.print(Text(f"using key file {candidate}", style="dim"))
            key = candidate
        key_file = _load_key_file(state, key)
        credentials["key"] = key_file.key
        used = f"key {key_file.key_id}"
    else:
        if key is not None:
            raise CliError(
                f"{file} is password-protected, not encrypted with a key file",
                ExitCode.WRONG_KEY,
                hint=f"use --password (or set ${PASSWORD_ENV})",
            )
        credentials["password"] = read_password(confirm=False)
        used = "a password"

    with state.err.status("Verifying and decrypting..."):
        decrypt_file(file, target, overwrite=force, **credentials)
    state.out.print(
        Text.assemble(
            ("Decrypted ", ""),
            (os.fspath(file), "bold"),
            (" -> ", ""),
            (os.fspath(target), "bold"),
            (f" ({human_bytes(target.stat().st_size)}, authenticated, {used}).", ""),
        )
    )
