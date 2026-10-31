"""Exit codes and the mapping from library exceptions to friendly messages.

A command line tool is an API for shell scripts, so failures must be
*distinguishable* without parsing English: ``puffer decrypt`` exits 5 when a
file was tampered with and 4 when the wrong key was supplied, and a script
can branch on that. The codes are part of the CLI contract, listed in
``puffer --help`` and never renumbered.

Every library error a user can cause is translated here, in one table, into
a one-line message plus an optional hint ("use --force to overwrite").
Nothing else in the CLI prints tracebacks: an unexpected exception becomes
"internal error" (with the traceback only under ``-v``), because a stack
trace is noise to a user and - worse - can leak values from a crypto call.
"""

from __future__ import annotations

import enum
import sys

from puffer.beacon.pulse import BeaconFormatError
from puffer.crypto.errors import AuthenticationError, FormatError, TruncatedError, WrongKeyError

__all__ = ["EXIT_CODE_HELP", "CliError", "ExitCode", "translate"]


class ExitCode(enum.IntEnum):
    """Process exit status of every ``puffer`` command (stable contract)."""

    OK = 0
    FAILURE = 1
    """Generic failure; for ``beacon verify``: the chain did not verify."""
    USAGE = 2
    """Bad command line (Typer's own code for usage errors)."""
    EXISTS = 3
    """Refusing to overwrite an existing file (pass ``--force``)."""
    WRONG_KEY = 4
    """The key or credential type does not match the encrypted file."""
    AUTHENTICATION = 5
    """Authentication failed: tampered data, or wrong key/password."""
    TRUNCATED = 6
    """The encrypted file is incomplete."""
    FORMAT = 7
    """Not a puffer file, or malformed input."""
    ENTROPY = 8
    """``--strict`` and the camera entropy target was not met."""
    UNAVAILABLE = 9
    """An optional component is not installed or not available."""
    IO = 10
    """A file, device or network operation failed."""
    INTERRUPTED = 130
    """Interrupted by Ctrl-C (128 + SIGINT, the shell convention)."""


EXIT_CODE_HELP = (
    "Exit status: 0 ok · 1 failed / beacon not verified · 2 usage · 3 file exists · "
    "4 wrong key · 5 authentication failed (tampered or wrong password) · 6 truncated · "
    "7 bad format · 8 entropy target missed with --strict · 9 component unavailable · "
    "10 I/O error · 130 interrupted."
)


class CliError(Exception):
    """A failure the user can act on: a message, an exit code and an optional hint.

    Commands raise it; the root command group prints it (see
    :mod:`puffer.cli.app`) and exits with ``code``. ``details`` are extra
    lines shown under the message (e.g. the source spec grammar).
    """

    def __init__(
        self,
        message: str,
        code: ExitCode = ExitCode.FAILURE,
        *,
        hint: str | None = None,
        details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint
        self.details = details

    def __str__(self) -> str:
        return self.message


def _os_error_message(exc: OSError) -> str:
    target = exc.filename if exc.filename is not None else ""
    reason = exc.strerror or str(exc)
    return f"{target}: {reason}" if target else reason


def _loaded_class(module: str, name: str) -> type[BaseException] | None:
    """``module.name`` if ``module`` is already imported, else ``None``.

    The frame sources pull in OpenCV and SciPy (most of a second). An
    exception can only be an instance of a class whose module was imported,
    so classifying errors this way never makes ``puffer --help`` slower.
    """
    cls = getattr(sys.modules.get(module), name, None)
    return cls if isinstance(cls, type) and issubclass(cls, BaseException) else None


def translate(exc: BaseException) -> CliError | None:
    """Map a known exception to a :class:`CliError`; ``None`` for unexpected ones.

    The order matters: ``FormatError`` and ``SourceSpecError`` are also
    ``ValueError`` and ``FileExistsError`` is an ``OSError``, so the specific
    classes are tested first. Messages from the crypto layer are safe to
    show: they name only public values (key ids, chunk indices).
    """
    if isinstance(exc, CliError):
        return exc
    if isinstance(exc, FileExistsError):
        return CliError(
            f"refusing to overwrite {exc.filename or 'an existing file'}",
            ExitCode.EXISTS,
            hint="pass --force to replace it",
        )
    if isinstance(exc, WrongKeyError):
        return CliError(f"wrong key: {exc}", ExitCode.WRONG_KEY)
    if isinstance(exc, AuthenticationError):
        return CliError(
            "authentication failed: the file was modified or corrupted, or the key/password "
            "is wrong (libsodium cannot tell which). No output was written.",
            ExitCode.AUTHENTICATION,
            details=str(exc),
        )
    if isinstance(exc, TruncatedError):
        return CliError(
            f"the encrypted file is truncated ({exc}); no output was written",
            ExitCode.TRUNCATED,
        )
    if isinstance(exc, FormatError):
        return CliError(f"not a valid puffer-cipher file: {exc}", ExitCode.FORMAT)
    if isinstance(exc, BeaconFormatError):
        return CliError(f"malformed beacon data: {exc}", ExitCode.FAILURE)
    spec_error = _loaded_class("puffer.sources.spec", "SourceSpecError")
    if spec_error is not None and isinstance(exc, spec_error):
        spec_help = getattr(sys.modules["puffer.sources.spec"], "SPEC_HELP", None)
        return CliError(str(exc), ExitCode.USAGE, details=spec_help)
    source_error = _loaded_class("puffer.sources.base", "SourceError")
    if source_error is not None and isinstance(exc, source_error):
        return CliError(f"cannot open the frame source: {exc}", ExitCode.IO)
    if isinstance(exc, FileNotFoundError):
        return CliError(f"no such file: {_os_error_message(exc)}", ExitCode.IO)
    if isinstance(exc, PermissionError):
        return CliError(f"permission denied: {_os_error_message(exc)}", ExitCode.IO)
    if isinstance(exc, IsADirectoryError):
        return CliError(f"is a directory: {_os_error_message(exc)}", ExitCode.IO)
    if isinstance(exc, OSError):
        return CliError(f"I/O error: {_os_error_message(exc)}", ExitCode.IO)
    return None
