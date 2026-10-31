"""Wrappers for the reference command-line tools ``ent`` and ``dieharder``.

The in-process suites (:mod:`.sp800_22`, :mod:`.ent`) are what the dashboard
uses, but reviewers trust numbers from the tools they already know. These
wrappers run the real binaries on a file and return structured results, so a
report can show both side by side and the test-suite can cross-validate them.

Robustness rules, because both tools were written for interactive use:

* commands are argument lists, never shell strings;
* every call has a timeout: ``dieharder -a`` on a small file can run for
  hours, rewinding the input again and again, and a hung child must never
  hang the caller. On timeout the child is killed and
  :class:`ToolTimeoutError` carries whatever results were already complete;
* output is parsed strictly; dieharder reports some errors (e.g. an
  unreadable file) on stdout with exit status 0, so error lines are detected
  explicitly instead of trusting the exit code;
* dieharder's "file was rewound" notices are kept: a p-value computed on
  recycled input is not independent evidence and the report must say so.
"""

from __future__ import annotations

import enum
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from puffer.analysis.ent import EntStats, derive_ent_stats

ENT = "ent"
DIEHARDER = "dieharder"
DIEHARDER_FILE_GENERATOR = 201  # file_input_raw: raw binary file, read as 32-bit words


class ExternalToolError(RuntimeError):
    """An external tool failed or produced output that could not be parsed."""


class ToolNotFoundError(ExternalToolError):
    """The tool's binary is not on ``PATH``."""


class ToolTimeoutError(ExternalToolError):
    """The tool exceeded its time budget and was killed.

    ``partial`` holds the dieharder results that completed before the
    deadline (empty for ``ent``), so a long run is not wasted entirely.
    """

    def __init__(self, message: str, partial: Sequence[DieharderResult] = ()) -> None:
        super().__init__(message)
        self.partial: list[DieharderResult] = list(partial)


def tool_path(name: str) -> str | None:
    """Absolute path of ``name`` on ``PATH``, or ``None``."""
    return shutil.which(name)


def ent_available() -> bool:
    return tool_path(ENT) is not None


def dieharder_available() -> bool:
    return tool_path(DIEHARDER) is not None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_tool(name: str) -> str:
    path = tool_path(name)
    if path is None:
        raise ToolNotFoundError(f"{name!r} is not installed or not on PATH")
    return path


def _require_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"input file not found: {p}")
    return p


# ---------------------------------------------------------------------------
# ent
# ---------------------------------------------------------------------------


def _ent_number(text: str) -> float | None:
    value = float(text)  # accepts ent's "-nan"
    return None if value != value else value


def parse_ent_terse(text: str) -> EntStats:
    """Parse ``ent -t`` CSV output (header row ``0,...``, data row ``1,...``).

    ``ent -t`` prints the primary figures only; the remaining fields are
    derived with :func:`puffer.analysis.ent.derive_ent_stats`. ent's
    sentinels map to ``None``: Monte Carlo ``nan`` (under 6 bytes) and
    serial correlation ``-100000`` (all values equal).
    """
    rows = {line.split(",", 1)[0]: line.split(",") for line in text.splitlines() if line.strip()}
    header, data = rows.get("0"), rows.get("1")
    if header is None or data is None or len(header) != 7 or len(data) != 7:
        raise ExternalToolError(f"unexpected ent -t output: {text!r}")
    if header[1] not in ("File-bytes", "File-bits"):
        raise ExternalToolError(f"unexpected ent -t header: {header!r}")
    try:
        size = int(data[1])
        entropy, chi_square, mean = (float(v) for v in data[2:5])
        monte = _ent_number(data[5])
        scc = _ent_number(data[6])
    except ValueError as exc:
        raise ExternalToolError(f"unparsable ent -t data row: {data!r}") from exc
    if size == 0:
        raise ExternalToolError("ent was given an empty file")
    return derive_ent_stats(
        size=size,
        bit_mode=header[1] == "File-bits",
        entropy=entropy,
        chi_square=chi_square,
        mean=mean,
        monte_carlo_pi=monte,
        serial_correlation=None if scc is None or scc <= -99999.0 else scc,
    )


def run_ent(path: str | Path, *, bit_mode: bool = False, timeout_s: float = 120.0) -> EntStats:
    """Run ``ent -t`` (``-b`` in bit mode) on ``path`` and parse the result."""
    exe = _require_tool(ENT)
    target = _require_file(path)
    cmd = [exe, "-t", *(["-b"] if bit_mode else []), str(target)]
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(f"ent timed out after {timeout_s} s on {target}") from exc
    if proc.returncode != 0:
        raise ExternalToolError(f"ent exited with {proc.returncode}: {proc.stderr.strip()}")
    return parse_ent_terse(proc.stdout)


# ---------------------------------------------------------------------------
# dieharder
# ---------------------------------------------------------------------------


class DieharderAssessment(enum.StrEnum):
    """dieharder's verdict column (WEAK: p in the outer 0.5 %, FAILED: outer 1e-6)."""

    PASSED = "PASSED"
    WEAK = "WEAK"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class DieharderResult:
    """One row of dieharder's result table.

    ``rewound`` is true when dieharder had already wrapped around the input
    file by the time this row was produced (``rewinds`` is its cumulative
    count): the p-value then partly re-tests data that was used before.
    """

    test_name: str
    ntup: int
    tsamples: int
    psamples: int
    p_value: float
    assessment: DieharderAssessment
    rewound: bool = False
    rewinds: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["assessment"] = self.assessment.value
        return out


_ROW = re.compile(
    r"^\s*(?P<name>\w+)\|\s*(?P<ntup>\d+)\|\s*(?P<tsamples>\d+)\|\s*(?P<psamples>\d+)\|"
    r"\s*(?P<p>[-+.\deE]+|nan)\|\s*(?P<assessment>PASSED|WEAK|FAILED)\s*$"
)
_REWOUND = re.compile(r"^#\s*The file \S+ was rewound (?P<count>\d+) times?")
_ERROR = re.compile(r"\bError:")


def parse_dieharder_output(text: str) -> list[DieharderResult]:
    """Parse dieharder's table output (stdout and stderr interleaved).

    Rewind notices precede the rows of the test during which they occurred
    and the count is cumulative, so once a notice is seen every later row of
    the same run is marked ``rewound``. Lines reporting an error raise
    :class:`ExternalToolError`.
    """
    results: list[DieharderResult] = []
    rewinds = 0
    for line in text.splitlines():
        if (match := _REWOUND.match(line.strip())) is not None:
            rewinds = max(rewinds, int(match["count"]))
        elif (match := _ROW.match(line)) is not None:
            results.append(
                DieharderResult(
                    test_name=match["name"],
                    ntup=int(match["ntup"]),
                    tsamples=int(match["tsamples"]),
                    psamples=int(match["psamples"]),
                    p_value=float(match["p"]),
                    assessment=DieharderAssessment(match["assessment"]),
                    rewound=rewinds > 0,
                    rewinds=rewinds,
                )
            )
        elif _ERROR.search(line):
            raise ExternalToolError(f"dieharder reported an error: {line.strip()}")
    return results


def _run_dieharder_once(cmd: list[str], timeout_s: float) -> tuple[str, bool]:
    """Run one dieharder process; return (output, timed_out)."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # keep rewind notices in order with the rows
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run has already killed and reaped the child; stdout holds
        # what was written before the kill (dieharder flushes every row).
        partial = exc.stdout or b""
        return partial.decode("utf-8", "replace"), True
    output = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise ExternalToolError(f"dieharder exited with {proc.returncode}: {output[-500:]}")
    return output, False


def run_dieharder(
    path: str | Path,
    *,
    tests: Sequence[int] | None = None,
    timeout_s: float,
    tsamples: int | None = None,
    psamples: int | None = None,
) -> list[DieharderResult]:
    """Run dieharder on a raw binary file (``-g 201 -f path``).

    ``tests=None`` runs the standard battery (``-a``) in one process.
    Otherwise each test number runs in its own ``-d N`` process, so results
    of finished tests survive a later timeout. ``tsamples``/``psamples`` map
    to ``-t``/``-p`` (dieharder's defaults when ``None``).

    ``timeout_s`` is the budget for the whole call and is required: there is
    no sensible default for a battery whose run time ranges from
    milliseconds to hours. When it runs out the current child is killed and
    :class:`ToolTimeoutError` is raised with the results gathered so far.
    """
    _require(timeout_s > 0, "timeout_s must be positive")
    exe = _require_tool(DIEHARDER)
    target = _require_file(path)
    base = [exe, "-g", str(DIEHARDER_FILE_GENERATOR), "-f", str(target)]
    if tsamples is not None:
        _require(tsamples > 0, "tsamples must be positive")
        base += ["-t", str(tsamples)]
    if psamples is not None:
        _require(psamples > 0, "psamples must be positive")
        base += ["-p", str(psamples)]
    selections = [["-a"]] if tests is None else [["-d", str(int(t))] for t in tests]
    deadline = time.monotonic() + timeout_s
    results: list[DieharderResult] = []
    for selection in selections:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolTimeoutError(f"dieharder budget of {timeout_s} s exhausted", results)
        output, timed_out = _run_dieharder_once([*base, *selection], remaining)
        results.extend(parse_dieharder_output(output))
        if timed_out:
            raise ToolTimeoutError(
                f"dieharder timed out after {timeout_s} s ({' '.join(selection)})", results
            )
    return results
