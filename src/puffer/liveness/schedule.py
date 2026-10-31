"""Server-controlled LED challenge: an endless, unpredictable on/off pattern.

The liveness check is a challenge-response protocol whose "response" is light.
The server draws fresh random bits *as time goes by*, an LED inside the tank
shows them, and the verifier checks that the camera saw exactly those bits a
short latency later. A recording made before a bit was drawn cannot contain
it, so a replayed video fails no matter how realistic it looks.

Design decisions
----------------
* **Fresh bits, lazily.** Bits come from ``random_bytes`` (``os.urandom`` by
  default) in small batches the first time a symbol in that batch is needed,
  and are remembered per symbol index so the LED controller and the verifier
  (different threads, different times) agree on every symbol. Nothing about
  the pattern is derivable from a seed or from the clock.
* **Manchester line code.** Each bit is a symbol of ``symbol_s`` seconds split
  into two *chips*: bit 1 = on then off, bit 0 = off then on. Every symbol has
  a mid-symbol transition, so the LED is on exactly half the time whatever the
  bits are (no DC drift for the camera's auto-exposure, no long runs that would
  make the detector's two brightness clusters go stale) and the bit is decoded
  *differentially* by comparing its two halves, which is immune to slow
  illumination changes. ``"nrz"`` (one chip per bit, state = bit) is kept for
  tests and experiments.
* **Bounded memory.** Batches older than a retention horizon are dropped.
  An expired symbol is never regenerated (that would silently contradict what
  the LED showed); asking for it raises :class:`ExpiredSymbolError`.
* **Times before ``t0`` are "off"** and carry no challenge bits.

Every lookup is guarded by one lock: batch generation and pruning are atomic,
so concurrent callers always observe one consistent history.

Security note: future bits exist in server memory before they are displayed.
They are never exposed through ``repr`` and must never be published ahead of
time; once displayed they are public by nature (anyone can film the LED).
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable
from typing import Final, Literal, NamedTuple, get_args

import numpy as np
import numpy.typing as npt

Encoding = Literal["manchester", "nrz"]
RandomBytes = Callable[[int], bytes]

_ENCODINGS: Final = frozenset(get_args(Encoding))


class ExpiredSymbolError(LookupError):
    """The requested symbol is older than the schedule's retention horizon."""


class Chips(NamedTuple):
    """Consecutive chips (half-symbols for Manchester, symbols for NRZ)."""

    chip: npt.NDArray[np.int64]  # chip index k; chip k covers [t0 + k*chip_s, t0 + (k+1)*chip_s)
    start: npt.NDArray[np.float64]  # start time of each chip
    state: npt.NDArray[np.bool_]  # commanded LED state during the chip


class ChallengeSchedule:
    """Endless random LED pattern implementing :class:`puffer.types.LedSchedule`.

    Parameters
    ----------
    symbol_s:
        Duration of one bit on the LED (seconds).
    encoding:
        ``"manchester"`` (default, two chips per bit) or ``"nrz"``.
    t0:
        Start of symbol 0 on the shared monotonic timeline. Earlier times are off.
    random_bytes:
        Source of fresh randomness, called with a byte count. ``os.urandom`` in
        production; tests pass a seeded generator to make patterns reproducible.
    batch_bits:
        Bits drawn per call to ``random_bytes`` (a multiple of 8).
    retention_s:
        How long (in symbol time) bits are remembered after the newest generated
        symbol. Must cover camera latency plus the verifier's window.
    """

    def __init__(
        self,
        symbol_s: float = 0.2,
        encoding: Encoding = "manchester",
        t0: float = 0.0,
        random_bytes: RandomBytes = os.urandom,
        *,
        batch_bits: int = 256,
        retention_s: float = 600.0,
    ) -> None:
        if not (math.isfinite(symbol_s) and symbol_s > 0):
            raise ValueError(f"symbol_s must be a positive finite number, got {symbol_s!r}")
        if encoding not in _ENCODINGS:
            raise ValueError(f"encoding must be one of {sorted(_ENCODINGS)}, got {encoding!r}")
        if not math.isfinite(t0):
            raise ValueError("t0 must be finite")
        if batch_bits <= 0 or batch_bits % 8:
            raise ValueError("batch_bits must be a positive multiple of 8")
        if not (math.isfinite(retention_s) and retention_s > 0):
            raise ValueError("retention_s must be a positive finite number")

        self._symbol_s = float(symbol_s)
        self._encoding: Encoding = encoding
        self._t0 = float(t0)
        self._chips_per_symbol = 2 if encoding == "manchester" else 1
        self._chip_s = self._symbol_s / self._chips_per_symbol
        self._random_bytes = random_bytes
        self._batch_bits = batch_bits
        self._retention_symbols = math.ceil(retention_s / self._symbol_s)
        # +1: the batch holding the oldest retained symbol may be only partly retained.
        self._keep_batches = max(2, math.ceil(self._retention_symbols / batch_bits) + 1)

        self._lock = threading.Lock()
        self._batches: dict[int, npt.NDArray[np.uint8]] = {}
        self._newest_batch: int | None = None
        self._horizon_batch: int | None = None  # batches below this are gone for good

    def __repr__(self) -> str:
        # Deliberately omits every bit: future challenge bits are secret.
        return (
            f"{type(self).__name__}(symbol_s={self._symbol_s!r}, "
            f"encoding={self._encoding!r}, t0={self._t0!r})"
        )

    # ------------------------------------------------------------------ timing

    @property
    def symbol_s(self) -> float:
        return self._symbol_s

    @property
    def encoding(self) -> Encoding:
        return self._encoding

    @property
    def t0(self) -> float:
        return self._t0

    @property
    def chip_s(self) -> float:
        """Duration of one chip: ``symbol_s / 2`` for Manchester, ``symbol_s`` for NRZ."""
        return self._chip_s

    @property
    def chips_per_symbol(self) -> int:
        return self._chips_per_symbol

    @property
    def retention_symbols(self) -> int:
        """Number of most recent symbols guaranteed to stay available."""
        return self._retention_symbols

    @property
    def held_symbols(self) -> int:
        """Symbols currently held in memory (at most retention plus two batches)."""
        with self._lock:
            return len(self._batches) * self._batch_bits

    def chip_index(self, t: float) -> int:
        """Index of the chip containing time ``t`` (negative before ``t0``)."""
        return math.floor((_finite(t) - self._t0) / self._chip_s)

    def symbol_index(self, t: float) -> int:
        """Index of the symbol (bit) containing time ``t`` (negative before ``t0``).

        Derived from :meth:`chip_index` so that symbol and chip boundaries can
        never disagree because of floating-point rounding.
        """
        return self.chip_index(t) // self._chips_per_symbol

    def chip_start(self, k: int) -> float:
        return self._t0 + k * self._chip_s

    def symbol_start(self, i: int) -> float:
        return self.chip_start(i * self._chips_per_symbol)

    # -------------------------------------------------------------------- bits

    def bit(self, i: int) -> int:
        """Challenge bit of symbol ``i`` (0 or 1)."""
        return int(self.bits(i, i + 1)[0])

    def bits(self, i0: int, i1: int) -> npt.NDArray[np.uint8]:
        """Challenge bits of symbols ``i0 .. i1 - 1`` as a new uint8 array of 0/1.

        Raises :class:`ExpiredSymbolError` for symbols beyond the retention
        horizon and ``ValueError`` for negative indices or a range longer than
        the schedule can remember.
        """
        if i0 < 0 or i1 < i0:
            raise ValueError(f"invalid symbol range [{i0}, {i1})")
        if i1 - i0 > self._retention_symbols:
            raise ValueError("symbol range exceeds the retention horizon")
        return self._bits_for(np.arange(i0, i1, dtype=np.int64))

    def state_at(self, t: float) -> bool:
        """Commanded LED state at monotonic time ``t``."""
        k = self.chip_index(t)
        if k < 0:
            return False
        symbol = k // self._chips_per_symbol
        with self._lock:
            bit = int(self._batch_locked(symbol // self._batch_bits)[symbol % self._batch_bits])
        return bool(bit ^ (k & 1)) if self._chips_per_symbol == 2 else bool(bit)

    def states_at(self, times: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Vectorised :meth:`state_at` (same arithmetic, bit-identical results)."""
        t = np.asarray(times, dtype=np.float64)
        if not np.all(np.isfinite(t)):
            raise ValueError("times must be finite")
        k = np.floor((t - self._t0) / self._chip_s).astype(np.int64)
        return self._chip_states(k)

    def chip_states(self, k0: int, k1: int) -> npt.NDArray[np.bool_]:
        """Commanded states of chips ``k0 .. k1 - 1`` (chips before ``t0`` are off)."""
        if k1 < k0:
            raise ValueError(f"invalid chip range [{k0}, {k1})")
        return self._chip_states(np.arange(k0, k1, dtype=np.int64))

    def chips_between(self, ta: float, tb: float) -> Chips:
        """All chips overlapping the half-open interval ``[ta, tb)``.

        Membership uses exactly the arithmetic of :meth:`state_at`: the last
        chip is the one containing the largest float below ``tb``.
        """
        k0 = self.chip_index(ta)
        k1 = self.chip_index(math.nextafter(_finite(tb), -math.inf)) + 1 if tb > ta else k0
        index = np.arange(k0, k1, dtype=np.int64)
        start = self._t0 + index.astype(np.float64) * self._chip_s
        return Chips(chip=index, start=start, state=self._chip_states(index))

    # --------------------------------------------------------------- internals

    def _chip_states(self, k: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        out = np.zeros(k.shape, dtype=np.bool_)
        live = k >= 0
        if not np.any(live):
            return out
        chips = k[live]
        bits = self._bits_for(chips // self._chips_per_symbol)
        if self._chips_per_symbol == 2:
            bits = bits ^ (chips & 1).astype(np.uint8)
        out[live] = bits.astype(np.bool_)
        return out

    def _bits_for(self, symbols: npt.NDArray[np.int64]) -> npt.NDArray[np.uint8]:
        """Bits of arbitrary (non-negative) symbol indices, generating batches on demand."""
        out = np.empty(symbols.shape, dtype=np.uint8)
        batch_of = symbols // self._batch_bits
        wanted = np.unique(batch_of)  # ascending, so pruning never hits a batch still to be read
        with self._lock:
            batches = [(int(b), self._batch_locked(int(b))) for b in wanted]
        for b, arr in batches:
            sel = batch_of == b
            out[sel] = arr[symbols[sel] - b * self._batch_bits]
        return out

    def _batch_locked(self, b: int) -> npt.NDArray[np.uint8]:
        """Return batch ``b``, drawing it on first use. Caller holds ``self._lock``."""
        cached = self._batches.get(b)
        if cached is not None:
            return cached
        if self._horizon_batch is not None and b < self._horizon_batch:
            raise ExpiredSymbolError(
                f"symbols {b * self._batch_bits}..{(b + 1) * self._batch_bits - 1} "
                "are older than the retention horizon"
            )
        n_bytes = self._batch_bits // 8
        raw = self._random_bytes(n_bytes)
        if len(raw) != n_bytes:
            raise ValueError(f"random_bytes({n_bytes}) returned {len(raw)} bytes")
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))  # MSB of byte 0 = first bit
        bits.flags.writeable = False
        self._batches[b] = bits

        if self._newest_batch is None or b > self._newest_batch:
            self._newest_batch = b
            horizon = b - self._keep_batches + 1
            if self._horizon_batch is None or horizon > self._horizon_batch:
                self._horizon_batch = horizon
                for old in [k for k in self._batches if k < horizon]:
                    del self._batches[old]
        return bits


def _finite(t: float) -> float:
    value = float(t)
    if not math.isfinite(value):
        raise ValueError(f"time must be finite, got {t!r}")
    return value
