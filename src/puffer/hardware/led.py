"""LED drivers for the liveness challenge (DESIGN.md section 5).

The anti-replay check needs an actuator that the *server* controls and the
camera can see: an LED blinking a fresh, unpredictable pattern. A recording
replayed into the pipeline shows an old pattern, so it cannot match the
challenge that is being commanded right now.

Three drivers implement :class:`puffer.types.LedDriver`:

* :class:`GpioLed` - a real LED on a Raspberry Pi GPIO pin (``gpiozero``,
  imported lazily so the package works on machines without it).
* :class:`SimulatedLed` - records every state change, and is itself a
  :class:`puffer.types.LedSchedule`, so the aquarium simulator can render
  exactly what was commanded (closing the loop schedule -> driver -> pixels).
* :class:`NullLed` - does nothing, for setups without an LED.

:class:`LedController` turns a :class:`~puffer.types.LedSchedule` (a pure
function of time) into driver commands from a background thread. Commands
are timestamped with the same monotonic clock the camera uses
(``time.monotonic``), which is what lets the verifier estimate the camera
latency as a simple time lag.
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
import warnings
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

from puffer.types import LedDriver, LedSchedule

__all__ = [
    "GpioLed",
    "LedController",
    "LedHardwareError",
    "NullLed",
    "SimulatedLed",
]

_log = logging.getLogger(__name__)


class LedHardwareError(RuntimeError):
    """The requested LED hardware (or its driver library) is not available."""


class NullLed:
    """An LED driver that ignores every command.

    Useful when no LED is installed: the rest of the pipeline keeps a single
    code path, and liveness is then reported as ``DISABLED`` by the engine.
    """

    def set(self, on: bool, t: float) -> None:
        """Ignore the command."""

    def close(self) -> None:
        """Nothing to release."""


class SimulatedLed:
    """A virtual LED that remembers what it was told, for the simulator.

    It records *state changes* as ``(t, on)`` pairs; repeated identical
    commands are coalesced because they carry no information and a
    controller ticking at 100 Hz would otherwise grow the history without
    bound. :meth:`state_at` answers "what was the LED doing at time ``t``",
    which makes this object a :class:`~puffer.types.LedSchedule` that the
    aquarium simulator can render (with its own camera latency).

    The history is capped at ``max_history`` transitions (oldest dropped), so
    a long-running demo uses constant memory. Queries older than the
    retained history return the oldest retained state. All methods are
    thread-safe: a controller thread writes while the render loop reads.
    """

    def __init__(self, *, initial: bool = False, max_history: int = 65536) -> None:
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self._initial = initial
        self._max_history = max_history
        self._times: list[float] = []
        self._states: list[bool] = []
        self._lock = threading.Lock()
        self._n_commands = 0
        self._last_t: float | None = None
        self._closed = False

    def set(self, on: bool, t: float) -> None:
        """Record a command; ``t`` must not go backwards (monotonic clock)."""
        with self._lock:
            if self._closed:
                raise RuntimeError("SimulatedLed is closed")
            if self._last_t is not None and t < self._last_t:
                raise ValueError(
                    f"LED command time went backwards: {t} < {self._last_t} "
                    "(commands must use one monotonic clock)"
                )
            self._n_commands += 1
            self._last_t = t
            current = self._states[-1] if self._states else self._initial
            if bool(on) == current:
                return
            self._times.append(t)
            self._states.append(bool(on))
            if len(self._times) > 2 * self._max_history:
                # Amortised trimming keeps appends O(1) on average.
                self._initial = self._states[-self._max_history - 1]
                del self._times[: -self._max_history]
                del self._states[: -self._max_history]

    def state_at(self, t: float) -> bool:
        """Commanded state at time ``t`` (the last change at or before ``t``)."""
        with self._lock:
            i = bisect.bisect_right(self._times, t)
            return self._states[i - 1] if i > 0 else self._initial

    @property
    def is_on(self) -> bool:
        """Most recently commanded state."""
        with self._lock:
            return self._states[-1] if self._states else self._initial

    @property
    def transitions(self) -> tuple[tuple[float, bool], ...]:
        """Retained state changes as ``(t, on)`` pairs, oldest first."""
        with self._lock:
            return tuple(zip(self._times, self._states, strict=True))

    @property
    def n_commands(self) -> int:
        """Total number of :meth:`set` calls, including coalesced ones."""
        with self._lock:
            return self._n_commands

    def close(self) -> None:
        """Reject further commands; the recorded history stays readable."""
        with self._lock:
            self._closed = True


class GpioLed:
    """An LED on a Raspberry Pi GPIO pin, driven through ``gpiozero``.

    ``gpiozero`` is imported lazily so that importing :mod:`puffer.hardware`
    never fails on a laptop. Off-Pi, ``gpiozero`` may be installed but has no
    pin factory; that case is turned into a :class:`LedHardwareError` with a
    hint instead of gpiozero's generic ``BadPinFactory``. Tests (and dry runs)
    can pass ``pin_factory=gpiozero.pins.mock.MockFactory()``.

    Wiring and resistor sizing are documented in ``docs/hardware.md``
    (default: GPIO17, active high).
    """

    def __init__(self, pin: int = 17, active_high: bool = True, *, pin_factory: Any = None) -> None:
        try:
            import gpiozero  # optional dependency, Pi only
        except ImportError as exc:
            raise LedHardwareError(
                "GpioLed needs gpiozero: pip install 'puffer-cipher[pi]' "
                "(or use led='simulated' / 'none' off a Raspberry Pi)"
            ) from exc
        self.pin = pin
        self.active_high = active_high
        try:
            with warnings.catch_warnings():
                # gpiozero warns once per backend it fails to load before it
                # gives up; the single error below is the useful message.
                warnings.simplefilter("ignore", gpiozero.PinFactoryFallback)
                self._led = gpiozero.LED(
                    pin, active_high=active_high, initial_value=False, pin_factory=pin_factory
                )
        except gpiozero.BadPinFactory as exc:
            raise LedHardwareError(
                "no GPIO pin factory available - GpioLed only works on a Raspberry Pi "
                "(install lgpio there); use led='simulated' for the aquarium simulator"
            ) from exc
        except gpiozero.GPIOZeroError as exc:
            raise LedHardwareError(f"cannot claim GPIO{pin} for the LED: {exc}") from exc
        self._closed = False

    def set(self, on: bool, t: float) -> None:
        """Drive the pin; ``t`` is accepted for protocol symmetry and unused."""
        if self._closed:
            raise RuntimeError("GpioLed is closed")
        if on:
            self._led.on()
        else:
            self._led.off()

    @property
    def is_on(self) -> bool:
        """Logical LED state as reported by gpiozero."""
        return bool(self._led.is_lit)

    def close(self) -> None:
        """Switch the LED off and release the pin (idempotent)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._led.off()
        finally:
            self._led.close()


class LedController:
    """Drive an :class:`~puffer.types.LedDriver` from a schedule in a thread.

    The schedule is a pure function of time, so the controller simply polls
    it every ``tick_s`` seconds and forwards *changes* to the driver, stamped
    with the controller clock (``time.monotonic`` by default - the same clock
    live cameras stamp frames with). Polling keeps the contract minimal
    (``state_at`` only); with ``tick_s = 10 ms`` the edge jitter is far below
    the 100 ms Manchester half-symbol, and the verifier absorbs any constant
    delay as part of the camera lag.

    A driver error stops the thread; it is logged and kept in :attr:`error`
    so the engine can surface it (liveness will fail on its own as well,
    because the camera stops seeing the pattern). :meth:`stop` switches the
    LED off. The driver's lifetime belongs to the caller: the controller
    never closes it.

    :meth:`tick` performs one update synchronously; it is what the thread
    calls, and it lets callers with their own loop (or deterministic tests)
    drive the LED without a thread.
    """

    def __init__(
        self,
        schedule: LedSchedule,
        driver: LedDriver,
        tick_s: float = 0.01,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not tick_s > 0:
            raise ValueError("tick_s must be > 0")
        self.schedule = schedule
        self.driver = driver
        self.tick_s = float(tick_s)
        self._clock = clock
        self._state: bool | None = None
        self._transitions = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.error: BaseException | None = None

    def tick(self, now: float | None = None) -> bool:
        """Apply the scheduled state for ``now`` (default: the clock); return it."""
        t = self._clock() if now is None else now
        state = bool(self.schedule.state_at(t))
        with self._lock:
            if state != self._state:
                self.driver.set(state, t)
                self._state = state
                self._transitions += 1
        return state

    @property
    def running(self) -> bool:
        """Whether the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def transitions(self) -> int:
        """Number of state changes sent to the driver so far."""
        with self._lock:
            return self._transitions

    def start(self) -> None:
        """Start the background thread (no-op if already running)."""
        if self.running:
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="puffer-led", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the thread and switch the LED off (idempotent)."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            self._thread = None
        if self.error is None:
            with self._lock:
                if self._state is not False:
                    self.driver.set(False, self._clock())
                    self._state = False

    def _run(self) -> None:
        next_tick = self._clock()
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                self.error = exc
                _log.exception("LED driver failed; liveness challenge stopped")
                return
            # Absolute deadlines avoid drift; if we fell behind (suspended
            # process, overloaded Pi), resynchronise instead of bursting.
            next_tick += self.tick_s
            delay = next_tick - self._clock()
            if delay < -self.tick_s:
                next_tick = self._clock()
                delay = 0.0
            self._stop.wait(max(delay, 0.0))

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
