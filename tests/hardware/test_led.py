"""LED drivers and the LED controller."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from puffer.hardware.led import GpioLed, LedController, LedHardwareError, NullLed, SimulatedLed
from puffer.types import LedDriver, LedSchedule


class _Square:
    """On for the first half of every ``period`` (a stand-in challenge schedule)."""

    def __init__(self, period: float) -> None:
        self.period = period

    def state_at(self, t: float) -> bool:
        return (t % self.period) < self.period / 2


class _Boom:
    def set(self, on: bool, t: float) -> None:
        raise OSError("GPIO went away")

    def close(self) -> None:
        pass


def test_drivers_implement_the_protocols() -> None:
    assert isinstance(NullLed(), LedDriver)
    assert isinstance(SimulatedLed(), LedDriver)
    assert isinstance(SimulatedLed(), LedSchedule)
    led = NullLed()
    led.set(True, 0.0)
    led.close()


def test_simulated_led_records_changes() -> None:
    led = SimulatedLed()
    assert led.state_at(-5.0) is False
    assert led.is_on is False
    led.set(False, 0.5)  # no change from the initial state
    led.set(True, 1.0)
    led.set(True, 1.5)  # repeated command: coalesced
    led.set(False, 2.0)
    assert led.transitions == ((1.0, True), (2.0, False))
    assert led.n_commands == 4
    assert [led.state_at(t) for t in (0.9, 1.0, 1.99, 2.0, 99.0)] == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_simulated_led_rejects_time_travel_and_use_after_close() -> None:
    led = SimulatedLed()
    led.set(True, 1.0)
    led.set(True, 3.0)
    with pytest.raises(ValueError, match="backwards"):
        led.set(False, 2.0)
    led.close()
    with pytest.raises(RuntimeError):
        led.set(False, 4.0)
    assert led.state_at(3.5) is True  # history stays readable
    with pytest.raises(ValueError, match="max_history"):
        SimulatedLed(max_history=0)


def test_simulated_led_history_is_bounded() -> None:
    led = SimulatedLed(max_history=10)
    for i in range(100):
        led.set(i % 2 == 0, float(i))
    assert len(led.transitions) <= 20
    assert [led.state_at(t + 0.5) for t in (97, 98, 99)] == [False, True, False]
    assert led.state_at(-1.0) == led.state_at(led.transitions[0][0] - 0.5)


def test_controller_tick_forwards_only_changes() -> None:
    led = SimulatedLed()
    controller = LedController(_Square(1.0), led)
    for t in (0.0, 0.1, 0.4, 0.5, 0.7, 1.0, 1.2):
        controller.tick(t)
    assert led.transitions == ((0.0, True), (0.5, False), (1.0, True))
    assert controller.transitions == 3
    controller.stop()  # never started: still switches the LED off
    assert led.is_on is False
    with pytest.raises(ValueError, match="tick_s"):
        LedController(_Square(1.0), led, tick_s=0.0)


def test_controller_thread_follows_the_schedule() -> None:
    led = SimulatedLed()
    period, tick = 0.08, 0.005
    with LedController(_Square(period), led, tick_s=tick) as controller:
        assert controller.running
        time.sleep(0.45)
    assert not controller.running
    changes = led.transitions
    assert len(changes) >= 8
    for t, on in changes[:-1]:
        # Each command is issued within a few ticks after the schedule edge.
        phase = t % period
        edge = 0.0 if on else period / 2
        assert 0.0 <= phase - edge < 4 * tick + 0.02
        assert _Square(period).state_at(t) is on
    assert led.is_on is False  # stop() leaves the LED dark


def test_controller_uses_the_injected_clock() -> None:
    now = [100.0]
    led = SimulatedLed()
    controller = LedController(_Square(2.0), led, clock=lambda: now[0])
    controller.tick()
    now[0] = 101.5
    controller.tick()
    assert led.transitions == ((100.0, True), (101.5, False))


def test_controller_survives_a_failing_driver() -> None:
    controller = LedController(_Square(0.05), _Boom(), tick_s=0.005)
    controller.start()
    deadline = time.monotonic() + 2.0
    while controller.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not controller.running
    assert isinstance(controller.error, OSError)
    controller.stop()  # does not raise


def _led_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "puffer-led")


def test_start_and_stop_are_idempotent() -> None:
    controller = LedController(_Square(0.1), SimulatedLed(), tick_s=0.01)
    before = _led_threads()
    controller.start()
    controller.start()
    assert _led_threads() == before + 1
    controller.stop()
    controller.stop()
    assert _led_threads() == before
    assert not controller.running


def test_gpio_led_with_mock_pins() -> None:
    from gpiozero.pins.mock import MockFactory

    factory = MockFactory()
    led = GpioLed(17, pin_factory=factory)
    pin = factory.pin(17)
    assert pin.state is False
    led.set(True, 0.0)
    assert pin.state is True
    assert led.is_on
    led.set(False, 0.1)
    assert pin.state is False
    led.set(True, 0.2)
    led.close()
    assert pin.state is False  # closing switches the LED off
    led.close()
    with pytest.raises(RuntimeError):
        led.set(True, 1.0)


def test_gpio_led_active_low() -> None:
    from gpiozero.pins.mock import MockFactory

    factory = MockFactory()
    led = GpioLed(27, active_high=False, pin_factory=factory)
    led.set(True, 0.0)
    assert factory.pin(27).state is False  # sinking current lights the LED
    led.close()


def test_gpio_led_drives_from_the_controller() -> None:
    from gpiozero.pins.mock import MockFactory

    factory = MockFactory()
    led = GpioLed(22, pin_factory=factory)
    controller = LedController(_Square(1.0), led)
    controller.tick(0.1)
    assert factory.pin(22).state is True
    controller.tick(0.6)
    assert factory.pin(22).state is False
    led.close()


def test_gpio_led_off_pi_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import gpiozero

    def no_pins(*args: object, **kwargs: object) -> None:
        raise gpiozero.BadPinFactory("Unable to load any default pin factory!")

    monkeypatch.setattr(gpiozero, "LED", no_pins)
    with pytest.raises(LedHardwareError, match="Raspberry Pi"):
        GpioLed(17)


def test_gpio_led_pin_in_use(monkeypatch: pytest.MonkeyPatch) -> None:
    import gpiozero

    def busy(*args: object, **kwargs: object) -> None:
        raise gpiozero.GPIOPinInUse("pin 17 is already in use")

    monkeypatch.setattr(gpiozero, "LED", busy)
    with pytest.raises(LedHardwareError, match="GPIO17"):
        GpioLed(17)


def test_gpio_led_without_gpiozero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gpiozero", None)
    with pytest.raises(LedHardwareError, match=r"puffer-cipher\[pi\]"):
        GpioLed()
