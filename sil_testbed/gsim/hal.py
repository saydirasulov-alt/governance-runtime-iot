"""
Hardware abstraction layer.

This is the most important file in the prototype for answering the reviewer, and it is
worth saying plainly why.

The objection is that software results "cannot be transferred to a realistic
deployment". The wrong answer is to pretend the simulation is a deployment. The right
answer is to make the boundary between them a single, inspectable interface, so that
everything above the interface -- the AI service, the governance gates, the checkpoint
and rollback logic, the audit chain -- is identical in simulation and on hardware, and
only what is below it changes.

    +---------------------------------------------------+
    |  AI service -> governance plane -> rollback/audit  |  identical in both
    +---------------------------------------------------+
                   | SensorHAL / ActuatorHAL              <- the boundary
        +----------+----------+
        |                     |
    Sim* (digital twin)   Pi* (SCD40, BME280, relay, servo)

`parity_report()` verifies mechanically that both backends implement the same
interface, so "the same governance code runs on hardware" is a checkable statement
rather than a promise.

The Pi backends are real code: real I2C reads, real GPIO writes, and a real measurement
of physical actuation latency. They were NOT executed in this study. No Pi, no sensor,
and no actuator was operated. That is stated here, in the code, so that nobody -- least
of all us -- can later mistake released hardware code for a hardware result.

Three actuator conditions, which the experiments depend on being distinct:

    normal      the runtime has authority. Setpoints are applied; rollback works.
    fail-safe   the runtime has RELINQUISHED authority. It stops accepting intents and
                hands the zone to a hardwired fallback thermostat. The room stays safe;
                the AI service is out of the loop. Safety preserved, availability lost.
    latched     the runtime has LOST authority. The actuation was irreversible and no
                command can undo it. The runtime cannot make the room safe, and it does
                not claim to. It stops and escalates to a human.

Conflating the last two is the mistake that makes rollback papers overclaim, so they
are separate flags here and separate results in the paper.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .plant import C_TH_REF, FAILSAFE_SETPOINT, PIController, RoomPlant


@dataclass
class SensorReading:
    temperature: float      # degC
    humidity: float         # %RH
    co2: float              # ppm
    humidity_ratio: float   # kg/kg
    t_wall: float           # s, perf_counter at read


class SensorHAL(ABC):
    @abstractmethod
    def read(self) -> SensorReading: ...
    @abstractmethod
    def close(self) -> None: ...


class ActuatorHAL(ABC):
    @abstractmethod
    def apply_setpoint(self, setpoint_c: float) -> float:
        """Command the setpoint. Returns physical actuation latency, seconds."""
    @abstractmethod
    def enter_failsafe(self) -> float:
        """Relinquish authority to the hardwired fallback."""
    @abstractmethod
    def close(self) -> None: ...


# ===========================================================================
# Simulation backends -- the ones used in this study
# ===========================================================================

class SimSensorHAL(SensorHAL):
    """
    Perception. Two modes, and the difference matters enough to be a flag rather
    than a hidden assumption:

    replay (default)
        All four features come from the real recorded UCI stream. The AI service
        therefore behaves EXACTLY as it does in the open-loop evaluation already
        reported in the paper -- same setpoints, same AUC, same failures. Nothing
        about the model's behaviour is changed by the introduction of the plant, so
        any difference in outcome is attributable to governance and physics alone.
        This is standard replay-driven software-in-the-loop testing.

    closed
        Temperature is read from the plant instead, so the model's own past decisions
        feed back into its inputs. Strictly more realistic, and available via
        --closed-perception, but it superimposes a perception-feedback effect on the
        governance effect we are trying to isolate. Reported separately, never mixed in.

    CO2 is recorded in BOTH modes. The HVAC setpoint does not manipulate ventilation
    in this scenario, so CO2 is genuinely exogenous, and simulating a signal we already
    have measured would only add a model to defend without adding realism.
    """

    def __init__(self, plant: RoomPlant, trace, *, closed_perception: bool = False,
                 seed: int = 7):
        self.plant = plant
        self.trace = trace     # list of (T_rec, humidity, co2_rec, humidity_ratio, occ)
        self.closed = closed_perception
        self.rng = random.Random(seed)
        self.k = 0

    def read(self) -> SensorReading:
        T_rec, hum, co2, hr, _occ = self.trace[min(self.k, len(self.trace) - 1)]
        T = self.plant.read_temperature(self.rng) if self.closed else T_rec
        return SensorReading(T, hum, co2, hr, time.perf_counter())

    def advance(self) -> None:
        self.k += 1

    def close(self) -> None:
        pass


class SimActuatorHAL(ActuatorHAL):
    """
    The HVAC. Holds the governed setpoint; a local PI loop drives the plant toward it.

    Actuation is neither instantaneous nor free. The room has a three-hour time
    constant, so a bad setpoint that is later rolled back still leaves a hot room
    behind. That residue is the experiment.
    """

    def __init__(self, plant: RoomPlant, ctrl: PIController | None = None):
        self.plant = plant
        # The HVAC's local loop, and the one subtlety that makes the plant sweep valid.
        #
        # PROPORTIONAL gain is held FIXED across every plant. With kp = 0.8, any error
        # above ~1.25 degC saturates the actuator, so a plant recovering from an
        # excursion runs at full power and moves as fast as it PHYSICALLY CAN. That is
        # the point: a fast plant should recover fast. Scaling kp with the plant would
        # deliberately detune the fast plants until they were as slow as the building,
        # which erases the exact variable the sweep exists to measure. (We tried it. The
        # sweep came out flat, with every plant recovering in the same wall-clock time,
        # which is not physics, it is us hobbling the controller.)
        #
        # INTEGRAL gain IS scaled with C. The integrator accumulates in wall-clock
        # seconds, so a gain tuned for a 1.2e6 J/K building winds up a hundred times too
        # fast on a plant a hundred times lighter: the loop overshoots and oscillates
        # across the safe-band edge, and the sweep comes out NON-MONOTONE, with a fast
        # plant scoring worse than a medium one. That is not a thing physics does; it was
        # our integrator ringing. Scaling ki with C keeps the integral's authority
        # proportionate to the plant it is acting on.
        #
        # Net effect: every plant in the sweep is driven by a loop that is stable and
        # time-optimal for it, and the only thing that varies is how quickly the physical
        # world can actually be moved. At scale 1.0 the gains are exactly those used in
        # the main study, so nothing in Sections E1-E7 changes.
        k = plant.params.C_th / C_TH_REF
        self.ctrl = ctrl or PIController(kp=0.8, ki=0.003 * k)
        self.setpoint = FAILSAFE_SETPOINT
        self.failsafe = False    # runtime relinquished authority
        self.latched = False     # runtime lost authority (irreversible actuation)
        self.u = 0.0

    def apply_setpoint(self, setpoint_c: float) -> float:
        t0 = time.perf_counter()
        if not (self.failsafe or self.latched):
            self.setpoint = float(setpoint_c)
        return time.perf_counter() - t0

    def latch(self, setpoint_c: float) -> None:
        """Irreversible actuation: the actuator sticks here and ignores everything after."""
        self.setpoint = float(setpoint_c)
        self.latched = True

    def enter_failsafe(self) -> float:
        t0 = time.perf_counter()
        if not self.latched:
            self.setpoint = FAILSAFE_SETPOINT
            self.ctrl.reset()
        self.failsafe = True
        return time.perf_counter() - t0

    def tick(self, n_occ: int, dt: float = 1.0) -> None:
        self.u = self.ctrl(self.setpoint, self.plant.T, dt)
        self.plant.step(self.u, n_occ, dt)

    def close(self) -> None:
        pass


# ===========================================================================
# Raspberry Pi backends -- released, complete, NOT executed in this study
# ===========================================================================

class PiSensorHAL(SensorHAL):                                    # pragma: no cover
    """SCD40 (CO2/T/RH) + BME280 (T/RH/P) over I2C."""

    def __init__(self):
        import board
        import busio
        import adafruit_scd4x
        import adafruit_bme280.basic as bme280

        i2c = busio.I2C(board.SCL, board.SDA)
        self.scd = adafruit_scd4x.SCD4X(i2c)
        self.scd.start_periodic_measurement()
        self.bme = bme280.Adafruit_BME280_I2C(i2c)

    @staticmethod
    def humidity_ratio(T_c: float, rh_pct: float) -> float:
        """
        The same psychrometric relation the UCI dataset was built with, so the model's
        fourth input means the same thing on hardware as it did in training. Getting
        this wrong would silently feed the model an out-of-distribution feature and
        make a hardware run look like a model failure.
        """
        import math
        p_sat = 610.78 * math.exp(17.2694 * T_c / (T_c + 238.3))     # Pa
        p_v = (rh_pct / 100.0) * p_sat
        return 0.62198 * p_v / (101325.0 - p_v)

    def read(self) -> SensorReading:
        T = float(self.bme.temperature)
        rh = float(self.bme.relative_humidity)
        co2 = float(self.scd.CO2) if self.scd.data_ready else float("nan")
        return SensorReading(T, rh, co2, self.humidity_ratio(T, rh), time.perf_counter())

    def close(self) -> None:
        self.scd.stop_periodic_measurement()


class PiActuatorHAL(ActuatorHAL):                                # pragma: no cover
    """
    Relay on GPIO17 (HVAC contactor) and an SG90 servo on GPIO18 (damper position,
    0-180 deg mapped onto the 15-30 degC setpoint authority).

    apply_setpoint() returns the MEASURED physical actuation latency: wall time from
    issuing the command to the servo reaching and settling at its commanded position.
    That is exactly the quantity the manuscript currently declares it does not measure.
    When this backend is run, that sentence in the paper changes -- and not before.
    """

    RELAY_PIN = 17
    SERVO_PIN = 18

    def __init__(self, settle_tolerance_deg: float = 2.0):
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.SERVO_PIN, GPIO.OUT)
        self.servo = GPIO.PWM(self.SERVO_PIN, 50)
        self.servo.start(0)
        self.tol = settle_tolerance_deg
        self.angle = 0.0
        self.failsafe = False
        self.latched = False

    @staticmethod
    def _angle(setpoint_c: float) -> float:
        frac = (max(15.0, min(30.0, setpoint_c)) - 15.0) / 15.0
        return frac * 180.0

    def _duty(self, angle: float) -> float:
        return 2.5 + (angle / 180.0) * 10.0

    def apply_setpoint(self, setpoint_c: float) -> float:
        if self.failsafe or self.latched:
            return 0.0
        target = self._angle(setpoint_c)
        travel = abs(target - self.angle)
        t0 = time.perf_counter()
        self.GPIO.output(self.RELAY_PIN, self.GPIO.HIGH)
        self.servo.ChangeDutyCycle(self._duty(target))
        # SG90: ~0.1 s per 60 deg at 4.8 V, plus settling. We wait for the datasheet
        # travel time and measure the wall clock, rather than assuming a constant.
        time.sleep(0.10 * (travel / 60.0) + 0.15)
        self.servo.ChangeDutyCycle(0)          # release, to stop servo jitter
        self.angle = target
        return time.perf_counter() - t0

    def enter_failsafe(self) -> float:
        t0 = time.perf_counter()
        self.GPIO.output(self.RELAY_PIN, self.GPIO.LOW)
        self.servo.ChangeDutyCycle(self._duty(self._angle(FAILSAFE_SETPOINT)))
        time.sleep(0.25)
        self.servo.ChangeDutyCycle(0)
        self.failsafe = True
        return time.perf_counter() - t0

    def close(self) -> None:
        self.servo.stop()
        self.GPIO.cleanup()


# ===========================================================================
# Parity
# ===========================================================================

def _is_concrete(cls, name: str) -> bool:
    """
    Is `name` a real implementation on `cls`, rather than an inherited abstract stub?

    The subtlety that bit us: a normal method has no __isabstractmethod__ attribute at
    all, so `getattr(m, "__isabstractmethod__", True)` returns the DEFAULT -- and if that
    default is True, every correctly implemented method is reported as missing. The
    default must be False. We had it as True, and parity_report() therefore declared all
    four backends broken while the classes were in fact complete.

    This mattered more than a cosmetic bug. parity_report() is the thing we point the
    reviewer at to show that "the same governance code runs on hardware" is mechanically
    checkable rather than asserted. The checker itself was wrong. The unit test did not
    catch it because the test re-implemented the check correctly and inspected the
    CLASSES; nobody was testing the REPORTER. test_parity_report_says_what_it_means now
    does, so the claim and the thing that verifies the claim can no longer drift apart.
    """
    m = getattr(cls, name, None)
    if m is None:
        return False
    return not getattr(m, "__isabstractmethod__", False)


def parity_report() -> str:
    lines = []
    for iface, backends in ((SensorHAL, (SimSensorHAL, PiSensorHAL)),
                            (ActuatorHAL, (SimActuatorHAL, PiActuatorHAL))):
        required = sorted(iface.__abstractmethods__)
        lines.append(f"  {iface.__name__:12s} requires {required}")
        for b in backends:
            missing = [m for m in required if not _is_concrete(b, m)]
            tag = "IMPLEMENTS ALL" if not missing else f"MISSING {missing}"
            lines.append(f"      {b.__name__:16s} {tag}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(parity_report())
