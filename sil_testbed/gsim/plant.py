"""
Physical plant model for the software-in-the-loop (SIL) governance testbed.

A digital twin of a single-zone office: a lumped-parameter (RC) thermal model of the
room, driven by an HVAC actuator. It is not a real room. Every physical quantity this
module produces is a simulated quantity and is labelled as such in the paper.

The plant exists for one reason. In a pure software testbed, a rejected intent and an
admitted-then-rolled-back intent look almost the same: both end with the state
restored. That is only true because software state has no inertia. A room does. Once a
bad setpoint has been admitted, the room starts heating, and rolling back the setpoint
does not roll back the heat. The plant is what makes that difference measurable, and
that difference is the answer to "under what realistic conditions would rollback be
required, and what does it actually buy you?"

Model
-----
    C dT/dt = (T_out - T)/R + P_hvac * u + Q_int,     u in [-1, +1]

    C       zone thermal capacitance (air + furnishings)      1.2e6  J/K
    R       envelope thermal resistance                       0.009  K/W
    tau=RC  envelope time constant                            ~3.0   h
    P_hvac  HVAC capacity                                     3000   W
    Q_int   internal gains, P_base + n_occ * P_person         200 + 100*n  W
    T_out   outdoor temperature                               12     degC

These are order-of-magnitude values for a small office zone, stated in full so they can
be challenged rather than buried. Two consequences of them drive every result:

    heating at full power         ~0.11 degC/min
    cooling at full power         ~0.21 degC/min

so an excursion takes tens of minutes to build and minutes to clear. Those two numbers
are why rollback is slow, and slowness is the finding.

CO2
---
The CO2 mass balance below is used ONLY as a plausibility check on the twin (its
steady state is compared against the recorded occupied CO2 in the UCI data). The CO2
that the AI service and the governance gates actually see is the REAL RECORDED CO2
from the dataset, never a simulated value. Ventilation is not manipulated by the HVAC
setpoint in this scenario, so CO2 is an exogenous input, and modelling it would have
meant inventing a signal we already have measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FAILSAFE_SETPOINT = 21.0    # the hardwired fallback thermostat
C_TH_REF = 1.2e6            # the reference zone capacitance the PI gains are tuned for


@dataclass
class PlantParams:
    C_th: float = 1.2e6      # J/K
    R_th: float = 0.009      # K/W   -> tau ~ 3 h
    P_hvac: float = 3000.0   # W
    P_base: float = 200.0    # W
    P_person: float = 100.0  # W/occupant
    T_out: float = 12.0      # degC
    sigma_T: float = 0.05    # degC  sensor noise

    # CO2, used only for the twin plausibility check
    V: float = 50.0          # m^3
    G_co2: float = 5.2e-6    # m^3/s per occupant
    Q_vent: float = 8.0e-3   # m^3/s
    C_out: float = 400.0     # ppm


@dataclass
class RoomPlant:
    params: PlantParams = field(default_factory=PlantParams)
    T: float = 21.0
    t: float = 0.0

    def step(self, u: float, n_occ: int, dt: float = 1.0) -> None:
        p = self.params
        u = max(-1.0, min(1.0, u))
        q = p.P_hvac * u + p.P_base + n_occ * p.P_person
        self.T += ((p.T_out - self.T) / p.R_th + q) / p.C_th * dt
        self.t += dt

    def read_temperature(self, rng) -> float:
        return self.T + rng.gauss(0.0, self.params.sigma_T)

    # -- diagnostics used by the tests and by the twin plausibility check --

    @property
    def tau_thermal_h(self) -> float:
        return self.params.R_th * self.params.C_th / 3600.0

    def equilibrium_T(self, u: float, n_occ: int) -> float:
        p = self.params
        return p.T_out + p.R_th * (p.P_hvac * u + p.P_base + n_occ * p.P_person)

    def max_heating_rate_c_per_min(self, T: float = 22.0, n_occ: int = 1) -> float:
        p = self.params
        return ((p.T_out - T) / p.R_th + p.P_hvac + p.P_base + n_occ * p.P_person) / p.C_th * 60

    def max_cooling_rate_c_per_min(self, T: float = 25.0, n_occ: int = 1) -> float:
        p = self.params
        return ((p.T_out - T) / p.R_th - p.P_hvac + p.P_base + n_occ * p.P_person) / p.C_th * 60

    def steady_state_co2(self, n_occ: int) -> float:
        p = self.params
        return p.C_out + 1e6 * n_occ * p.G_co2 / p.Q_vent


class PIController:
    """The HVAC's own local loop. Governance sets the target; this tracks it."""

    def __init__(self, kp: float = 0.8, ki: float = 0.003):
        self.kp, self.ki = kp, ki
        self.integ = 0.0

    def reset(self) -> None:
        self.integ = 0.0

    def __call__(self, setpoint: float, measured: float, dt: float = 1.0) -> float:
        e = setpoint - measured
        raw = self.kp * e + self.ki * (self.integ + e * dt)
        u = max(-1.0, min(1.0, raw))
        if u == raw:                       # anti-windup
            self.integ += e * dt
        return u


# ---------------------------------------------------------------------------
# The independent physical-safety oracle.
#
# This is NOT the governance gate, and the gate never sees it. It reads the TRUE
# plant temperature and the TRUE occupancy, neither of which is available to the
# runtime. Keeping the oracle strictly outside the governed path is what makes the
# false-negative measurements non-circular: the gate is scored by something it
# cannot influence and cannot observe.
#
# The bands themselves encode the point of the whole paper. The safe set is
# CONTEXT-DEPENDENT: 29 degC is a perfectly reasonable temperature for an empty
# room and an unacceptable one for an occupied room. A gate that checks only the
# setpoint cannot tell those apart, no matter how the bounds are tuned.
# ---------------------------------------------------------------------------

SAFE_BAND_OCCUPIED = (20.0, 25.0)
SAFE_BAND_VACANT = (15.0, 30.0)

# When occupancy changes, the band the room must satisfy changes INSTANTLY, but the
# room cannot. A zone sitting at the 17 degC vacant setback must climb 3 degC to
# re-enter the occupied band, and at the plant's maximum heating rate of
# 0.109 degC/min that takes
#
#       (20.0 - 17.0) / 0.109  ~=  28 minutes
#
# of full-power heating. No controller of any kind -- governed, ungoverned, or
# clairvoyant -- can beat that. It is a property of the room, not of the software.
# Charging that unavoidable transient to the governance layer would add the same large
# offset to every arm and drown the effect we are trying to measure.
#
# So for a grace window after each occupancy transition the oracle applies the UNION of
# the outgoing and incoming bands. The union is [15, 30], which still catches every
# genuine hazard -- an overheat to 40 degC is flagged inside the grace window exactly as
# it is outside it -- while forgiving the ramp that physics makes compulsory.
#
# The window is derived from the plant rather than tuned: 30 min, the next round number
# above the 28 min the slew-rate calculation above requires. The exposure WITHOUT any
# grace window is computed and logged alongside every result, so this choice can be
# inspected rather than taken on trust.
TRANSITION_GRACE_S = 1800.0


def current_bands() -> tuple[tuple[float, float], tuple[float, float]]:
    """
    The bands, read at CALL time.

    gates.py must not do `from .plant import SAFE_BAND_OCCUPIED`: that copies the tuple at
    import time, so an experiment that varies the bands moves the SCORER while the ORACLE
    POLICY silently keeps the old ones. We shipped that bug and it made the oracle policy
    look like it was failing when it had simply never been given those bands.

    An accessor fixes it in both builds -- the package and the generated single file --
    because a function always reads whatever globals it currently has.
    """
    return SAFE_BAND_OCCUPIED, SAFE_BAND_VACANT


def safe_band(n_occ: int) -> tuple[float, float]:
    occ, vac = current_bands()
    return occ if n_occ > 0 else vac


def _excursion(T: float, band: tuple[float, float]) -> float:
    lo, hi = band
    if T > hi:
        return T - hi
    if T < lo:
        return lo - T
    return 0.0


class SafetyOracle:
    """
    The independent physical-safety oracle.

    It is NOT the governance gate, and the gate cannot see it. It reads the TRUE plant
    temperature and the TRUE occupancy, neither of which is available to the runtime.
    Scoring the gate with something it can neither observe nor influence is what makes
    the false-negative measurement non-circular.

    The bands encode the point of the whole paper: the safe set is CONTEXT-DEPENDENT.
    29 degC is a perfectly reasonable temperature for an empty room and an unacceptable
    one for an occupied room. A gate that sees only the setpoint cannot tell those
    apart, no matter how carefully its bounds are tuned.
    """

    def __init__(self, grace_s: float = TRANSITION_GRACE_S):
        self.grace_s = grace_s
        self.prev_occ: int | None = None
        self.t_change: float = -1e9

    def update(self, n_occ: int, t: float) -> None:
        if self.prev_occ is not None and n_occ != self.prev_occ:
            self.t_change = t
        self.prev_occ = n_occ

    def band(self, n_occ: int, t: float) -> tuple[float, float]:
        cur = safe_band(n_occ)
        if t - self.t_change < self.grace_s:
            other = safe_band(0 if n_occ > 0 else 1)
            return (min(cur[0], other[0]), max(cur[1], other[1]))     # union
        return cur

    def excursion(self, T: float, n_occ: int, t: float) -> float:
        """Degrees outside the safe band; 0 inside. Integrated to give exposure."""
        return _excursion(T, self.band(n_occ, t))

    def is_unsafe(self, T: float, n_occ: int, t: float) -> bool:
        return self.excursion(T, n_occ, t) > 0.0

    @staticmethod
    def strict_excursion(T: float, n_occ: int) -> float:
        """No grace window at all. Logged alongside, so the choice stays inspectable."""
        return _excursion(T, safe_band(n_occ))
