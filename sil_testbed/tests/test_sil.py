"""
Tests for the software-in-the-loop (SIL) testbed.

These are not decorative. Three of them encode bugs we actually shipped and then caught
only because the plant has inertia, and they exist so we cannot ship them again:

    test_checkpoint_must_be_proven      a checkpoint taken on admission captures the very
                                        setpoint that is about to cause the excursion
    test_rejection_is_an_action         a gate that only vetoes leaves the actuator on a
                                        setpoint chosen for the old context
    test_monitor_has_a_deadband         a monitor with no deadband fires rollbacks on
                                        sensor noise and the rollback count becomes chatter

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsim import gates as G
from gsim.hal import (ActuatorHAL, PiActuatorHAL, PiSensorHAL, SensorHAL,
                      SimActuatorHAL, SimSensorHAL)
from gsim.loop import (ROLLBACK_TRIGGER_C, SETTLE_TOL_C, plant_dt,
                       rollback_trigger_for)
from gsim.plant import (PIController, PlantParams, RoomPlant, SafetyOracle,
                        TRANSITION_GRACE_S, SAFE_BAND_OCCUPIED, SAFE_BAND_VACANT,
                        FAILSAFE_SETPOINT)


# --------------------------------------------------------------------- plant

def test_plant_reaches_analytic_equilibrium():
    """The twin must actually satisfy its own steady-state equation."""
    p = RoomPlant(T=21.0)
    for _ in range(200_000):
        p.step(1.0, 1, 1.0)
    assert p.T == pytest.approx(p.equilibrium_T(1.0, 1), abs=0.05)


def test_plant_integration_step_is_not_load_bearing():
    """
    The experiments integrate at dt = 5 s for speed. If that changed the answer, the
    answer would be an artifact of the integrator. It does not.
    """
    a, b = RoomPlant(T=21.0), RoomPlant(T=21.0)
    for _ in range(3600):
        a.step(0.8, 1, 1.0)
    for _ in range(720):
        b.step(0.8, 1, 5.0)
    assert a.T == pytest.approx(b.T, abs=1e-3)


def test_thermal_inertia_is_what_makes_rollback_slow():
    """Cooling is bounded by the plant, not by the software. This is the whole paper."""
    p = RoomPlant()
    assert p.max_cooling_rate_c_per_min() < 0
    assert abs(p.max_cooling_rate_c_per_min()) < 1.0     # degC/min, i.e. minutes not ms


def test_pi_controller_tracks_and_does_not_wind_up():
    p = RoomPlant(T=21.0)
    c = PIController()
    for _ in range(40_000):
        p.step(c(24.0, p.T, 1.0), 0, 1.0)
    assert p.T == pytest.approx(24.0, abs=0.3)


# -------------------------------------------------------------------- oracle

def test_failsafe_setpoint_is_safe_in_every_context():
    """
    21 degC must lie inside BOTH bands. The entire safe_state fallback rests on this: it
    is what lets the runtime act safely WITHOUT knowing the context it does not have.
    """
    assert SAFE_BAND_OCCUPIED[0] <= FAILSAFE_SETPOINT <= SAFE_BAND_OCCUPIED[1]
    assert SAFE_BAND_VACANT[0] <= FAILSAFE_SETPOINT <= SAFE_BAND_VACANT[1]


def test_oracle_grace_forgives_the_ramp_but_not_a_hazard():
    o = SafetyOracle()
    o.update(0, 0.0)
    o.update(1, 100.0)                       # someone walks in at t = 100 s
    # 18 degC with people in the room: below the occupied band, but the room physically
    # cannot be anywhere else yet. Forgiven inside the grace window.
    assert o.excursion(18.0, 1, 200.0) == 0.0
    # 40 degC is a hazard in any context and must be flagged even inside the window.
    assert o.excursion(40.0, 1, 200.0) > 0
    # Once the window closes, the room is expected to have caught up.
    assert o.excursion(18.0, 1, 100.0 + TRANSITION_GRACE_S + 1) > 0


def test_oracle_is_context_dependent_which_is_the_point():
    o = SafetyOracle()
    o.update(0, 0.0)
    t = TRANSITION_GRACE_S * 3
    assert o.excursion(29.0, 0, t) == 0.0    # 29 degC in an empty room: fine
    o.update(1, t)
    t2 = t + TRANSITION_GRACE_S + 1
    assert o.excursion(29.0, 1, t2) > 0      # the same 29 degC with people in it: not fine


# ------------------------------------------------------------------- policy

def test_corrected_policy_differs_from_shipped_by_exactly_one_rule():
    a, b = G.shipped_policy(), G.corrected_policy(755.0)
    ra = [r for g in a["gates"] for r in g["rules"]]
    rb = [r for g in b["gates"] for r in g["rules"]]
    assert len(rb) == len(ra) + 1
    extra = [r for r in rb if r not in ra]
    assert len(extra) == 1 and extra[0]["type"] == "conditional_range"
    # Every other rule is untouched. The fix is one dictionary entry, not a code change:
    # that is what makes the shipped-vs-corrected comparison honest, because nothing else
    # can be quietly different between the two arms.
    assert [r for r in rb if r in ra] == ra


def test_shipped_policy_admits_the_unsafe_setpoint_and_corrected_does_not():
    ctx = {"intent_id": "i", "device_id": "d", "action": "set_temperature",
           "setpoint": 29.5, "timestamp": 1.0, "source": "s", "queue_depth": 0,
           "co2": 1100.0, "true_occupancy": 1}
    assert G.evaluate(G.shipped_policy(), ctx)[0] == "PASS"        # within [15,30]: admitted
    assert G.evaluate(G.corrected_policy(755.0), ctx)[0] == "REJECT"


def test_context_predicate_does_not_fire_when_the_room_is_empty():
    """The corrected gate must not become a blanket restriction."""
    ctx = {"intent_id": "i", "device_id": "d", "action": "set_temperature",
           "setpoint": 29.5, "timestamp": 1.0, "source": "s", "queue_depth": 0,
           "co2": 500.0, "true_occupancy": 0}
    assert G.evaluate(G.corrected_policy(755.0), ctx)[0] == "PASS"


def test_the_documented_g2_gap_is_real():
    ctx = {"intent_id": "i", "device_id": "d", "action": "set_temperature",
           "setpoint": 22.0, "timestamp": 1.0, "source": None, "queue_depth": 0,
           "co2": 500.0}
    assert G.evaluate(G.shipped_policy(), ctx)[0] == "PASS"       # source unchecked
    p = G.shipped_policy()
    for g in p["gates"]:
        for r in g["rules"]:
            if r.get("field") == "source":
                r["enabled"] = True
    assert G.evaluate(p, ctx)[0] == "REJECT"                       # one flag closes it


# ------------------------------------------------------------------- audit

def test_audit_chain_detects_tampering():
    a = G.AuditLog()
    for i in range(50):
        a.append({"seq": i, "event": "ADMIT"})
    assert a.verify() == (True, None)
    a.entries[20]["record"]["event"] = "REJECT"       # rewrite history
    ok, idx = a.verify()
    assert not ok and idx == 20


# ---------------------------------------------------------- regressions we caused

def test_checkpoint_must_be_proven_not_merely_admitted():
    """
    REGRESSION. We originally committed a checkpoint whenever an intent was admitted and
    the room happened to still be in band. But a room is in band right after a bad
    setpoint is admitted simply because it has not heated up yet. The checkpoint then
    captured the setpoint that was about to cause the excursion, rollback restored the
    fault, and the measured benefit of rollback was exactly zero.

    The rule is: a setpoint earns a checkpoint only by holding the room, i.e. the plant
    must have SETTLED at it.
    """
    p = RoomPlant(T=22.0)
    act = SimActuatorHAL(p)
    act.apply_setpoint(29.5)                       # admitted, room still at 22 and in band
    settled = abs(p.T - act.setpoint) <= SETTLE_TOL_C
    assert not settled, "a setpoint the room has not reached must not earn a checkpoint"
    for _ in range(6000):
        act.tick(1, 5.0)
    assert abs(p.T - act.setpoint) <= SETTLE_TOL_C  # now it has proven itself


def test_rejection_must_be_an_action_not_an_absence():
    """
    REGRESSION. A gate that rejects and does nothing else leaves the actuator tracking a
    setpoint chosen for the OLD context. With that semantics the ORACLE policy scored
    WORSE than an imperfect one, which is a contradiction. Rejection must move the
    actuator to a context-safe state.
    """
    p = RoomPlant(T=17.0)                          # room was empty, setback to 17
    act = SimActuatorHAL(p)
    act.apply_setpoint(17.0)
    for _ in range(4000):
        act.tick(0, 5.0)
    assert p.T == pytest.approx(17.0, abs=0.6)

    # Someone walks in. The model still says "vacant, 17 degC" and the gate rejects it.
    # Doing nothing keeps the room at 17 -- cold, occupied, unsafe.
    o = SafetyOracle()
    o.update(0, p.t)
    o.update(1, p.t)
    t_after = p.t + TRANSITION_GRACE_S + 1
    assert o.excursion(p.T, 1, t_after) > 0, "reject-and-hold leaves the room unsafe"

    # Reverting to the context-independent safe state fixes it.
    act.apply_setpoint(FAILSAFE_SETPOINT)
    for _ in range(4000):
        act.tick(1, 5.0)
    assert o.excursion(p.T, 1, p.t + TRANSITION_GRACE_S + 1) == 0.0


def test_monitor_deadband_exceeds_the_sensor_noise_floor():
    """
    REGRESSION. With no deadband the monitor fired rollbacks on 0.01 degC excursions --
    below the 0.05 degC sensor noise floor. It was rolling back in response to noise, the
    rollback count measured chatter, and the rollback budget then "fired", which looked
    like a safety result and was a bug in our monitor.
    """
    sigma = RoomPlant().params.sigma_T
    assert ROLLBACK_TRIGGER_C >= 5 * sigma


def test_the_deadband_tracks_the_noise_floor_it_claims_to_guard_against():
    """
    A CONSTANT deadband is only a deadband for the one sigma it was chosen for.

    The robustness study sweeps sensor noise to sigma = 0.5 degC. A fixed 0.25 degC trigger
    is five sigma at the default 0.05 and HALF a sigma at 0.5 -- so under the very
    perturbation designed to test the monitor, the monitor would have degraded into a noise
    detector, and the exposures it produced would have been chatter. The check meant to
    validate the result would instead have quietly invalidated it.

    This pins the derivation, and it pins the thing that must NOT move: at the default sigma
    it has to return exactly the constant the paper's numbers were produced with.
    """
    assert rollback_trigger_for(0.05) == pytest.approx(ROLLBACK_TRIGGER_C)   # paper unchanged
    for sigma in (0.05, 0.25, 0.5):
        assert rollback_trigger_for(sigma) >= 5 * sigma


# ---------------------------------------------------------------------- HAL

def test_sim_and_pi_backends_implement_the_same_interface():
    """
    The claim 'the same governance code runs on hardware' must be mechanically checkable,
    not asserted in prose.
    """
    for iface, backends in ((SensorHAL, (SimSensorHAL, PiSensorHAL)),
                            (ActuatorHAL, (SimActuatorHAL, PiActuatorHAL))):
        for b in backends:
            for m in iface.__abstractmethods__:
                impl = getattr(b, m, None)
                assert impl is not None, f"{b.__name__} is missing {m}"
                assert not getattr(impl, "__isabstractmethod__", False), \
                    f"{b.__name__}.{m} is still abstract"


def test_latched_actuator_cannot_be_commanded_back():
    """Irreversibility must be irreversible, or FAILED_SAFE means nothing."""
    p = RoomPlant(T=22.0)
    act = SimActuatorHAL(p)
    act.latch(29.9)
    act.apply_setpoint(21.0)
    assert act.setpoint == 29.9
    act.enter_failsafe()
    assert act.setpoint == 29.9, "a latched actuator must not be recoverable in software"


def test_parity_report_says_what_it_means():
    """
    REGRESSION, and the sharpest one.

    parity_report() is what shows that "the same governance code
    runs on hardware" is a checkable fact. It was checking `__isabstractmethod__` with a
    default of True, so every correctly implemented method came back as MISSING and the
    report declared all four backends broken -- while the classes were, in fact, complete.

    The existing interface test did not catch this, because it re-implemented the check
    correctly and inspected the CLASSES. Nobody was testing the REPORTER. So the claim and
    the thing that verifies the claim were free to drift apart, and they did.

    It surfaced only when the study was run on a second machine and a human read the
    output. Which is the argument for running it on a second machine.
    """
    from gsim.hal import parity_report
    rep = parity_report()
    assert "MISSING" not in rep, f"parity_report() reports a missing method:\n{rep}"
    assert rep.count("IMPLEMENTS ALL") == 4


def test_integration_step_follows_the_plant_and_the_monitor():
    """
    REGRESSION, and the one that would have poisoned the plant sweep.

    A 5 s Euler step is fine for the 3 h building (dt/tau < 5e-4) and simply wrong for a
    22 s plant (dt/tau = 0.23): the plant moves further in one step than the safe band is
    wide, so the excursion depth becomes an artifact of the integrator. The old test only
    checked the DEFAULT plant, so it passed while every swept plant was producing
    numerical noise.

    The step must also resolve the monitor. With a fixed 5 s step, a 1 s monitor and a 5 s
    monitor returned byte-identical results -- which is impossible, and was the clue.
    """
    for tau_s, mon in [(10800.0, 5.0), (10800.0, 1.0), (60.0, 5.0), (22.0, 1.0)]:
        dt = plant_dt(tau_s, mon)
        assert dt <= tau_s / 50.0 + 1e-9, "step too coarse for the plant"
        assert dt <= mon / 2.0 + 1e-9, "step cannot resolve the monitor"
        assert dt <= 5.0 and dt >= 0.05


def test_fast_plant_integration_is_actually_accurate():
    """The swept fast plant must integrate to the same answer at half the step."""
    pp = PlantParams(C_th=1.2e6 * 0.005)          # tau ~ 54 s
    a, b = RoomPlant(T=21.0, params=pp), RoomPlant(T=21.0, params=pp)
    dt = plant_dt(pp.R_th * pp.C_th, 1.0)
    n = int(600 / dt)
    for _ in range(n):
        a.step(1.0, 1, dt)
    for _ in range(2 * n):
        b.step(1.0, 1, dt / 2)
    assert a.T == pytest.approx(b.T, abs=0.05), "fast-plant integration is step-dependent"


def test_failsafe_setpoint_has_margin_not_just_membership():
    """
    REGRESSION. The old test only asserted the fail-safe setpoint was INSIDE both bands.
    Inside is not enough: 21.0 sits on the very edge of a [21,24] occupied band, so any
    undershoot by the local loop is an immediate safety violation. Under the band
    sensitivity sweep this made the oracle policy look like it was leaking exposure.

    A context-independent safe state must sit in the INTERIOR of the intersection of every
    context band, with real margin. The natural choice is the midpoint.
    """
    lo = max(SAFE_BAND_OCCUPIED[0], SAFE_BAND_VACANT[0])
    hi = min(SAFE_BAND_OCCUPIED[1], SAFE_BAND_VACANT[1])
    margin = min(FAILSAFE_SETPOINT - lo, hi - FAILSAFE_SETPOINT)
    assert margin >= 1.0, (
        f"fail-safe {FAILSAFE_SETPOINT} has only {margin:.1f} degC of margin inside "
        f"[{lo},{hi}]; the midpoint {(lo+hi)/2:.1f} would have {(hi-lo)/2:.1f}")


def test_grace_window_is_long_enough_for_the_bands_it_is_used_with():
    """
    REGRESSION. The grace window is 1800 s because the plant needs ~28 min to climb from
    the 17 degC vacant setback into a [20,25] occupied band. That derivation is band
    specific, and the constant is not. With a [21,24] band the climb takes 37 min and a
    30 min window is too short, so exposure that no controller could have avoided gets
    charged to governance anyway.

    Whatever bands are in force, the window must cover the ramp they require.
    """
    from gsim.aimodel import SETPOINT_VACANT
    p = RoomPlant()
    rate = p.max_heating_rate_c_per_min(T=SETPOINT_VACANT, n_occ=1)
    needed_s = 60.0 * (SAFE_BAND_OCCUPIED[0] - SETPOINT_VACANT) / rate
    assert TRANSITION_GRACE_S >= needed_s, (
        f"grace {TRANSITION_GRACE_S/60:.0f} min < the {needed_s/60:.0f} min the plant "
        f"needs to reach {SAFE_BAND_OCCUPIED[0]} degC from the {SETPOINT_VACANT} degC setback")


def test_the_oracle_policy_actually_uses_the_current_bands():
    """
    REGRESSION. gates.py did `from .plant import SAFE_BAND_OCCUPIED`, which copies the
    tuple at import time. Vary the bands in an experiment and the SCORER moves while the
    ORACLE POLICY silently keeps the old numbers -- so the "oracle" is graded against
    bands it was never given. Our band-sensitivity test hit this and reported that the
    oracle policy failed. It had not failed. It had never been an oracle for those bands.
    """
    import gsim.plant as P
    orig = P.SAFE_BAND_OCCUPIED
    try:
        P.SAFE_BAND_OCCUPIED = (21.0, 24.0)
        rule = [r for g in G.oracle_policy()["gates"] for r in g["rules"]
                if r["type"] == "conditional_range"][0]
        assert (rule["min"], rule["max"]) == (21.0, 24.0), \
            "the oracle policy is using stale bands captured at import time"
    finally:
        P.SAFE_BAND_OCCUPIED = orig


def test_the_monitor_reads_a_sensor_and_the_oracle_reads_the_truth():
    """
    REGRESSION. The runtime monitor read plant.T -- the TRUE state -- so it was an oracle,
    not a monitor, and the deadband we justified as "five sigma of sensor noise" was
    guarding against noise that never entered the loop. The scoring oracle must keep
    reading the truth; the monitor must not.
    """
    import inspect
    from gsim import loop as L
    src = inspect.getsource(L.run_closed_loop)
    assert "monitor_reads_sensor" in src
    assert "T_seen" in src and "exc_seen" in src
    assert "exc_seen > trigger_c" in src, \
        "the monitor must trigger on what it can SEE, not on ground truth"
