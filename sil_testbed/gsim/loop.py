"""
The governed control loop.

              recorded sensors ---> AI service ---> intent
                                                      |
                                          governance plane (declarative policy)
                                                      |
                                              admit / reject / throttle
                                                      |
                                        ActuatorHAL.apply_setpoint()
                                                      |
                                          +---------- PLANT ----------+
                                          |  3-hour thermal inertia   |
                                          +---------------------------+
                                                      |
                                        runtime safety monitor (5 s)
                                                      |
                                     rollback  ->  budget  ->  FAILED_SAFE

The actuation loop is closed through the physics. That is the loop the reviewer is
asking about, and it is the one that produces the paper's new numbers.

Three mechanisms, and the experiments exist to separate them:

  ADMISSION   refuse the intent. Costs nothing. Works only if the policy is right.
  ROLLBACK    the intent was admitted and the room is now out of band. Restore the
              last checkpoint. This ENDS the excursion. It does not undo it: the room
              is still hot and takes minutes to come back. Rollback is not a time
              machine, and the plant is what proves it.
  FAILED_SAFE the runtime can no longer guarantee safety. Two very different reasons:
                budget exhausted   rollback keeps firing, so the model is systematically
                                   wrong and rolling back forever is not a strategy.
                                   The runtime relinquishes to the fallback thermostat.
                                   Safety kept, availability lost.
                latched actuator   the actuation was irreversible. The runtime has lost
                                   authority entirely and CANNOT make the room safe.
                                   It stops and escalates. It makes no safety claim,
                                   because it has none to make.

The last distinction is the one that keeps this honest. A rollback mechanism that
reports success in the latched case would be lying, and the manuscript would be lying
with it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from . import gates as G
from .aimodel import SetpointModel
from .hal import ActuatorHAL, SensorHAL, SimActuatorHAL, SimSensorHAL
from .plant import FAILSAFE_SETPOINT, PlantParams, RoomPlant, SafetyOracle

CONTROL_PERIOD_S = 60        # the AI service issues one intent per minute (UCI cadence)

# The integration step is DERIVED FROM THE PLANT, not fixed.
#
# The main study uses a building with tau = 3 h, and a 5 s step is numerically
# indistinguishable from a 1 s one there (dt/tau < 5e-4). But the plant sweep in
# run_sweep.py drives tau down to under a minute, and a 5 s step on a 22 s plant means
# dt/tau = 0.23: forward Euler is then simply wrong, and it is wrong in the direction
# that matters, because the plant moves further in one step than the safe band is wide.
# Our step-size test only ever checked the DEFAULT plant, so it passed while the swept
# plants produced numbers that were an artifact of the integrator.
#
# The step must also resolve the monitor. A monitor that samples every 1 s cannot be
# simulated with a 5 s step -- we ran exactly that and got byte-identical results for a
# 1 s and a 5 s monitor, which should have been impossible and was the clue.
#
# So: dt is small enough for the plant AND small enough for the monitor, and capped so
# the slow-plant runs stay fast.
def plant_dt(tau_s: float, monitor_period_s: float) -> float:
    return max(0.05, min(5.0, tau_s / 50.0, monitor_period_s / 2.0))


MONITOR_PERIOD_S = 5.0       # the runtime safety monitor polls the sensor every 5 s
# Rolling back forever is not a safety strategy: if the runtime is rolling back every
# few minutes, the model is systematically wrong and the policy is not catching it, and
# a system that keeps letting the same bad command through and then undoing it is just
# oscillating. A budget converts that into an explicit, auditable decision: hand the
# zone to the fallback thermostat and call a human.
#
# The budget is a POLICY CHOICE, not a fact, so it is off by default in the safety
# comparisons -- otherwise it silently truncates the arms it fires on and makes a
# shut-down system look safe -- and it gets its own experiment, where the safety it buys
# is weighed against the availability it costs.
ROLLBACK_BUDGET = None       # None = unlimited
ROLLBACK_WINDOW_S = 3600.0
SETTLE_TOL_C = 0.5           # a setpoint must hold the room this closely to earn a checkpoint

# The monitor needs a deadband. Without one it fires a rollback whenever the room grazes
# the band edge by 0.01 degC, which is below the sensor noise floor (sigma = 0.05 degC):
# the runtime would be rolling back in response to noise, and the rollback count would
# measure chatter rather than governance. 0.25 degC is five sigma, so a trigger means the
# room really is out of band and not merely being measured badly.
#
# The deadband is DERIVED from the noise floor rather than fixed, because a constant is
# only a deadband for the one sigma it was chosen for. At the default sigma = 0.05 the
# derivation returns exactly 0.25, so every number in the paper is unchanged. But the
# robustness study sweeps sigma to 0.5 degC, and a fixed 0.25 there is HALF a standard
# deviation -- no deadband at all. The monitor would fire on noise, the rollback count
# would measure chatter, and the very experiment meant to test the monitor would have been
# measuring a bug in it. We have made that exact mistake once already.
ROLLBACK_TRIGGER_SIGMAS = 5.0
ROLLBACK_TRIGGER_C = 0.25      # what the derivation returns at the default sigma = 0.05


def rollback_trigger_for(sigma_t: float) -> float:
    """The monitor's deadband for a given sensor-noise floor. Five sigma, always."""
    return ROLLBACK_TRIGGER_SIGMAS * sigma_t


@dataclass
class RollbackEvent:
    t_violation_s: float
    t_detect_s: float
    t_recovered_s: float | None
    detect_latency_s: float          # violation -> the monitor noticed it
    physical_recovery_s: float | None  # restore command -> room back inside the band
    peak_excursion_c: float
    exposure_c_min: float            # degC-min accrued during THIS excursion
    offending_setpoint: float
    restored_setpoint: float


@dataclass
class LoopResult:
    arm: str
    policy: str
    minutes: int
    fallback: str = "safe_state"
    intents: int = 0
    admitted: int = 0
    rejected: int = 0
    unsafe_minutes: int = 0
    unsafe_exposure_c_min: float = 0.0
    strict_exposure_c_min: float = 0.0     # same, with NO transition grace window
    peak_excursion_c: float = 0.0
    rollbacks: int = 0
    failed_safe: bool = False
    failed_safe_reason: str | None = None
    failed_safe_minute: int | None = None
    rollback_events: list[RollbackEvent] = field(default_factory=list)
    gate_latency_ms: list[float] = field(default_factory=list)
    audit_ok: bool = True
    audit_entries: int = 0
    governed_minutes: int = 0      # minutes the AI service was actually in the loop
    trace_min: list[float] = field(default_factory=list)
    trace_T: list[float] = field(default_factory=list)
    trace_sp: list[float] = field(default_factory=list)
    trace_occ: list[int] = field(default_factory=list)


def run_closed_loop(
    model: SetpointModel,
    trace: list[tuple[float, float, float, float, int]],
    policy: dict,
    *,
    arm: str = "",
    fallback: str = "safe_state",
    enable_rollback: bool = True,
    rollback_budget: int | None = ROLLBACK_BUDGET,
    irreversible_above_c: float | None = None,
    closed_perception: bool = False,
    minutes: int | None = None,
    seed: int = 7,
    keep_trace: bool = False,
    plant_params: PlantParams | None = None,
    monitor_period_s: float = MONITOR_PERIOD_S,
    monitor_reads_sensor: bool = True,
) -> LoopResult:
    n = min(minutes or len(trace), len(trace))
    plant = RoomPlant(T=float(trace[0][0]),     # start the twin at the recorded temperature
                      params=plant_params or PlantParams())
    dt = plant_dt(plant.params.R_th * plant.params.C_th, monitor_period_s)
    steps_per_control = max(1, int(round(CONTROL_PERIOD_S / dt)))
    trigger_c = rollback_trigger_for(plant.params.sigma_T)   # 0.25 at the default sigma
    sensor: SensorHAL = SimSensorHAL(plant, trace, closed_perception=closed_perception, seed=seed)
    act: ActuatorHAL = SimActuatorHAL(plant)

    # Two temperature readings, and they must never be the same object.
    #
    # The ORACLE scores the run. It reads the TRUE plant state, because it is ground
    # truth and ground truth does not have measurement error.
    #
    # The MONITOR decides whether to roll back. It reads a NOISY SENSOR, because that is
    # what a monitor in a real building has. Until we wrote this, the monitor read
    # plant.T directly -- it was an oracle, not a monitor, and the deadband we justified
    # as "five sigma of sensor noise" was protecting against noise that never entered the
    # loop. Worse, when we first tried to test this we added noise to the ORACLE instead,
    # which corrupted the ground truth and produced the absurd result that sensor noise
    # makes rollback MORE effective.
    monitor_rng = random.Random(seed + 9001)

    oracle = SafetyOracle()
    state = G.GovernanceState()
    state.commit(G.Checkpoint(FAILSAFE_SETPOINT, policy["name"], 0, G.GENESIS))
    audit = G.AuditLog()

    r = LoopResult(arm=arm or policy["name"], policy=policy["name"], minutes=n,
                   fallback=fallback)

    def restore_target() -> float:
        """
        Where the runtime sends the actuator when it refuses, or undoes, an intent.

        This turns out to matter more than the accuracy of the gate itself, which is
        the least obvious thing the testbed taught us.

        hold        do nothing; the actuator keeps its current setpoint. This is what a
                    pure filter does, and it is not safe: the setpoint it keeps was
                    chosen for the OLD context.
        checkpoint  revert to the last setpoint that provably held the room in band.
                    Still context-blind: a 17 degC setback that was verified safe while
                    the room was EMPTY is not safe once someone walks in. Reverting to
                    it reverts to a different wrong answer.
        safe_state  command the context-independent safe setpoint. 21 degC lies inside
                    BOTH the occupied band [20,25] and the vacant band [15,30], so it is
                    safe without knowing which one applies -- which is precisely the
                    knowledge the runtime does not have. This is a degraded mode: the AI
                    service is out of the loop while it is active, and that cost is real
                    and is reported as availability.
        """
        if fallback == "hold":
            return _sp(act)
        if fallback == "checkpoint" and state.checkpoint is not None:
            return state.checkpoint.setpoint
        return FAILSAFE_SETPOINT

    seq = 0
    viol_start: float | None = None
    viol_peak = 0.0
    viol_exposure = 0.0
    pending: RollbackEvent | None = None
    rb_times: list[float] = []
    monitor_due = 0.0

    for m in range(n):
        occ = int(trace[m][4])
        rd = sensor.read()

        # ---- the AI service decides, with no idea whether it is right ----
        sp = model.predict(rd.temperature, rd.humidity, rd.co2, rd.humidity_ratio)
        seq += 1

        ctx = {
            "intent_id": f"i-{seq:06d}",
            "device_id": "hvac-zone-1",
            "action": "set_temperature",
            "setpoint": sp,
            "timestamp": rd.t_wall,
            "source": "ai-service-1",
            "queue_depth": 0,
            "co2": rd.co2,             # REAL recorded CO2. The CORRECTED policy reads this.
            "true_occupancy": occ,     # only the undeployable ORACLE policy reads this.
        }

        t0 = time.perf_counter()
        decision, gate_id, _ = G.evaluate(policy, ctx)
        r.gate_latency_ms.append((time.perf_counter() - t0) * 1e3)
        r.intents += 1

        # ---- actuate ----
        if state.state == G.GovernanceState.FAILED_SAFE:
            r.rejected += 1
        elif decision == "PASS":
            r.admitted += 1
            if irreversible_above_c is not None and sp > irreversible_above_c:
                act.apply_setpoint(sp)
                act.latch(sp)                       # no command can undo this
                audit.append({"seq": seq, "event": "IRREVERSIBLE_ACTUATION",
                              "setpoint": round(sp, 2)})
                audit.append({"seq": seq, "event": "FAILED_SAFE",
                              "reason": "actuation is irreversible; runtime has no authority"})
                state.enter_failed_safe()
                r.failed_safe = True
                r.failed_safe_reason = "irreversible actuation"
                r.failed_safe_minute = m
            else:
                act.apply_setpoint(sp)
                audit.append({"seq": seq, "event": "ADMIT", "setpoint": round(sp, 2),
                              "co2": round(rd.co2, 1)})
        else:
            # A rejection must be an ACTION, not merely the absence of one.
            #
            # Vetoing a bad intent does not make the room safe: the actuator keeps
            # tracking whatever setpoint it already held, and if the context has
            # changed, that old setpoint is now wrong too. A gate that only says "no"
            # is a veto, not a controller. Concretely: the model wrongly reports the
            # room vacant and asks for 17 degC. Even a perfect gate that rejects this
            # leaves the room tracking the previous setpoint -- and if that was also a
            # vacant setback, the room stays cold with people in it. We measured exactly
            # this: with reject-and-do-nothing, the ORACLE policy scored WORSE than an
            # imperfect one, which is a contradiction that only a physical plant can
            # surface.
            #
            # So rejection reverts the actuator to the last verified checkpoint, which
            # is what the governance state machine says it does. That single line is the
            # difference between a gate that filters and a gate that governs.
            r.rejected += 1
            tgt = restore_target()
            act.apply_setpoint(tgt)
            audit.append({"seq": seq, "event": decision, "gate": gate_id,
                          "setpoint": round(sp, 2), "co2": round(rd.co2, 1),
                          "reverted_to": round(tgt, 2)})

        # ---- the plant moves; the monitor watches ----
        for _ in range(steps_per_control):
            act.tick(occ, dt)
            now = plant.t
            oracle.update(occ, now)
            exc = oracle.excursion(plant.T, occ, now)          # TRUE state: scoring
            T_seen = (plant.read_temperature(monitor_rng) if monitor_reads_sensor
                      else plant.T)
            exc_seen = oracle.excursion(T_seen, occ, now)      # SENSOR: the monitor's view
            r.strict_exposure_c_min += (
                oracle.strict_excursion(plant.T, occ) * (dt / 60.0))

            if exc > 0.0:
                r.unsafe_exposure_c_min += exc * (dt / 60.0)
                r.peak_excursion_c = max(r.peak_excursion_c, exc)
                viol_exposure += exc * (dt / 60.0)
                viol_peak = max(viol_peak, exc)
                if viol_start is None:
                    viol_start = now
            else:
                if pending is not None:
                    pending.t_recovered_s = now
                    pending.physical_recovery_s = now - pending.t_detect_s
                    pending.peak_excursion_c = viol_peak
                    pending.exposure_c_min = viol_exposure
                    r.rollback_events.append(pending)
                    pending = None
                viol_start, viol_peak, viol_exposure = None, 0.0, 0.0

                # What makes a checkpoint trustworthy.
                #
                # The obvious rule -- "checkpoint whenever an intent is admitted and the
                # room is currently in band" -- is wrong, and wrong in a way that
                # silently destroys the mechanism. A room is in band right after a bad
                # setpoint is admitted simply because it has not heated up YET. Commit
                # there and the checkpoint captures the very setpoint that is about to
                # cause the excursion; rollback then faithfully restores the fault, and
                # the measured benefit of rollback is exactly zero. We hit this, and the
                # only reason we caught it is that the plant has inertia: in a testbed
                # without physics the two rules are indistinguishable.
                #
                # A setpoint earns a checkpoint only by PROVING it holds the room safely:
                # the plant must be inside the safe envelope AND settled at the commanded
                # setpoint. Until the room has actually reached the setpoint, we do not
                # know what that setpoint does.
                settled = abs(plant.T - _sp(act)) <= SETTLE_TOL_C
                if (state.state == G.GovernanceState.RUNNING
                        and decision == "PASS" and settled):
                    state.commit(G.Checkpoint(_sp(act), policy["name"], seq, audit.head))

            # -------- runtime safety monitor, on its own 5 s cadence --------
            if now >= monitor_due:
                monitor_due = now + monitor_period_s

                if (exc_seen > trigger_c and enable_rollback and pending is None
                        and state.state == G.GovernanceState.RUNNING):

                    t_detect = now
                    restored = restore_target()
                    offending = _sp(act)
                    act.apply_setpoint(restored)

                    audit.append({"seq": seq, "event": "ROLLBACK",
                                  "from": round(offending, 2), "to": round(restored, 2)})
                    r.rollbacks += 1
                    rb_times.append(now)
                    pending = RollbackEvent(
                        t_violation_s=viol_start if viol_start is not None else now,
                        t_detect_s=t_detect,
                        t_recovered_s=None,
                        detect_latency_s=t_detect - (viol_start if viol_start is not None else now),
                        physical_recovery_s=None,
                        peak_excursion_c=viol_peak,
                        exposure_c_min=viol_exposure,
                        offending_setpoint=offending,
                        restored_setpoint=restored,
                    )

                    # Rolling back over and over is not a safety strategy: it means the
                    # model is systematically wrong and the policy is not catching it.
                    recent = [t for t in rb_times if now - t <= ROLLBACK_WINDOW_S]
                    if (rollback_budget is not None and len(recent) >= rollback_budget
                            and state.state == G.GovernanceState.RUNNING):
                        act.enter_failsafe()
                        audit.append({"seq": seq, "event": "FAILED_SAFE",
                                      "reason": f"rollback budget exhausted "
                                                f"({len(recent)} in {ROLLBACK_WINDOW_S/60:.0f} min)"})
                        state.enter_failed_safe()
                        r.failed_safe = True
                        r.failed_safe_reason = "rollback budget exhausted"
                        r.failed_safe_minute = m

        if oracle.is_unsafe(plant.T, occ, plant.t):
            r.unsafe_minutes += 1
        if state.state != G.GovernanceState.FAILED_SAFE:
            r.governed_minutes += 1

        if keep_trace:
            r.trace_min.append(m)
            r.trace_T.append(plant.T)
            r.trace_sp.append(_sp(act))
            r.trace_occ.append(occ)

        if isinstance(sensor, SimSensorHAL):
            sensor.advance()

    if pending is not None:
        pending.peak_excursion_c = viol_peak
        pending.exposure_c_min = viol_exposure
        r.rollback_events.append(pending)      # rolled back, but never recovered

    r.audit_ok, _ = audit.verify()
    r.audit_entries = len(audit.entries)
    return r


def _sp(a: ActuatorHAL) -> float:
    return float(getattr(a, "setpoint", FAILSAFE_SETPOINT))
