# A software-in-the-loop (SIL) testbed for runtime governance in AI-driven IoT

A physics-based digital twin of a smart-building HVAC zone, closed around a real model
trained on real sensor data and a real declarative governance plane.

**Scope, stated once and plainly: no physical hardware was operated.** Every physical
quantity reported here comes from the digital twin in `gsim/plant.py`, whose model and
parameters are given in full so they can be challenged. The Raspberry Pi backends in
`gsim/hal.py` are complete and released, and they implement the same interface, so the
hardware evaluation is an interface swap rather than a rewrite. But they have not been
run, and nothing here should be read as a hardware result.

---

## Run it

```bash
pip install numpy pandas scikit-learn matplotlib pytest

python -m pytest tests/ -q     # 17 tests
python run_sil.py              # ~40 s, writes results/
python make_figures.py         # writes figures/
```

---

## Why a plant, and why it changes the answers

A software-only testbed cannot tell the difference between rejecting an intent and
admitting it and then rolling it back. Both end with the state restored, because software
state has no inertia. A room does. Once a bad setpoint has been admitted, the room starts
heating, and rolling back the setpoint does not roll back the heat.

Adding the plant made three of our own design choices visibly wrong. They are worth listing
because each is now a regression test, and because each was invisible before:

| What we had | Why it was wrong | Test |
|---|---|---|
| Checkpoint on admission, if the room is in band | A room is in band right after a bad setpoint is admitted only because it has not heated up *yet*. The checkpoint captured the setpoint that was about to cause the excursion, so rollback restored the fault and its measured benefit was exactly zero. | `test_checkpoint_must_be_proven_not_merely_admitted` |
| Reject = do nothing | A veto is not a controller. The actuator keeps tracking a setpoint chosen for the *old* context. Under this semantics the **oracle** policy scored *worse* than an imperfect one, which is a contradiction. | `test_rejection_must_be_an_action_not_an_absence` |
| Monitor with no deadband | It fired rollbacks on 0.01 °C excursions, below the 0.05 °C sensor-noise floor. The rollback budget then "fired" and entered FAILED_SAFE, which looked like a safety result and was a bug in our monitor. | `test_monitor_deadband_exceeds_the_sensor_noise_floor` |

---

## What the testbed found

**1. Rollback recovers almost nothing.** Under distribution shift, admission control cuts
unsafe exposure from 1185 to 955 °C·min. Adding rollback takes it to 954 — a 0.0%
improvement — and does not reduce the peak excursion at all (3.75 → 3.81 °C).

It cannot. The governance *decision* takes 0.44 ms (measured on the real MQTT stack). The
*physical recovery* takes a median of 60 minutes and up to 146. The exposure has already
been paid before the mechanism can act. Rollback ends an excursion; it does not undo one.

**2. Prevention works where recovery does not.** Same runtime, same rollback, same plant,
one extra predicate that happens to be right: the oracle policy lands at 1.8 °C·min. The
runtime mechanisms are not the limiting factor. The policy's estimate of context is.

**3. The residual risk belongs to the context estimator, not the runtime.** The deployable
CO₂ predicate sits between the two: 0.5 °C·min in-distribution (as good as the oracle),
727.5 under shift (as bad as no predicate at all), because under shift CO₂ stops tracking
occupancy. That gap is a direct measurement of how much residual *physical* risk is
attributable to the context estimator.

**4. Rejection semantics dominate rejection accuracy.** A perfect gate that reverts to a
context-blind state (924.8 °C·min) loses to an imperfect gate that reverts to a
context-independent safe one (727.5). The perfect gate with the safe fallback: 1.8. What
the runtime *does* when it says no matters more than how accurately it says it. This is
invisible in a software-only testbed, where restoring a variable is instantaneous and all
three fallbacks look identical.

**5. Against irreversible actuation, rollback is not available at all.** With a latching
actuator, the shipped policy loses authority and enters FAILED_SAFE with 4952 °C·min
accrued and 10.8% availability. The runtime does not claim to have recovered, because it
has not. The corrected policy never admits the intent, so the irreversible actuator is
never reached.

**6. The rollback budget never fires** (9 rollbacks in 162 h). We report this rather than
dropping it: a mechanism that never fires is not a contribution and we do not present it
as one.

---

## Layout

```
gsim/
  plant.py     RC thermal digital twin + the independent safety oracle
  hal.py       the sim/hardware boundary; Sim* and Pi* backends + parity check
  aimodel.py   MLP setpoint regressor, trained on UCI Occupancy (Light excluded)
  gates.py     declarative policy (shipped / corrected / oracle), checkpoints, audit chain
  loop.py      the governed control loop
run_sil.py       all experiments
make_figures.py  figures, generated from results/ only -- no hand-typed numbers
tests/           17 tests, three of which encode bugs we shipped and caught
ds/              UCI Occupancy Detection (Candanedo & Feldheim, 2016)
```

## The hardware boundary

`parity_report()` checks mechanically that `SimSensorHAL`/`SimActuatorHAL` and
`PiSensorHAL`/`PiActuatorHAL` implement the same interface, so "the same governance code
runs on hardware" is a checkable statement rather than a promise. `PiActuatorHAL` drives a
relay on GPIO17 and an SG90 servo on GPIO18 and returns the *measured* physical actuation
latency — the quantity the manuscript currently declares it does not measure. When that
backend is run, that sentence in the paper changes. Not before.
