# A Closed-Loop Study of Runtime Governance for AI-Driven IoT

[![Tests](https://github.com/saydirasulov-alt/governance-runtime-iot/actions/workflows/ci.yml/badge.svg)](https://github.com/saydirasulov-alt/governance-runtime-iot/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducibility package and deployable reference service for:

> **A Closed-Loop Study of Runtime Governance for AI-Driven IoT**
> N. Saydirasulov Saydirasulovic, D. A. Davronbekov, M. M. Makhmudov, Y. I. Cho. *Sensors* (MDPI).

Runtime governance for AI-driven IoT is usually presented as a stack of mechanisms, namely admission
gates, checkpoint rollback, and tamper-evident audit, whose value is *assumed* rather than *measured*.
This package closes the control loop through a physical plant model and measures which mechanism
actually reduces physical risk. The headline result is negative and reproducible: **admission control
removes most of the unsafe physical exposure, while checkpoint rollback removes only a further 0.2%**,
because the governance decision costs 0.44 ms while the plant recovers in a median of 61 minutes.
Sweeping plant speed turns this into a transferable criterion: rollback protects safety only on plants
restored faster than the next command arrives and sampled faster than they leave the safe set.

This is a **software-in-the-loop** study. No physical sensor or actuator hardware was operated; the
Raspberry Pi backend is released and parity-checked so the hardware evaluation is an interface swap.

---

## The main result (one command, ~30 minutes)

```bash
pip install -r requirements.txt
cd sil_testbed
python run_all.py
```

Runs the 25 tests, the closed-loop safety experiments (E1–E7), the figures, the plant sweep, and the
robustness audit, and writes a host-stamped transcript to `results/RUN_ALL_LOG.txt`. The headline
figures (`1185.3 -> 954.8 -> 953.2 -> 727.7 -> 1.2` degC-min under distribution shift, `18/18`
robustness checks) drop straight out of it. Every plotted value is read from the results file, so no
figure can contradict its table. To just watch the AI model train, from `sil_testbed/` run
`python full_training.py`.

## The deployable governance service (Tier 3 prototype)

**Self-contained (no Docker), on a real MQTT stack:**
```bash
python run_governance_demo.py
```
Starts a real embedded MQTT broker, a governance plane gating 240 real control intents, an actuator,
and a disk-backed hash-chained audit log; the governance decision latency reported in the paper
(median 0.44 ms) is measured on this path.

**As a deployed system (real Mosquitto + real Open Policy Agent):**
```bash
docker compose up
```

**On a Raspberry Pi (not run in the paper; released for the hardware rung):**
```bash
pip install RPi.GPIO gpiozero
python -m devices.gpio_actuator --broker <broker-ip> --device room-1
```
The governance plane is unchanged: the simulator and the GPIO actuator honour the same MQTT contract.
**No physical actuation latency, power, or hardware number is claimed or measured in the paper.**

**Tests:** from `sil_testbed/`, `pytest -q` (25 tests, encoding the failure modes found and fixed while
building the testbed).

---

## What maps to what in the paper

| Paper | Where |
|---|---|
| Closed-loop safety, rollback = 0.2% (the principal result) | `sil_testbed/run_sil.py` |
| The rollback operating-envelope criterion (95% / 4.8%) | `sil_testbed/run_sweep.py` |
| Robustness audit, 18/18 across models/seeds/noise/bands | `sil_testbed/run_robustness.py` |
| Real learned model on real UCI data, training + generalization | `sil_testbed/full_training.py`, `real_ai_governance.py` |
| Tier 3 prototype on a real MQTT + OPA stack (0.44 ms) | `run_governance_demo.py`, `docker-compose.yml` |
| Governance overhead across backends (Appendix) | `experiment_runner.py` -> `stats_analysis.py` |
| Gate-cost vs complexity (Appendix B) | `gate_complexity_benchmark.py` |
| Corrected-G2 ablation, 18.7% -> 0.0% (Appendix D) | `verify_corrected_g2.py` |
| Declarative gates == legacy gates | `test_policy_loader.py` |
| OPA Rego conformance | `opa_conformance.py` + `fixtures/` |
| Audit chain (tamper evidence) | `verify_audit_chain.py` |

## Layout

```
sil_testbed/  the closed-loop study: plant twin, safety oracle, governance loop,
              run_all.py,