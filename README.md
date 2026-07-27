# A Closed-Loop Measurement Study of Runtime Governance in AI-Driven Smart-Building Climate Control

[![Tests](https://github.com/saydirasulov-alt/governance-runtime-iot/actions/workflows/ci.yml/badge.svg)](https://github.com/saydirasulov-alt/governance-runtime-iot/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21374935.svg)](https://doi.org/10.5281/zenodo.21374935)

Reproducibility package and deployable reference service for:

> **A Closed-Loop Measurement Study of Runtime Governance in AI-Driven Smart-Building Climate Control**
> N. Saydirasulov Saydirasulovic, D. A. Davronbekov, M. M. Makhmudov, Y. I. Cho. *Sensors* (MDPI).

Runtime governance for AI-driven IoT is usually presented as a stack of mechanisms, namely admission
gates, checkpoint rollback, and tamper-evident audit, whose value is *assumed* rather than *measured*.
This package closes the control loop through a physical plant model and measures which mechanism
actually reduces physical risk. The headline result is negative and reproducible: **admission control
removes a substantial fraction of the unsafe physical exposure, while checkpoint rollback removes only a further 0.2%**,
because the governance decision costs 0.44 ms while the plant recovers in a median of 61 minutes.
Sweeping plant speed turns this into a transferable criterion: rollback protects safety only on plants
restored faster than the next command arrives and sampled faster than they leave the safe set.

This is a **software-in-the-loop** study. No physical sensor or actuator hardware was operated; the
Raspberry Pi backend is released and parity-checked so the hardware evaluation is an interface swap.

---

## The main result (one command, ~30 minutes)

```bash
git clone https://github.com/saydirasulov-alt/governance-runtime-iot.git
cd governance-runtime-iot          # every command below is run from this directory
pip install -r requirements.txt

cd sil_testbed
python run_all.py
```

Runs the 25 tests, the closed-loop safety experiments (E1–E7), the figures, the plant sweep, and the
robustness audit, and writes a host-stamped transcript to `results/RUN_ALL_LOG.txt`. The headline
figures (`1185.3 -> 954.8 -> 953.2 -> 727.7 -> 1.2` degC-min under distribution shift, `18/18`
robustness checks) drop straight out of it. Every plotted value is read from the results file, so no
figure can contradict its table.

The headline numbers in one line, as the paper reports them:

| Arm | Unsafe exposure under shift |
|---|---|
| Ungoverned | 1185.3 °C·min |
| Shipped policy, no rollback | 954.8 °C·min |
| Shipped policy + rollback | 953.2 °C·min |

Admission control removes 19.4% of the unsafe exposure; checkpoint rollback removes a further
**0.2%** in the reference deterministic run, and 0.1%–0.4% across the four sensor-noise seeds
(7, 11, 23, 101) of the robustness audit. These are deterministic paired outcomes on one trace, not
samples from a population, so no confidence interval is attached to them; bootstrap intervals appear
only where a seed distribution exists (the 30-seed backend and scalability experiments).

**The release evaluated in the paper is `v1.2.0`, Git commit `34a5fcd`.** Later releases change
documentation and metadata only; no executable line differs, and `verify_paper_numbers.py --all`
returns 127 PASS / 0 FAIL on every one of them. To just watch the AI model train, from `sil_testbed/` run
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
| Learned vs. deterministic rule-based control under identical governance (Sec. 8.5, Table 16) | `sil_testbed/experiments/rule_vs_ai_baseline.py` |
| Robustness audit, 18/18 across models/seeds/noise/bands | `sil_testbed/run_robustness.py` |
| Real learned model on real UCI data, training + generalization | `sil_testbed/full_training.py`, `real_ai_governance.py` |
| Tier 3 prototype: same gates on a real MQTT stack, in-process decision point (0.44 ms) | `run_governance_demo.py` |
| OPA HTTP decision-point latency (out-of-process path) | `experiments/opa_http_latency.py` + `docker-compose.yml` |
| Governance overhead across backends (Appendix) | `experiment_runner.py` -> `stats_analysis.py` |
| Gate-cost vs complexity (Appendix B) | `gate_complexity_benchmark.py` |
| Corrected-G2 ablation, 18.7% -> 0.0% (Appendix D) | `verify_corrected_g2.py` |
| Declarative gates == legacy gates | `test_policy_loader.py` |
| OPA Rego conformance | `opa_conformance.py` + `fixtures/` |
| Audit chain (tamper evidence) | `verify_audit_chain.py` |

## Measuring the OPA HTTP decision point

`experiments/opa_http_latency.py` measures governance-decision latency through a **real OPA
server over HTTP** (the out-of-process policy path), so that number is a measurement, not a model:

```bash
docker run --rm -p 8181:8181 openpolicyagent/opa:1.18.2 run --server   # or: docker compose up -d opa
python experiments/opa_http_latency.py                                  # 240 intents
```

It uploads `policy_gates.rego`, replays 240 intents (120 admit / 120 reject), checks every OPA
decision against the runtime's own gate logic, and reports a host-stamped median / P90 to
`results/opa_http_latency.csv`. The in-process decision path (the 0.44 / 0.09 ms figures) and this
OPA HTTP path are reported separately because they measure different architectures.

## Reproduction record

Every number reported in the paper is checked by one command:

```bash
cd governance-runtime-iot          # repository root
python verify_paper_numbers.py --all      # 127 checks
```

The machine-generated transcript of a full end-to-end run is committed under `reproduction/`:

| File | Contents |
|---|---|
| `reproduction/MASTER_REPRODUCTION_LOG.txt` | full session transcript, ending in `RESULT: 127 PASS / 0 FAIL (127 checks)` |
| `reproduction/REPRODUCTION_EVIDENCE.md` | environment table, per-group breakdown, host-dependent latencies |
| `reproduction/CLAIM_AUDIT.md` | claim-by-claim audit against the manuscript |
| `reproduction/OPA_SERVER_LOG.txt` | OPA v1.18.2 server log for the live HTTP measurement |
| `results/`, `sil_testbed/results/` | per-experiment logs, CSVs and figures |

The dataset is verified at load time: `ds/datatraining.txt`, 604,818 bytes, SHA-256
`034506256a005e0ecdec7395d93a21bbe81fff30077edd023306c1b5156c631f`.

The only occurrences of the string `FAIL` in the master log are the state name `FAILED_SAFE`,
which is an expected outcome of experiment E3, and the final `0 FAIL` count.

## Checkpoint semantics, and the scope of oracle independence

A successful commit creates a **provisional** record. It becomes an **eligible** recovery target
only after the actuator has acknowledged the command and the plant has stabilized inside the safe
envelope. Rollback and rejection both restore the latest *eligible* checkpoint.

The admission gates G1-G4 never receive the safety oracle, which is what makes the false-negative
measurements non-circular. In this software-in-the-loop testbed the *recovery* path is idealized:
the monitor's band selection and the checkpoint-eligibility predicate read the evaluation bands,
whose occupancy argument is the ground-truth label. We do not claim a direction or a magnitude for
that bias. What the experiment supports is the sharper statement that even with ground-truth context
available to the recovery predicates, rollback removed only 0.2% of unsafe exposure on this slow
plant. A deployable instantiation must use runtime-visible observables only. See the comments around
the checkpoint commit in `sil_testbed/gsim/loop.py`.

## Figure 1 source

`docs/fig1_architecture.tex` is the TikZ source of the architecture figure in the paper, with the
compiled `docs/fig1_architecture.pdf` alongside it.

## Layout

```
sil_testbed/            the closed-loop study: plant twin, safety oracle, governance
                       loop, and the runners run_all.py, run_sil.py, run_sweep.py,
                       run_robustness.py, full_training.py
run_governance_demo.py  self-contained Tier-3 prototype: starts a real embedded MQTT
                       broker and measures the governed control loop end-to-end
docker-compose.yml      optional deployed stack (Eclipse Mosquitto + Open Policy Agent)
policy_gates.rego       G1-G4 gates as OPA Rego; fixtures/ holds the conformance cases
policy_loader.py        declarative gate loader (equivalence-tested vs the legacy gates)
experiment_runner.py    governance-overhead backends  ->  stats_analysis.py (appendix)
gate_complexity_benchmark.py   gate cost vs policy complexity (Appendix B)
verify_corrected_g2.py  corrected-G2 ablation, 18.7% -> 0.0% (Appendix D)
verify_audit_chain.py   tamper-evident audit-chain verification
reproduce_all.py        one command for the backend/appendix tables + audit-chain checks
requirements.txt        runtime dependencies  (requirements-lock.txt pins exact versions)
.github/workflows/      continuous-integration workflow (runs the test suite)
tests/                  pytest suite (25 SIL tests) + test_policy_loader.py equivalence
results/                generated traces and host-stamped run logs
```

## Citing

See `CITATION.cff`. The archived release carries a Zenodo DOI; cite the version you ran.
