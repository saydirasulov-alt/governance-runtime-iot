# Instrumented Governance Runtime for AI-Driven IoT

Reference implementation and reproducibility package for the paper:

> **Instrumented Governance Runtime for AI-Driven IoT: Policy Overhead, Rollback, and Adversarial Characterization**
> N. Saydirasulov Saydirasulovic, D. A. Davronbekov, M. M. Makhmudov, Y. I. Cho.
> *Sensors* (MDPI), Special Issue: Advances in AI-Driven Technologies for Intelligent Sensors and IoT Systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

## Overview

A software-instrumented governance-as-code runtime that enforces policy-gated admission of
sensor-originated control intents in AI-driven IoT systems. It implements four deterministic
governance gates (G1 Safety, G2 Privacy, G3 Resilience, G4 Auditability), checkpoint-based
rollback, a tamper-evident SHA256 audit chain, and an M/M/1 queue-pressure model. The package
reproduces every experiment and table reported in the paper.

## Repository structure

```
.
├── governance_runtime.py          # Core runtime: G1-G4 gates (loaded from config), queue model, 3 backends, audit chain
├── policy_loader.py               # Declarative policy loader (YAML/JSON gate specifications)
├── policy_config.yaml / .json     # Default gate policy (operator-editable, no code change)
├── policy_config_corrected.*      # Corrected-G2 variant (source predicate enabled)
├── policy_http_server.py          # HTTP REST policy backend (G1-G4)
├── policy_cli_evaluator.py        # Subprocess CLI policy backend (G1-G4)
├── policy_gates.rego              # OPA Rego encoding of G1-G4
├── opa_conformance.py             # Generates fixtures so the Rego policy can be checked in the OPA engine
├── fixtures/                      # OPA conformance inputs + expected decisions
├── experiment_runner.py           # Runs all 6 experiments x 30 seeds
├── stats_analysis.py              # Wilcoxon, Cliff's delta, bootstrap CIs (Tables 5, 7, 8)
├── gate_complexity_benchmark.py   # Per-evaluation cost vs gate complexity (Table 6)
├── verify_corrected_g2.py         # Reproduces Table 11 (corrected-G2: 18.7% -> 0.0%)
├── ai_failure_characterization.py # AI-failure study vs independent physical oracle (Table 12, Figure 7)
├── rollback_demo.py               # Stateful rollback over a digital-twin actuator (Figure 3): verified restore + measured MTTR + FAILED-SAFE
├── hardware_rollback_harness.py   # Hardware-in-the-loop rollback over Raspberry Pi GPIO (see HARDWARE_README.md)
├── test_policy_loader.py          # Verifies config loader == legacy gates (400k intents, 0 mismatches)
├── verify_audit_chain.py          # Demonstrates the tamper-evident SHA256 audit chain (G4)
├── experiment_config.json         # Seeds and device counts
├── results/                       # Pre-generated traces (6 CSVs, ~116K rows) + AI-failure ROC + rollback MTTR
└── requirements.txt
```

## Requirements

- Python 3.11+
- NumPy, SciPy (analysis); PyYAML (optional, YAML configs; JSON twins work without it);
  Matplotlib (optional, regenerates Figures 3 and 7)

```bash
pip install -r requirements.txt
```

## Usage

All commands run from the repository root.

```bash
# Main experiments and statistics
python3 experiment_runner.py        # generates results/*.csv
python3 stats_analysis.py           # Tables 5, 7, 8

# Per-revision verification scripts (each reproduces a specific result)
python3 test_policy_loader.py            # config loader == legacy gates (400k intents, 0 mismatches)
python3 gate_complexity_benchmark.py     # Table 6: gate cost vs complexity
python3 verify_corrected_g2.py           # Table 11: corrected-G2, 18.7% -> 0.0%
python3 ai_failure_characterization.py   # Table 12 / Figure 7: AI-failure vs independent physical oracle
python3 rollback_demo.py                 # Figure 3: verified stateful rollback + measured MTTR + FAILED-SAFE
python3 verify_audit_chain.py            # tamper-evident SHA256 audit chain (G4)
python3 opa_conformance.py               # OPA conformance fixtures (validate policy_gates.rego in the OPA engine)
```

### Declarative, operator-editable policy

Gates G1-G4 are defined in `policy_config.yaml` (or `.json`) as an ordered list of typed
predicates and loaded at runtime. An operator changes the safe-range bounds, allowed actions,
mandatory metadata fields, the throttle threshold, or the G2 source predicate by editing that
file, with no code change. The bundled default is byte-for-byte equivalent to the original
gates (verified by `test_policy_loader.py`).

### Hardware-in-the-loop rollback (optional)

`hardware_rollback_harness.py` runs the rollback FSM against a real relay/servo over Raspberry
Pi GPIO and measures physical restore latency. It falls back to a software mock driver on a
laptop. See `HARDWARE_README.md` for wiring and run instructions.

### Live backend execution (optional)

```bash
python3 policy_http_server.py        # start in a separate terminal
USE_REAL_HTTP=1 USE_REAL_SUBPROCESS=1 python3 experiment_runner.py
```

## Governance gates

| Gate | Function     | Checks                                                      |
| ---- | ------------ | ----------------------------------------------------------- |
| G1   | Safety       | Setpoint bounds, allowed action                             |
| G2   | Privacy      | Metadata integrity (timestamp; source in corrected variant) |
| G3   | Resilience   | Queue depth / service health                                |
| G4   | Auditability | Valid identifiers; SHA256 hash-chained audit log            |

## Reproducibility notes

- Confidence intervals use 10,000 bootstrap resamples with a fixed seed (42).
- Experiments use 30 independent seeds.
- The default G2 predicate checks timestamp but not source, producing the ~18.7%
  partial-corruption bypass analyzed in the paper. Enabling the source predicate (the
  corrected variant) closes it to 0.0%; see `verify_corrected_g2.py` (Table 11).
- The AI-failure characterization (`ai_failure_characterization.py`, Table 12 / Figure 7)
  evaluates the gate against an independent, context-dependent physical-safety oracle, so the
  reported detection rates are non-circular.
- Rollback recovery latency is measured from a real digital-twin restoration
  (`rollback_demo.py`, Figure 3), not assigned as a constant. The tail (P99) is host-dependent.
- The audit log is a SHA256 hash chain; any modification of a past record is detectable; see
  `verify_audit_chain.py`.

## Scope

This is a software-instrumented characterization. The runtime constructs the checkpoint
sequence and audit chain and demonstrates rollback over a digital-twin actuator with measured
control-path timing; physical actuator-reversal latency is measured by the GPIO hardware
harness and is the aspect framed as future hardware work.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

Released under the MIT License. See `LICENSE`.
