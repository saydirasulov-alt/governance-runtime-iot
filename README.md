# Instrumented Governance Runtime for AI-Driven IoT

Reference implementation and reproducibility package for the paper:

> **Instrumented Governance Runtime for AI-Driven IoT: Policy Overhead, Rollback, and Adversarial Characterization**
> N. Saydirasulov Saydirasulovic, D. A. Davronbekov, M. M. Makhmudov, Y. I. Cho
> *Sensors* (MDPI), Special Issue: Advances in AI-Driven Technologies for Intelligent Sensors and IoT Systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

---

## Overview

This repository contains a software-instrumented governance-as-code runtime that enforces
policy-gated admission of sensor-originated control intents in AI-driven IoT systems. It
implements four deterministic governance gates (G1 Safety, G2 Privacy, G3 Resilience,
G4 Auditability), checkpoint-based rollback, a tamper-evident SHA256 audit chain, and an
M/M/1 queue-pressure model. The package reproduces every experiment reported in the paper.

## Repository structure

```
.
├── governance_runtime.py     # Core runtime: G1-G4 gates, ablation flags, queue model, 3 backends, audit chain
├── policy_http_server.py     # HTTP REST policy backend (G1-G4)
├── policy_cli_evaluator.py   # Subprocess CLI policy backend (G1-G4)
├── policy_gates.rego         # OPA Rego encoding of G1-G4 (demonstrates portability)
├── experiment_runner.py      # Runs all 6 experiments x 30 seeds
├── stats_analysis.py         # Wilcoxon, Cliff's delta, bootstrap CIs (Tables 5-7)
├── verify_corrected_g2.py    # Reproduces Table 10 (corrected-G2 ablation: 18.7% -> 0.0%)
├── verify_audit_chain.py     # Demonstrates the tamper-evident SHA256 audit chain (G4)
├── experiment_config.json    # Seeds and device counts
├── results/                  # Pre-generated traces (6 CSV files, ~116K rows)
└── requirements.txt
```

## Requirements

- Python 3.11+
- NumPy, SciPy

```bash
pip install -r requirements.txt
```

## Usage

All commands are run from the repository root.

```bash
# 1. Generate experiment traces (writes results/*.csv)
python3 experiment_runner.py

# 2. Reproduce the statistics in Tables 5-7
python3 stats_analysis.py

# 3. Reproduce Table 10 (corrected-G2 ablation)
python3 verify_corrected_g2.py

# 4. Demonstrate the tamper-evident audit chain (G4)
python3 verify_audit_chain.py
```

### Live backend execution (optional)

By default, the HTTP REST and subprocess CLI backends use a deterministic, seed-controlled
timing model for cross-machine reproducibility. To evaluate against live backends:

```bash
python3 policy_http_server.py          # start in a separate terminal
USE_REAL_HTTP=1 USE_REAL_SUBPROCESS=1 python3 experiment_runner.py
```

## Governance gates

| Gate | Function     | Checks                                                          |
|------|--------------|-----------------------------------------------------------------|
| G1   | Safety       | Setpoint bounds, allowed action                                 |
| G2   | Privacy      | Metadata integrity (timestamp; source in corrected variant)     |
| G3   | Resilience   | Queue depth / service health                                    |
| G4   | Auditability | Valid identifiers; SHA256 hash-chained audit log                |

## Reproducibility notes

- Confidence intervals use 10,000 bootstrap resamples with a fixed seed (42).
- Scalability is evaluated over 30 independent seeds.
- The default G2 predicate checks timestamp but not source, producing the ~18.7%
  partial-corruption bypass analyzed in the paper. Enabling `g2_check_source=True`
  (the corrected variant) closes it to 0.0% — see `verify_corrected_g2.py`.
- The audit log is a SHA256 hash chain; any modification of a past record is detectable —
  see `verify_audit_chain.py`.

## Scope

This is a software-instrumented characterization. The runtime constructs the checkpoint
sequence and audit chain and models the rollback control-path timing (MTTR); physical
actuator reversal is a design-level provision validated in future hardware work.

## Citation

If you use this code, please cite the paper (see [CITATION.cff](CITATION.cff)).

## License

Released under the MIT License. See [LICENSE](LICENSE).
