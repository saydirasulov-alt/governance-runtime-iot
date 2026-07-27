# Independent reproduction record

This file records an end-to-end reproduction of every reported number, performed on a machine other than the one used to produce the manuscript and under a dependency stack that is **not** the pinned one.

## Environment

| | Manuscript (pinned) | Independent reproduction |
|---|---|---|
| Host | reference workstation | PC-Lab |
| OS | Ubuntu 22.04 / Windows 10 | Windows 11 (AMD64) |
| Python | 3.10 (3.10.20 / 3.10.12) | **3.13.13** |
| NumPy | 2.2.6 | **2.4.6** |
| pandas | 2.3.3 | **3.0.3** |
| scikit-learn | 1.7.2 | **1.9.0** |
| Release / commit | v1.2.0 / `34a5fcd` | v1.2.0 / `34a5fcd10b815e6cebb6150c3a0342e6240b9bcc` (verified by `git rev-parse HEAD`) |
| Physical hardware operated | none | none |

Dataset fingerprint verified at load: `datatraining.txt`, 604,818 bytes, SHA-256 `034506256a005e0ecdec7395d93a21bbe81fff30077edd023306c1b5156c631f`.

## Result

`python verify_paper_numbers.py --all` → **127 PASS / 0 FAIL (127 checks)**, covering:

| Group | Checks |
|---|---|
| Model / training (Table 10) | 12 |
| E1 closed loop under shift (Table 14) | 16 |
| E4 in-distribution | 5 |
| Controller baseline, learned vs rule vs constant | 15 |
| E6 rejection semantics | 6 |
| E3 irreversible actuation | 5 |
| E7 rollback budget | 6 |
| Floor (perfect and constant controllers) | 4 |
| Real-AI residuals and AUC | 13 |
| Gap finder | 7 |
| E8 plant sweep — all 33 cells | 33 |
| E9 robustness — 18 perturbations | 5 |

Unit tests: 25 passed. Audit chain: 9761 entries, verified intact; tampering with a single record on disk causes verification to fail.

## Selected values, reproduced identically

| Quantity | Manuscript | Reproduction |
|---|---|---|
| Training iterations / parameters | 327 / 705 | 327 / 705 |
| Loss, first → last | 178.7 → 0.34 | 178.7284 → 0.3443 |
| MAE train / held-out / shifted (°C) | 0.40 / 1.69 / 3.32 | 0.4024 / 1.6887 / 3.3185 |
| Occupancy recovery | 96.4% | 96.40% (TP 1603, TN 6247, FP 167, FN 126) |
| E1 exposures (°C·min) | 1185.3 / 954.8 / 953.2 / 727.7 / 1.2 | identical |
| E1 rollbacks | 0 / 0 / 9 / 5 / 1 | identical |
| Median physical recovery | 61 min | 60.7 min (3640 s) |
| E3 latched exposure / availability | 4951.7 / 10.8% | identical, FAILED_SAFE at minute 1050 |
| E6 rejection semantics | 924.8 / 918.7 / 1.2 / 734.0 / 727.9 / 727.7 | identical |
| Controller baseline (shift) | 953.2 / 1427.7 / 0.0 | identical, 306 / 0 / 0 rejections |
| Plant sweep | 33 cells | all 33 identical |
| Robustness | C1, C2, C3 hold 18/18 | identical |

## Host-dependent measurements (not pass/fail checked)

| Quantity | Manuscript | Reproduction |
|---|---|---|
| MQTT decision path, median / P90 | 0.44 / 0.64 ms | 0.447 / 0.593 ms |
| OPA over HTTP, median | 0.94 ms | 0.666 ms |
| OPA decision agreement | 240/240 | 240/240 |
| Gate evaluation, shipped / corrected | 4.5 / 5.2 µs | 4.0 / 4.4 µs and 5.1 / 5.6 µs on two runs |
| Context-predicate cost | +0.5 to +0.7 µs | +0.40 µs (the manuscript range was widened to +0.4–0.7 µs accordingly) |

Live latencies move with the host and between runs on one host; the manuscript states this and no claim rests on a single value.

## Commands executed

```
git rev-parse HEAD
python full_training.py
python run_all.py                      # tests, E1–E7, figures, E8 sweep, E9 robustness
python experiments/rule_vs_ai_baseline.py
python reproduce_all.py                # Tier-1 tables, ablations, audit chain, OPA conformance
python verify_paper_numbers.py --all   # 127 checks
python run_governance_demo.py          # live MQTT stack
python experiments/opa_http_latency.py # live OPA HTTP
```

Total wall clock: approximately 65 minutes.

## Full transcripts

The complete machine-generated logs are included in this directory:

| File | Contents |
|---|---|
| `MASTER_REPRODUCTION_LOG.txt` | full session transcript, 996 lines, ending in `RESULT: 127 PASS / 0 FAIL (127 checks)` at line 935 |
| `RUN_ALL_LOG.txt` | host-stamped closed-loop, plant sweep and robustness log, 514 lines |
| `REPRODUCTION_LOG.txt` | Tier-1 tables, ablations, audit chain and OPA conformance |
| `OPA_HTTP_LOG.txt`, `opa_http_latency.csv` | live OPA decision latencies, 240 requests |
| `OPA_SERVER_LOG.txt` | the OPA v1.18.2 server's own log for that session |

The only occurrences of the string `FAIL` in the master log are the state name `FAILED_SAFE`, which is an expected outcome of experiment E3, and the final `0 FAIL` count.
