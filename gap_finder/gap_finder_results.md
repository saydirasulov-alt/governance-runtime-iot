# Automated policy-gap finder -- results (30 seeds, bootstrap 95% CIs)

## E1. Ranked mutation gap-discovery

| Policy configuration | Exploitable physical gap % | 95% CI |
|---|---|---|
| drop G3/queue_depth | 48.6 | [48.2, 49.0] |
| raise G3 threshold 200 | 48.6 | [48.2, 49.0] |
| drop G1/setpoint | 30.6 | [30.3, 30.9] |
| weaken G1 setpoint band [10,35] | 28.7 | [28.3, 29.0] |
| drop G1/action | 28.6 | [28.3, 28.9] |
| drop G2/timestamp | 22.5 | [22.2, 22.7] |
| drop G4/intent_id | 21.3 | [21.0, 21.6] |
| SHIPPED policy (no mutation) | 20.2 | [19.9, 20.5] |
| drop G2/source | 20.2 | [19.9, 20.5] |
| drop G4/device_id | 20.2 | [19.9, 20.5] |

## E2. Method comparison (recall on 6-fault battery)

| Method | Recall | 95% CI |
|---|---|---|
| manual injection (1 hand-picked) | 0.17 | [0.17, 0.17] |
| policy-consistency testing (no oracle) | 0.00 | [0.00, 0.00] |
| random + physical oracle | 1.00 | [1.00, 1.00] |
| OURS: coverage-guided + oracle | 1.00 | [1.00, 1.00] |

## E3. Cost

Full finder ran in 0.11 s (10 configurations).
