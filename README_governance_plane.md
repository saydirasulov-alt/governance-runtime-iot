# Governance Plane for AI-Driven IoT — deployable reference service

A **runnable governance service**, not a bundle of experiment scripts. It sits between AI
inference services and actuators on a real MQTT bus, gates every control intent through
policy gates G1–G4 evaluated by a real policy engine, and writes a persistent,
tamper-evident hash-chained audit log.

Nothing in the measurement path is simulated: real MQTT broker, real network hop, real
policy decision point, real disk.

## Run it (self-contained, no Docker)

```bash
pip install -r requirements.txt
pytest -q                      # 9 tests, including the documented G2 gap
python experiments/e2e_demo.py # end-to-end on a real MQTT stack
```

Expected (measured on a real stack, not modelled). The latency line below is one host's
output and is reproduced verbatim from that run; decision latencies are host-dependent and
move by several times between machines. The paper reports 0.44 ms median on its reference
host and 0.447 ms on an independent one (Table 12 and `reproduction/REPRODUCTION_EVIDENCE.md`).
No claim in the paper rests on a single one of these values:

```
intents published        : 240
admitted / rejected      : 120 / 120
MEASURED decision latency: median 0.099 ms   P90 0.317 ms
audit records            : 240   chain intact: True
after tampering one record on disk, chain verifies: False

--- live policy-gap diagnosis ---
admitted under SHIPPED policy   : 40/40   <-- the G2 gap
admitted under CORRECTED policy : 0/40    <-- gap closed
```

## Run it as a deployed system (Docker: real Mosquitto + real OPA)

```bash
docker compose up
```
Brings up Eclipse Mosquitto, a real Open Policy Agent server evaluating
`policy/governance.rego`, the governance plane, and an actuator.

## Run it on a Raspberry Pi (real hardware)

```bash
pip install RPi.GPIO gpiozero
python -m devices.gpio_actuator --broker <broker-ip> --device room-1
```
The governance plane is unchanged. Relay on GPIO17, servo on GPIO18. The simulator and the
GPIO actuator honour the same MQTT contract, so the governed control path is identical.

## Architecture

```
AI service ──intents/<dev>──▶  Governance plane  ──actuators/<dev>/cmd──▶  Actuator
 (real model)                   │  G1 safety                                (sim or Pi GPIO)
                                │  G2 metadata
                                │  G3 queue health   PDP: inline | HTTP | real OPA
                                │  G4 auditability
                                ▼
                       results/audit.jsonl  (append-only SHA-256 hash chain)
```

## Layout
```
runtime/     governance_service.py  pdp.py  policy_loader.py  audit.py  run.py
policy/      governance.rego                 # real OPA policy
configs/     policy_gates.yaml               # declarative gates (no code edits)
             policy_gates_corrected.yaml     # the G2 fix
agents/      ai_service.py                   # publishes intents over MQTT
devices/     actuator_sim.py  gpio_actuator.py
experiments/ e2e_demo.py
tests/       test_policy.py  test_audit.py
```

## Policy is configuration, not code
Gates are declared in `configs/policy_gates.yaml` as typed predicates and loaded at
runtime. Changing bounds, allowed actions, mandatory fields, the throttle threshold, or the
G2 source predicate requires no code change. `configs/policy_gates_corrected.yaml` differs
from the shipped policy by exactly one line and closes the documented G2 gap.

## License
MIT.
