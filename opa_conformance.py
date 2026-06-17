"""OPA conformance harness: generates input fixtures and the runtime's PASS/REJECT
decision for each, so the supplied Rego policy (policy_gates.rego) can be checked
to reproduce the runtime decisions using the OPA engine, e.g.:

    opa eval -i fixtures/case_007.json -d policy_gates.rego "data.governance.allow"

and comparing the boolean against this harness's expected_allow. This substantiates
that the policy is expressible and runnable in an external engine, not only in Python.

Run from supplementary/: python3 opa_conformance.py
"""
import json, time, os, random
from governance_runtime import GovernanceRuntime

def expected(intent, qd, rt):
    d, _ = rt._evaluate_gates(intent, qd)
    return d == "PASS"   # OPA allow == all gates pass

if __name__ == "__main__":
    os.makedirs("fixtures", exist_ok=True)
    rt = GovernanceRuntime(g2_check_source=True)  # full policy matching the Rego
    random.seed(7)
    cases = []
    for i in range(20):
        intent = {
            "setpoint": round(random.uniform(10, 35), 1),
            "action": random.choice(["set_temperature","set_mode","set_fan","reboot"]),
            "timestamp": None if i % 6 == 0 else time.time(),
            "source": None if i % 5 == 0 else f"ai_service_{i%3}",
            "intent_id": f"i{i}", "device_id": f"d{i}",
            "queue_depth": random.choice([0, 10, 120]),
        }
        qd = intent["queue_depth"]
        exp = expected(intent, qd, rt)
        # OPA input shape mirrors the Rego (input.setpoint, input.action, ...)
        opa_input = {"setpoint": intent["setpoint"], "action": intent["action"],
                     "timestamp": intent["timestamp"], "source": intent["source"],
                     "intent_id": intent["intent_id"], "device_id": intent["device_id"],
                     "queue_depth": qd}
        with open(f"fixtures/case_{i:03d}.json","w") as f:
            json.dump(opa_input, f, indent=2)
        cases.append({"case": f"case_{i:03d}.json", "expected_allow": exp})
    with open("fixtures/expected_decisions.json","w") as f:
        json.dump(cases, f, indent=2)
    passed = sum(1 for c in cases if c["expected_allow"])
    print(f"Generated {len(cases)} OPA conformance fixtures in fixtures/")
    print(f"  expected allow=true : {passed}")
    print(f"  expected allow=false: {len(cases)-passed}")
    print("Verify with the OPA engine:")
    print('  for f in fixtures/case_*.json; do opa eval -i "$f" -d policy_gates.rego "data.governance.allow"; done')
    print("and compare each result against fixtures/expected_decisions.json.")
