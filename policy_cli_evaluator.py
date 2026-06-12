"""Subprocess CLI Policy Evaluator - evaluates G1-G4 (identical logic to inline)"""
import sys, json
POLICY = {"t_min": 15.0, "t_max": 30.0,
    "allowed_actions": ["set_temperature", "set_mode", "set_fan"], "q_max": 100}

intent = json.loads(sys.stdin.read())
# G1: Safety
if intent.get("setpoint") is not None:
    if intent["setpoint"] < POLICY["t_min"] or intent["setpoint"] > POLICY["t_max"]:
        print(json.dumps({"decision": "REJECT", "gate": "G1"})); sys.exit()
if intent.get("action") not in POLICY["allowed_actions"]:
    print(json.dumps({"decision": "REJECT", "gate": "G1"})); sys.exit()
# G2: Privacy (timestamp only)
if intent.get("timestamp") is None:
    print(json.dumps({"decision": "REJECT", "gate": "G2"})); sys.exit()
# G3: Resilience
if intent.get("queue_depth", 0) >= POLICY["q_max"]:
    print(json.dumps({"decision": "THROTTLE", "gate": "G3"})); sys.exit()
# G4: Auditability
if intent.get("intent_id") is None or intent.get("device_id") is None:
    print(json.dumps({"decision": "REJECT", "gate": "G4"})); sys.exit()
print(json.dumps({"decision": "PASS"}))
