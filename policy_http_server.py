"""HTTP REST Policy Backend - evaluates G1-G4 (identical logic to inline)"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

POLICY = {"t_min": 15.0, "t_max": 30.0,
    "allowed_actions": ["set_temperature", "set_mode", "set_fan"], "q_max": 100}

class PolicyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        result = evaluate_all_gates(data)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    def log_message(self, *args): pass

def evaluate_all_gates(intent):
    # G1: Safety
    if intent.get("setpoint") is not None:
        if intent["setpoint"] < POLICY["t_min"] or intent["setpoint"] > POLICY["t_max"]:
            return {"decision": "REJECT", "gate": "G1", "reason": "setpoint out of bounds"}
    if intent.get("action") not in POLICY["allowed_actions"]:
        return {"decision": "REJECT", "gate": "G1", "reason": "invalid action"}
    # G2: Privacy (timestamp only - source NOT checked)
    if intent.get("timestamp") is None:
        return {"decision": "REJECT", "gate": "G2", "reason": "missing timestamp"}
    # G3: Resilience
    if intent.get("queue_depth", 0) >= POLICY["q_max"]:
        return {"decision": "THROTTLE", "gate": "G3", "reason": "queue saturated"}
    # G4: Auditability
    if intent.get("intent_id") is None:
        return {"decision": "REJECT", "gate": "G4", "reason": "missing intent_id"}
    if intent.get("device_id") is None:
        return {"decision": "REJECT", "gate": "G4", "reason": "missing device_id"}
    return {"decision": "PASS"}

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8081), PolicyHandler)
    print("Policy HTTP server on :8081 (G1-G4)")
    server.serve_forever()
