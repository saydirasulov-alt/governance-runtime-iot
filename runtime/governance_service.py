"""Governance plane as a REAL MQTT service.

Subscribes to control intents published by AI services, evaluates the policy gates
through a real Policy Decision Point (inline / HTTP / real OPA), and either publishes
the admitted command to the actuator topic or publishes a rejection. Every decision is
appended to a persistent, tamper-evident hash-chained audit log.

Nothing here is simulated: real MQTT broker, real network hop, real PDP, real disk.

Topics
  in : intents/<device_id>        (JSON control intent from an AI service)
  out: actuators/<device_id>/cmd  (admitted command)
       governance/rejected        (rejected / throttled intents)
       governance/metrics         (periodic counters)
"""
import json, time, os, threading
from runtime.mqtt_compat import make_client
from .pdp import make_pdp
from .audit import AuditLog

INTENT_TOPIC = "intents/#"
CMD_TOPIC = "actuators/{device}/cmd"
REJECT_TOPIC = "governance/rejected"


class GovernanceService:
    def __init__(self, broker="127.0.0.1", port=1883, pdp="inline",
                 policy="configs/policy_gates.yaml", audit_path="results/audit.jsonl",
                 queue_depth_fn=None, **pdp_kw):
        self.pdp = make_pdp(pdp, **(dict(policy_path=policy) if pdp == "inline" else pdp_kw))
        self.audit = AuditLog(audit_path)
        self.broker, self.port = broker, port
        self.stats = {"admitted": 0, "rejected": 0, "throttled": 0}
        self.latencies = []
        self.queue_depth_fn = queue_depth_fn or (lambda: 0)
        self.client = make_client("governance-plane")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, c, u, f, rc):
        c.subscribe(INTENT_TOPIC, qos=1)

    def _on_message(self, c, u, msg):
        t0 = time.perf_counter()
        try:
            intent = json.loads(msg.payload.decode())
        except Exception:
            return
        intent["queue_depth"] = self.queue_depth_fn()
        decision, gate = self.pdp.decide(intent)          # REAL policy call
        dev = intent.get("device_id", "unknown")

        if decision == "PASS":
            c.publish(CMD_TOPIC.format(device=dev),
                      json.dumps({"setpoint": intent.get("setpoint"),
                                  "action": intent.get("action"),
                                  "intent_id": intent.get("intent_id")}), qos=1)
            self.stats["admitted"] += 1
            rec = self.audit.append("ADMIT", {"intent_id": intent.get("intent_id"),
                                              "device_id": dev,
                                              "setpoint": intent.get("setpoint")})
        else:
            key = "throttled" if decision == "THROTTLE" else "rejected"
            self.stats[key] += 1
            c.publish(REJECT_TOPIC, json.dumps({"intent_id": intent.get("intent_id"),
                                                "decision": decision, "gate": gate}), qos=1)
            rec = self.audit.append(decision, {"intent_id": intent.get("intent_id"),
                                               "device_id": dev, "gate": gate})
        self.latencies.append((time.perf_counter() - t0) * 1000.0)   # REAL, measured

    def start(self):
        self.client.connect(self.broker, self.port, keepalive=30)
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop(); self.client.disconnect()

    def health(self):
        ok, n = self.audit.verify()
        return {"audit_intact": ok, "audit_records": n, **self.stats}
