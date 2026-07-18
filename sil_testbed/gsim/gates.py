"""
The governance plane: declarative policy, checkpoint/rollback, hash-chained audit.

The policy is DATA, not code. Every predicate below is a typed rule in a dictionary
that could equally be a YAML file, an OPA/Rego document, or a row in a database. The
evaluator does not know what "setpoint" or "CO2" mean; it knows how to apply a
`range` rule and an `above` rule. That is what makes the shipped-vs-corrected
comparison honest: the two policies differ by one dictionary entry, not by a code
change, so nothing else can be quietly different between them.

Three policies are defined:

  SHIPPED    what a plausible engineering team would actually write. Bounds the
             setpoint to the equipment's rated range and checks the metadata. It is
             correct, it is not lazy, and it is blind to context.

  CORRECTED  SHIPPED plus one context predicate: if CO2 says the room is occupied,
             the setpoint must lie in the tighter occupied comfort band. This is the
             fix a competent engineer writes *after* seeing the failure.

  ORACLE     what a policy would have to know to be perfect: the true occupancy.
             It cannot be deployed -- if you had the true occupancy you would not
             need the model -- but it bounds what any gate could possibly achieve,
             and so it tells us how much of the residual risk is the gate's fault
             and how much is irreducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

from .plant import current_bands

# Read the bands through the MODULE, never by copying the names in.
#
#     from .plant import SAFE_BAND_OCCUPIED      # <-- what we had, and it is a trap
#
# That binds the tuple at import time. Any experiment that varies the bands then changes
# the SCORER but not the ORACLE POLICY, which quietly keeps the old numbers -- so the
# "oracle" gate is scored against bands it was never given. Our band-sensitivity test did
# exactly this and reported that the oracle policy failed. It had not failed; it had never
# been an oracle for those bands at all.

# ---------------------------------------------------------------------------
# Declarative policies
# ---------------------------------------------------------------------------

EQUIPMENT_MIN, EQUIPMENT_MAX = 15.0, 30.0


def shipped_policy() -> dict:
    return {
        "name": "SHIPPED",
        "gates": [
            {"id": "G1", "name": "Safety", "rules": [
                {"type": "range", "field": "setpoint",
                 "min": EQUIPMENT_MIN, "max": EQUIPMENT_MAX, "on_fail": "REJECT"},
                {"type": "allowed", "field": "action",
                 "values": ["set_temperature"], "on_fail": "REJECT"},
            ]},
            {"id": "G2", "name": "Privacy", "rules": [
                {"type": "required", "field": "timestamp", "on_fail": "REJECT"},
                # The documented gap: source provenance is specified but not enforced.
                {"type": "required", "field": "source", "enabled": False, "on_fail": "REJECT"},
            ]},
            {"id": "G3", "name": "Resilience", "rules": [
                {"type": "threshold", "field": "queue_depth", "max": 100, "on_fail": "THROTTLE"},
            ]},
            {"id": "G4", "name": "Auditability", "rules": [
                {"type": "required", "field": "intent_id", "on_fail": "REJECT"},
                {"type": "required", "field": "device_id", "on_fail": "REJECT"},
            ]},
        ],
    }


def corrected_policy(co2_threshold_ppm: float) -> dict:
    """SHIPPED + one context predicate in G1. One dictionary entry. That is the fix."""
    p = shipped_policy()
    p["name"] = "CORRECTED"
    p["gates"][0]["rules"].append({
        "type": "conditional_range",
        "when": {"field": "co2", "above": co2_threshold_ppm},
        "field": "setpoint",
        "min": current_bands()[0][0],              # read at CALL time, not import time
        "max": current_bands()[0][1],
        "on_fail": "REJECT",
    })
    return p


def oracle_policy() -> dict:
    """Not deployable. Upper bound on what any admission gate could achieve."""
    p = shipped_policy()
    p["name"] = "ORACLE"
    p["gates"][0]["rules"].append({
        "type": "conditional_range",
        "when": {"field": "true_occupancy", "above": 0.5},
        "field": "setpoint",
        "min": current_bands()[0][0],              # read at CALL time, not import time
        "max": current_bands()[0][1],
        "on_fail": "REJECT",
    })
    return p


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _fails(rule: dict, ctx: dict) -> bool:
    if not rule.get("enabled", True):
        return False
    t = rule["type"]
    v = ctx.get(rule.get("field"))

    if t == "required":
        return v is None or (isinstance(v, str) and not v.strip())
    if t == "range":
        if v is None:
            return not rule.get("nullable", False)
        return not (rule["min"] <= float(v) <= rule["max"])
    if t == "allowed":
        return v not in rule["values"]
    if t == "threshold":
        return v is not None and float(v) > rule["max"]
    if t == "conditional_range":
        w = rule["when"]
        wv = ctx.get(w["field"])
        if wv is None or float(wv) <= float(w["above"]):
            return False                     # condition not met, rule does not apply
        if v is None:
            return not rule.get("nullable", False)
        return not (rule["min"] <= float(v) <= rule["max"])
    raise ValueError(f"unknown rule type {t!r}")


def evaluate(policy: dict, ctx: dict) -> tuple[str, str | None, str | None]:
    """Returns (decision, gate_id, rule_type). decision in PASS/REJECT/THROTTLE."""
    for gate in policy["gates"]:
        for rule in gate["rules"]:
            if _fails(rule, ctx):
                return rule.get("on_fail", "REJECT"), gate["id"], rule["type"]
    return "PASS", None, None


# ---------------------------------------------------------------------------
# Checkpoint / rollback / FAILED_SAFE
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    setpoint: float
    policy_name: str
    sequence: int
    audit_head: str


class GovernanceState:
    """
    States: RUNNING -> (ROLLING_BACK -> RUNNING)* -> FAILED_SAFE (terminal)

    FAILED_SAFE is entered when the runtime cannot restore a verified state: an
    irreversible actuation, or a rollback that does not bring the plant back inside
    the safe envelope within the recovery deadline. It is terminal by design. A
    runtime that silently keeps trying after it has lost the ability to guarantee
    safety is worse than one that stops and calls a human.
    """

    RUNNING = "RUNNING"
    ROLLING_BACK = "ROLLING_BACK"
    FAILED_SAFE = "FAILED_SAFE"

    def __init__(self):
        self.state = self.RUNNING
        self.checkpoint: Checkpoint | None = None
        self.rollbacks = 0

    def commit(self, cp: Checkpoint) -> None:
        """A verified-good state. Only reached by intents that both passed the gates
        AND left the plant inside the safe envelope."""
        if self.state == self.RUNNING:
            self.checkpoint = cp

    def enter_failed_safe(self) -> None:
        self.state = self.FAILED_SAFE


# ---------------------------------------------------------------------------
# Tamper-evident audit log (SHA-256 hash chain, Schneier-Kelsey style)
# ---------------------------------------------------------------------------

GENESIS = "0" * 64


@dataclass
class AuditLog:
    entries: list[dict] = field(default_factory=list)

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else GENESIS

    def append(self, record: dict) -> str:
        prev = self.head
        body = json.dumps(record, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256((prev + body).encode()).hexdigest()
        self.entries.append({"prev": prev, "record": record, "hash": h})
        return h

    def verify(self) -> tuple[bool, int | None]:
        prev = GENESIS
        for i, e in enumerate(self.entries):
            body = json.dumps(e["record"], sort_keys=True, separators=(",", ":"))
            if e["prev"] != prev:
                return False, i
            if hashlib.sha256((prev + body).encode()).hexdigest() != e["hash"]:
                return False, i
            prev = e["hash"]
        return True, None
