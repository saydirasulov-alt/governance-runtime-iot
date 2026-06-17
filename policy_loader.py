"""Declarative policy loader for the Governance Runtime.

Gates G1-G4 are defined in an external configuration file (YAML or JSON) as an
ordered list of predicates, rather than being hardcoded in Python. This lets an
operator change the governance policy -- bounds, allowed actions, which fields are
mandatory, the throttle threshold, or whether the G2 source-field check is active --
WITHOUT editing any source code. The same file can be evaluated by the Python
runtime and (for the safety subset) by the OPA engine via policy_gates.rego.

Rule semantics (chosen to match the original hardcoded _evaluate_gates exactly):
    range     -> if field present and not None and (val < min or val > max): fail
                 (nullable: a missing/None field is skipped, not failed)
    allowed   -> if field value is not in `values`: fail  (None also fails)
    required  -> if field is missing or None: fail
    threshold -> if field value >= max: fail
Rules with `enabled: false` are skipped. The first failing rule decides the
outcome, returning (on_fail, gate_id). If no rule fails, the result is ("PASS", None).
"""
import json
import os

# Default policy, embedded so the runtime works even with no config file present.
# Byte-for-byte equivalent to the legacy hardcoded gates.
DEFAULT_POLICY = {
    "gates": [
        {"id": "G1", "name": "Safety", "rules": [
            {"type": "range", "field": "setpoint", "min": 15.0, "max": 30.0,
             "nullable": True, "on_fail": "REJECT"},
            {"type": "allowed", "field": "action",
             "values": ["set_temperature", "set_mode", "set_fan"], "on_fail": "REJECT"},
        ]},
        {"id": "G2", "name": "Privacy", "rules": [
            {"type": "required", "field": "timestamp", "on_fail": "REJECT"},
            # Source-field check: disabled by default (the specification gap analysed
            # in the paper). Enabling it is the "corrected-G2" variant -- a pure config
            # change, no code edit. See policy_config_corrected.yaml.
            {"type": "required", "field": "source", "enabled": False, "on_fail": "REJECT"},
        ]},
        {"id": "G3", "name": "Resilience", "rules": [
            {"type": "threshold", "field": "queue_depth", "max": 100, "on_fail": "THROTTLE"},
        ]},
        {"id": "G4", "name": "Auditability", "rules": [
            {"type": "required", "field": "intent_id", "on_fail": "REJECT"},
            {"type": "required", "field": "device_id", "on_fail": "REJECT"},
        ]},
    ]
}


def load_policy(path=None):
    """Load a policy spec from YAML or JSON. None -> bundled default.

    If a .yaml/.yml path is given but PyYAML is unavailable, a sibling .json file
    with the same stem is used as a dependency-free fallback.
    """
    if path is None:
        return DEFAULT_POLICY
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy config not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f:
            return json.load(f)
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            twin = os.path.splitext(path)[0] + ".json"
            if os.path.exists(twin):
                with open(twin) as f:
                    return json.load(f)
            raise ImportError(
                "PyYAML not installed and no sibling .json fallback found for "
                f"{path}. Run `pip install pyyaml` or provide {twin}.")
    raise ValueError(f"Unsupported policy config extension: {ext}")


def _rule_fails(rule, ctx):
    """Return True if this rule's predicate is violated by ctx."""
    if rule.get("enabled", True) is False:
        return False
    rtype = rule["type"]
    field = rule.get("field")
    val = ctx.get(field)
    if rtype == "range":
        if val is None:
            return False if rule.get("nullable", True) else True
        return val < rule["min"] or val > rule["max"]
    if rtype == "allowed":
        return val not in rule["values"]
    if rtype == "required":
        return val is None
    if rtype == "threshold":
        if val is None:
            val = 0
        return val >= rule["max"]
    raise ValueError(f"Unknown rule type: {rtype}")


def evaluate_policy(policy, ctx):
    """Evaluate gates in order. Return (decision, gate_id).

    ctx is the intent dict augmented with 'queue_depth'. The first failing rule
    determines the outcome; otherwise ("PASS", None).
    """
    for gate in policy["gates"]:
        for rule in gate["rules"]:
            if _rule_fails(rule, ctx):
                return rule.get("on_fail", "REJECT"), gate["id"]
    return "PASS", None


def set_rule_enabled(policy, gate_id, field, enabled):
    """Toggle a named rule (e.g., G2/source) in-place. Returns the policy."""
    for gate in policy["gates"]:
        if gate["id"] == gate_id:
            for rule in gate["rules"]:
                if rule.get("field") == field:
                    rule["enabled"] = enabled
    return policy
