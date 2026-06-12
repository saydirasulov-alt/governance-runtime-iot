"""Demonstrates the tamper-evident SHA256 audit chain (G4 Auditability).
Run from the supplementary/ directory: python3 verify_audit_chain.py
"""
import hashlib, json, time
from governance_runtime import GovernanceRuntime

def verify_chain(audit_log):
    """Recompute the hash chain; return True only if intact."""
    prev = "0" * 64
    for rec in audit_log:
        if rec["prev_hash"] != prev:
            return False
        check = dict(rec); h = check.pop("hash")
        if hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest() != h:
            return False
        prev = h
    return True

if __name__ == "__main__":
    rt = GovernanceRuntime()
    for i in range(20):
        rt.admit({"setpoint": 22.0, "action": "set_temperature", "timestamp": time.time(),
                  "source": f"s{i}", "intent_id": f"i{i}", "device_id": f"d{i}"}, queue_depth=0)

    print(f"Audit records appended: {len(rt.audit_log)}")
    print(f"Intact chain verifies as valid: {verify_chain(rt.audit_log)}")

    # Tamper with one record; the chain must now fail verification
    rt.audit_log[5]["id"] = "TAMPERED"
    print(f"After tampering one record, chain verifies as valid: {verify_chain(rt.audit_log)}")
    print("\nThe audit log is a SHA256 hash chain (each record commits to the previous),")
    print("so any modification of a past record is detectable. This realizes the G4")
    print("auditability gate referenced in the manuscript.")
