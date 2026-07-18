import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.audit import AuditLog

def test_chain_intact_and_tamper_evident():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.jsonl")
        log = AuditLog(p)
        for i in range(20):
            log.append("ADMIT", {"intent_id": f"i{i}"})
        ok, n = log.verify()
        assert ok and n == 20
        lines = open(p).readlines()
        r = json.loads(lines[5]); r["detail"]["intent_id"] = "TAMPERED"
        lines[5] = json.dumps(r) + "\n"
        open(p, "w").writelines(lines)
        ok2, _ = AuditLog(p).verify()
        assert ok2 is False          # tampering on disk is detected
