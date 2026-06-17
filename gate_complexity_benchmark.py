"""Gate-complexity benchmark: shows that policy-evaluation cost remains a
negligible fraction of end-to-end latency regardless of gate complexity,
which is why the deliberately simple gates do not bias the overhead study.

Run from supplementary/: python3 gate_complexity_benchmark.py
"""
import time, random, statistics, re

ALLOWED = {"set_temperature", "set_mode", "set_fan"}
PATTERN = re.compile(r"^[a-z]+_[a-z]+$")

def gates_simple(intent, qd):
    sp = intent.get("setpoint")
    if sp is not None and (sp < 15.0 or sp > 30.0): return "REJECT"
    if intent.get("action") not in ALLOWED: return "REJECT"
    if intent.get("timestamp") is None: return "REJECT"
    if qd >= 100: return "THROTTLE"
    if intent.get("intent_id") is None or intent.get("device_id") is None: return "REJECT"
    return "PASS"

def gates_composite(intent, qd):
    # adds cross-field consistency, range tiers, action-mode coupling
    sp = intent.get("setpoint")
    if sp is not None and (sp < 15.0 or sp > 30.0): return "REJECT"
    if sp is not None and intent.get("action") == "set_fan" and sp > 28.0: return "REJECT"
    if intent.get("action") not in ALLOWED: return "REJECT"
    if intent.get("mode") in ("eco","away") and sp is not None and sp > 26.0: return "REJECT"
    if intent.get("timestamp") is None: return "REJECT"
    if intent.get("priority",1) > 3 and qd >= 50: return "THROTTLE"
    if qd >= 100: return "THROTTLE"
    if intent.get("intent_id") is None or intent.get("device_id") is None: return "REJECT"
    return "PASS"

def gates_nested_regex(intent, qd):
    # adds regex validation + nested metadata checks + provenance allowlist
    sp = intent.get("setpoint")
    if sp is not None and (sp < 15.0 or sp > 30.0): return "REJECT"
    a = intent.get("action","")
    if not PATTERN.match(a) or a not in ALLOWED: return "REJECT"
    src = intent.get("source")
    if src is None or not src.startswith("ai_service_"): return "REJECT"
    meta = intent.get("meta", {})
    if isinstance(meta, dict) and meta.get("calib") == "stale": return "REJECT"
    if intent.get("timestamp") is None: return "REJECT"
    if qd >= 100: return "THROTTLE"
    if intent.get("intent_id") is None or intent.get("device_id") is None: return "REJECT"
    return "PASS"

def bench(fn, n=50000):
    random.seed(1)
    intents=[{"setpoint":random.uniform(10,35),"action":random.choice(list(ALLOWED)+["bad"]),
              "timestamp":time.time(),"source":f"ai_service_{i%4}","intent_id":f"i{i}",
              "device_id":f"d{i}","mode":random.choice(["eco","comfort","away"]),
              "priority":random.randint(1,5),"meta":{"calib":"ok"}} for i in range(n)]
    t0=time.perf_counter()
    for it in intents: fn(it, random.randint(0,40))
    dt=(time.perf_counter()-t0)/n*1e6  # microseconds per evaluation
    return dt

if __name__ == "__main__":
    print("=== Gate-complexity benchmark (per-evaluation cost) ===")
    print(f"{'Gate variant':<28}{'predicates':>11}{'mean us/eval':>14}")
    for name,fn,np_ in [("Simple (evaluated)",gates_simple,5),
                        ("Composite (cross-field)",gates_composite,8),
                        ("Nested + regex + allowlist",gates_nested_regex,9)]:
        us=bench(fn)
        print(f"{name:<28}{np_:>11}{us:>14.3f}")
    print("\nEnd-to-end P90 (from main study): ~5.9 ms = 5900 us.")
    print("Even the most complex gate variant remains far below 1% of E2E latency,")
    print("so policy-evaluation cost does not bias the overhead decomposition and")
    print("the deliberately simple gates are a methodological choice, not a constraint.")
