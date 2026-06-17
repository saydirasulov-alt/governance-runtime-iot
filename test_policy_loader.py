"""Verification for the config-based policy loader (ISH 1).

Run from the repo root:  python3 test_policy_loader.py

Demonstrates, against reviewer concern (governance policy 'only definable by
direct modification of Python source code'):
  1. Gates load from an external YAML/JSON file, not hardcoded Python.
  2. The default config is byte-for-byte equivalent to the legacy gates
     (400k randomised intents, zero mismatches).
  3. Governance behaviour changes by editing the CONFIG ONLY:
       - corrected-G2 (source check) closes the partial-corruption bypass,
       - tightening the safe band rejects previously-admitted setpoints.
"""
import random, time, copy
from governance_runtime import GovernanceRuntime
from policy_loader import load_policy, evaluate_policy


def legacy_gates(intent, qd, g2_check_source):
    cfg = {"t_min": 15.0, "t_max": 30.0,
           "allowed_actions": ["set_temperature", "set_mode", "set_fan"], "q_max": 100}
    sp = intent.get("setpoint")
    if sp is not None and (sp < cfg["t_min"] or sp > cfg["t_max"]): return "REJECT", "G1"
    if intent.get("action") not in cfg["allowed_actions"]: return "REJECT", "G1"
    if intent.get("timestamp") is None: return "REJECT", "G2"
    if g2_check_source and intent.get("source") is None: return "REJECT", "G2"
    if qd >= cfg["q_max"]: return "THROTTLE", "G3"
    if intent.get("intent_id") is None or intent.get("device_id") is None: return "REJECT", "G4"
    return "PASS", None


def test_equivalence(N=200000):
    random.seed(0)
    actions = ["set_temperature", "set_mode", "set_fan", "bad_action", None]
    mism = 0
    for g2 in (False, True):
        rt = GovernanceRuntime(g2_check_source=g2)
        for _ in range(N):
            intent = {
                "setpoint": random.choice([None, 14.9, 15.0, 22.0, 30.0, 30.1, 8.0, 41.0]),
                "action": random.choice(actions),
                "timestamp": random.choice([None, time.time()]),
                "source": random.choice([None, "s1"]),
                "intent_id": random.choice([None, "i1"]),
                "device_id": random.choice([None, "d1"]),
            }
            qd = random.choice([0, 50, 99, 100, 150])
            if rt._evaluate_gates(intent, qd) != legacy_gates(intent, qd, g2):
                mism += 1
    print(f"[1] Config vs legacy: {2*N:,} comparisons, {mism} mismatches "
          f"-> {'EQUIVALENT' if mism == 0 else 'FAILED'}")
    return mism == 0


def test_g2_via_config():
    def run(cfg):
        total = bypass = 0
        for seed in range(1, 31):
            random.seed(seed)
            rt = GovernanceRuntime(policy_config=cfg)
            for i in range(200):
                intent = {"setpoint": 22.0, "action": "set_temperature", "timestamp": time.time(),
                          "source": None, "intent_id": f"i_{i}", "device_id": f"d_{i}"}
                r = rt.admit(intent, queue_depth=0)
                total += 1
                if r["outcome"] == "ADMITTED" and r.get("ground_truth_unsafe"): bypass += 1
        return 100.0 * bypass / total
    gap = run("policy_config.yaml")
    fix = run("policy_config_corrected.yaml")
    ok = gap > 90 and fix == 0.0
    print(f"[2] Partial-corruption via config only: gap={gap:.1f}%  corrected={fix:.1f}%  "
          f"-> {'OK' if ok else 'FAILED'}")
    return ok


def test_tighten_band():
    base = load_policy("policy_config.yaml")
    tight = copy.deepcopy(base)
    for g in tight["gates"]:
        if g["id"] == "G1":
            g["rules"][0]["max"] = 25.0   # operator narrows safe band, no code edit
    def ctx(sp): return {"setpoint": sp, "action": "set_temperature", "timestamp": time.time(),
                         "source": "s", "intent_id": "i", "device_id": "d", "queue_depth": 0}
    ok = (evaluate_policy(base, ctx(27.0))[0] == "PASS"
          and evaluate_policy(tight, ctx(27.0)) == ("REJECT", "G1"))
    print(f"[3] Tighten safe band [15,30]->[15,25] via config: setpoint=27 "
          f"PASS->REJECT -> {'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    print("=== ISH 1: config-based policy loader verification ===")
    results = [test_equivalence(), test_g2_via_config(), test_tighten_band()]
    print("\nALL PASSED" if all(results) else "\nSOME TESTS FAILED")
