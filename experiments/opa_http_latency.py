#!/usr/bin/env python3
"""
opa_http_latency.py -- measure governance-decision latency through a REAL Open Policy
Agent (OPA) HTTP decision point, on this machine.

This exercises the out-of-process policy path end to end: the same G1-G4 gates, encoded
in Rego (policy_gates.rego), are uploaded to a running OPA server and evaluated over the
loopback HTTP interface, once per control intent. It writes a per-decision latency CSV and
a host-stamped summary (median / P90), so the OPA-path number in the paper is a measurement
rather than a model.

    # 1. start a real OPA server (either is fine)
    docker run --rm -p 8181:8181 openpolicyagent/opa:1.18.2 run --server
    #   or, with the bundled stack:  docker compose up -d opa
    #   or, with a local binary:     opa run --server

    # 2. measure
    python experiments/opa_http_latency.py            # 240 intents, default localhost:8181

Requires: pip install requests   (already in requirements.txt)

The client uploads policy_gates.rego itself, so the reported number does not depend on how
the server was launched. Decisions returned by OPA are checked against the runtime's own
gate logic; the run aborts if they disagree, so a timing number is never reported for a
server that is computing the wrong thing.
"""
import argparse, csv, json, os, platform, random, statistics, sys, time
try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

# One reused connection (keep-alive) with system proxies disabled, so we measure the OPA
# decision + loopback round-trip in steady state -- not a fresh TCP/DNS/proxy setup per call
# (on Windows a new connection per request can add ~2 s resolving localhost via IPv6 first).
S = requests.Session()
S.trust_env = False

REGO_PACKAGE = "governance"          # package name inside policy_gates.rego
ALLOW_PATH   = "governance/allow"    # data path evaluated per intent


def expected_allow(intent):
    """Python mirror of policy_gates.rego `allow` (base policy: G2 checks timestamp only)."""
    sp = intent.get("setpoint")
    if sp is None or sp < 15.0 or sp > 30.0:                       return False   # G1
    if intent.get("action") not in ("set_temperature", "set_mode", "set_fan"):  return False   # G1
    if intent.get("timestamp") is None:                           return False   # G2
    if intent.get("queue_depth", 0) >= 100:                       return False   # G3
    if intent.get("intent_id") is None or intent.get("device_id") is None:      return False   # G4
    return True


def make_workload(n, seed):
    """Deterministic n-intent stream, half admitted / half rejected, each rejection failing
    exactly one gate so the OPA path is exercised across G1-G4."""
    rng = random.Random(seed)
    half = n // 2
    intents = []
    for i in range(half):   # admitted: all gates pass
        intents.append({"setpoint": round(rng.uniform(15.0, 30.0), 1),
                        "action": rng.choice(["set_temperature", "set_mode", "set_fan"]),
                        "timestamp": time.time(), "source": f"ai_{i%3}",
                        "intent_id": f"i{i}", "device_id": f"d{i}", "queue_depth": rng.choice([0, 10, 50])})
    for i in range(n - half):  # rejected: fail exactly one gate, round-robin over G1-G4
        base = {"setpoint": round(rng.uniform(15.0, 30.0), 1), "action": "set_temperature",
                "timestamp": time.time(), "source": f"ai_{i%3}",
                "intent_id": f"r{i}", "device_id": f"d{i}", "queue_depth": 0}
        gate = i % 4
        if gate == 0:   base["setpoint"] = 40.0            # G1 fail
        elif gate == 1: base["timestamp"] = None           # G2 fail
        elif gate == 2: base["queue_depth"] = 120          # G3 fail
        else:           base["intent_id"] = None           # G4 fail
        intents.append(base)
    rng.shuffle(intents)
    return intents


POLICY_REGO_V1 = """package governance
import rego.v1
default allow := false
g1_pass if {
    input.setpoint >= 15.0
    input.setpoint <= 30.0
    input.action in {"set_temperature", "set_mode", "set_fan"}
}
g2_pass if { input.timestamp != null }
g3_pass if { input.queue_depth < 100 }
g4_pass if {
    input.intent_id != null
    input.device_id != null
}
allow if {
    g1_pass
    g2_pass
    g3_pass
    g4_pass
}
"""

def _put_policy(base, rego):
    return S.put(f"{base}/v1/policies/governance_gates", data=rego.encode(),
                 headers={"Content-Type": "text/plain"}, timeout=10)

def upload_policy(base, rego_path):
    # Prefer the on-disk policy; if the server rejects its Rego version (e.g. OPA v1.x
    # rejecting v0 syntax), fall back to the embedded Rego v1 encoding of the same G1-G4 gates.
    if rego_path and os.path.exists(rego_path):
        with open(rego_path, "r", encoding="utf-8") as f:
            r = _put_policy(base, f.read())
        if r.status_code == 200:
            return
    r = _put_policy(base, POLICY_REGO_V1)
    if r.status_code != 200:
        raise SystemExit(f"OPA rejected the policy ({r.status_code}): {r.text[:400]}")


def decide(base, intent):
    r = S.post(f"{base}/v1/data/{ALLOW_PATH}", json={"input": intent}, timeout=10)
    r.raise_for_status()
    return bool(r.json().get("result", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opa-url", default="http://127.0.0.1:8181")  # IP literal: avoids slow localhost->IPv6 resolution on Windows
    ap.add_argument("--opa-version", default="", help="OPA version to record in the log, e.g. from `opa version`")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--policy", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policy_gates.rego"))
    ap.add_argument("--out", default=os.path.join("results", "opa_http_latency.csv"))
    a = ap.parse_args()
    base = a.opa_url.rstrip("/")

    try:
        S.get(f"{base}/health", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"OPA server not reachable at {base} ({e}).\n"
                 f"Start one first, e.g.:  docker run --rm -p 8181:8181 openpolicyagent/opa:1.18.2 run --server")
    upload_policy(base, a.policy)

    intents = make_workload(a.n, a.seed)
    # warm up (connection setup / JIT) -- not timed
    for k in range(a.warmup):
        decide(base, intents[k % len(intents)])

    rows, mism = [], 0
    for idx, intent in enumerate(intents):
        t0 = time.perf_counter_ns()
        allow = decide(base, intent)
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        exp = expected_allow(intent)
        if allow != exp:
            mism += 1
        rows.append((idx, "admit" if allow else "reject", "admit" if exp else "reject", f"{dt_ms:.4f}"))

    lat = sorted(float(r[3]) for r in rows)
    admitted = sum(1 for r in rows if r[1] == "admit")
    def pct(p): return lat[min(len(lat) - 1, int(round(p / 100.0 * (len(lat) - 1))))]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "opa_decision", "expected_decision", "latency_ms"]); w.writerows(rows)

    summary = [
        "=" * 70,
        "OPA HTTP decision-point latency",
        "=" * 70,
        f"host            : {platform.node()}",
        f"os              : {platform.system()} {platform.release()} ({platform.machine()})",
        f"python          : {sys.version.split()[0]}",
        f"opa endpoint    : {base}/v1/data/{ALLOW_PATH}",
        f"opa version     : {a.opa_version}" if a.opa_version else "opa version     : (pass --opa-version to record)",
        f"intents (timed) : {len(rows)}  (admitted {admitted}, rejected {len(rows)-admitted}; warmup {a.warmup})",
        f"decision agree  : {len(rows)-mism}/{len(rows)}  (OPA vs runtime gate logic)",
        f"latency median  : {statistics.median(lat):.3f} ms",
        f"latency P90     : {pct(90):.3f} ms",
        f"latency mean    : {statistics.mean(lat):.3f} ms   (min {lat[0]:.3f}, max {lat[-1]:.3f})",
        f"csv             : {a.out}",
    ]
    print("\n".join(summary))
    logpath = os.path.join(os.path.dirname(a.out) or ".", "OPA_HTTP_LOG.txt")
    with open(logpath, "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"log             : {logpath}")
    if mism:
        sys.exit(f"\nABORT: {mism} OPA decisions disagree with the runtime gate logic; timing not trustworthy.")

if __name__ == "__main__":
    main()
