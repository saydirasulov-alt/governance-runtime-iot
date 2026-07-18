"""Entrypoint: run the governance plane as a long-lived service."""
import argparse, time, signal, sys
from .governance_service import GovernanceService

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--pdp", choices=["inline", "http", "opa"], default="inline")
    ap.add_argument("--policy", default="configs/policy_gates.yaml")
    ap.add_argument("--opa-url", default="http://127.0.0.1:8181/v1/data/governance/decision")
    ap.add_argument("--audit", default="results/audit.jsonl")
    a = ap.parse_args()
    kw = {"url": a.opa_url} if a.pdp == "opa" else ({} if a.pdp == "inline" else {})
    gov = GovernanceService(broker=a.broker, port=a.port, pdp=a.pdp,
                            policy=a.policy, audit_path=a.audit, **kw)
    gov.start()
    print(f"[governance] up: broker={a.broker}:{a.port} pdp={a.pdp} policy={a.policy}", flush=True)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    try:
        while not stop["v"]:
            time.sleep(5)
            print(f"[governance] {gov.health()}", flush=True)
    finally:
        gov.stop()

if __name__ == "__main__":
    main()
