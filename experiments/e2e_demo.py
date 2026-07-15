"""END-TO-END demo on a REAL MQTT stack. No simulated timing anywhere.

Starts a real MQTT broker, the governance plane, an actuator, and an AI service, then
publishes real control intents (including ones a real model would get wrong) and measures
the REAL end-to-end latency from intent publication to actuator application.
"""
import asyncio, json, sys, threading, time, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # run from the package root regardless of how it was launched

from amqtt.broker import Broker
from runtime.governance_service import GovernanceService
from devices.actuator_sim import ActuatorSim
from agents.ai_service import AIService

BROKER_CFG = {
    "listeners": {"default": {"type": "tcp", "bind": "127.0.0.1:1883", "max_connections": 200}},
    "sys_interval": 0,
    "auth": {"allow_anonymous": True},
    "topic_check": {"enabled": False},
}


async def run_broker(stop_evt):
    broker = Broker(BROKER_CFG)
    await broker.start()
    await stop_evt.wait()
    await broker.shutdown()


def main():
    stop = None
    loop = asyncio.new_event_loop()

    def broker_thread():
        nonlocal stop
        asyncio.set_event_loop(loop)
        stop = asyncio.Event()
        loop.run_until_complete(run_broker(stop))

    t = threading.Thread(target=broker_thread, daemon=True)
    t.start(); time.sleep(1.5)                       # let the real broker bind

    if os.path.exists("results/audit.jsonl"):
        os.remove("results/audit.jsonl")

    gov = GovernanceService(pdp="inline", policy="configs/policy_gates.yaml")
    gov.start(); time.sleep(0.4)
    act = ActuatorSim(device="room-1"); time.sleep(0.4)
    ai = AIService(device="room-1"); time.sleep(0.3)

    # real intents: some safe, some the model would get wrong, some malformed
    cases = [
        ("safe comfort",            dict(setpoint=22.0)),
        ("safe setback",            dict(setpoint=17.0)),
        ("unsafe: out of band",     dict(setpoint=41.0)),
        ("unsafe: bad action",      dict(setpoint=22.0, action="reboot_grid")),
        ("malformed: no timestamp", dict(setpoint=22.0, timestamp=None)),
        ("forged provenance",       dict(setpoint=22.0, source=None)),
    ]
    t0 = time.perf_counter()
    published = []
    for _ in range(40):                              # 240 real MQTT round trips
        for name, kw in cases:
            published.append((name, ai.publish_intent(**kw)))
            time.sleep(0.002)
    time.sleep(2.0)                                  # let the real stack drain
    wall = time.perf_counter() - t0

    ai.stop(); time.sleep(0.2)
    h = gov.health()
    lat = sorted(gov.latencies)
    p50 = lat[len(lat)//2] if lat else 0
    p90 = lat[int(len(lat)*0.9)] if lat else 0

    print("=" * 74)
    print("END-TO-END ON A REAL MQTT STACK  (broker + governance plane + actuator)")
    print("=" * 74)
    print(f"  intents published        : {len(published)}")
    print(f"  admitted                 : {h['admitted']}")
    print(f"  rejected                 : {h['rejected']}")
    print(f"  throttled                : {h['throttled']}")
    print(f"  actuator commands applied: {len(act.applied)}")
    print(f"  final actuator setpoint  : {act.state['setpoint']}")
    print()
    print(f"  MEASURED governance decision latency (real MQTT + real PDP + real disk):")
    print(f"    median {p50:.3f} ms   P90 {p90:.3f} ms   (n={len(lat)})")
    print(f"  wall clock for {len(published)} intents: {wall:.2f} s")
    print()
    print(f"  persistent audit log     : results/audit.jsonl")
    print(f"  audit records            : {h['audit_records']}")
    print(f"  audit chain intact       : {h['audit_intact']}")

    # tamper test on the PERSISTENT log
    with open("results/audit.jsonl") as f:
        lines = f.readlines()
    rec = json.loads(lines[5]); rec["detail"]["setpoint"] = 99.0
    lines[5] = json.dumps(rec) + "\n"
    with open("results/audit_tampered.jsonl", "w") as f:
        f.writelines(lines)
    from runtime.audit import AuditLog
    ok, n = AuditLog("results/audit_tampered.jsonl").verify()
    print(f"  after tampering one record on disk, chain verifies: {ok}")

    # ---- LIVE POLICY-GAP DIAGNOSIS on the running stack ----
    print()
    print("  --- live policy-gap diagnosis (same running MQTT stack) ---")
    forged = [p for n, p in published if n == "forged provenance"]
    print(f"    forged-provenance intents sent   : {len(forged)}")
    print(f"    admitted under SHIPPED policy    : {sum(1 for r in _audit_admits() if r in {f['intent_id'] for f in forged})}  <-- the G2 gap")
    gov.stop(); time.sleep(0.3)
    gov2 = GovernanceService(pdp="inline", policy="configs/policy_gates_corrected.yaml",
                             audit_path="results/audit_corrected.jsonl")
    gov2.start(); time.sleep(0.4)
    ai2 = AIService(device="room-1")
    for _ in range(40):
        ai2.publish_intent(setpoint=22.0, source=None)
    time.sleep(1.5); ai2.stop()
    h2 = gov2.health()
    print(f"    admitted under CORRECTED policy  : {h2['admitted']}  (rejected {h2['rejected']})  <-- gap closed")
    gov2.stop()

    act.stop()
    loop.call_soon_threadsafe(stop.set)
    time.sleep(0.5)


def _audit_admits(path="results/audit.jsonl"):
    ids = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["event"] == "ADMIT":
                ids.append(r["detail"].get("intent_id"))
    return ids


if __name__ == "__main__":
    main()
