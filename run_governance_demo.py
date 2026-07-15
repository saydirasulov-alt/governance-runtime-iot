"""
SELF-CONTAINED end-to-end governance demo on a REAL MQTT stack.

One file. No package layout, no relative imports, no data folder, no path juggling.
Save it anywhere and run:  python run_governance_demo.py

Requires:  pip install paho-mqtt amqtt pyyaml
(works on paho-mqtt 1.x and 2.x)

It starts a real embedded MQTT broker, runs a governance plane that gates control
intents through policy gates G1-G4, an actuator that applies admitted commands, and an
AI service that publishes intents. Everything below is measured on the real stack: real
broker, real network hop, real disk-backed hash-chained audit. Nothing is simulated.
"""
import asyncio, json, hashlib, os, threading, time, uuid, tempfile
import paho.mqtt.client as mqtt

# ---------- paho 1.x / 2.x compatibility ----------
def make_client(cid):
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=cid, protocol=mqtt.MQTTv311)
    except AttributeError:
        return mqtt.Client(client_id=cid, protocol=mqtt.MQTTv311)

# ---------- declarative policy gates G1-G4 ----------
def evaluate(intent, check_source=False):
    sp = intent.get("setpoint")
    if sp is not None and (sp < 15.0 or sp > 30.0):        return "REJECT", "G1"
    if intent.get("action") not in ("set_temperature", "set_mode", "set_fan"): return "REJECT", "G1"
    if intent.get("timestamp") is None:                    return "REJECT", "G2"
    if check_source and intent.get("source") is None:      return "REJECT", "G2"   # corrected variant
    if intent.get("queue_depth", 0) >= 100:                return "THROTTLE", "G3"
    if intent.get("intent_id") is None or intent.get("device_id") is None: return "REJECT", "G4"
    return "PASS", None

# ---------- persistent, tamper-evident hash-chained audit ----------
class Audit:
    def __init__(self, path):
        self.path = path; self.prev = "0"*64; self.seq = 0
        open(path, "w").close()
    def append(self, event, detail):
        self.seq += 1
        rec = {"seq": self.seq, "event": event, "detail": detail, "prev": self.prev}
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        with open(self.path, "a") as f: f.write(json.dumps(rec) + "\n")
        self.prev = rec["hash"]
    @staticmethod
    def verify(path):
        prev = "0"*64; n = 0
        for line in open(path):
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r["prev"] != prev: return False, n
            c = dict(r); h = c.pop("hash")
            if hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest() != h: return False, n
            prev = h; n += 1
        return True, n

# ---------- governance plane (real MQTT service) ----------
class Governance:
    def __init__(self, audit_path, check_source=False):
        self.audit = Audit(audit_path); self.check_source = check_source
        self.stats = {"admitted":0,"rejected":0,"throttled":0}; self.lat = []
        self.c = make_client("governance")
        self.c.on_connect = lambda c,u,f,rc: c.subscribe("intents/#", qos=1)
        self.c.on_message = self._msg
    def _msg(self, c, u, m):
        t0 = time.perf_counter()
        try: intent = json.loads(m.payload.decode())
        except Exception: return
        d, gate = evaluate(intent, self.check_source)
        dev = intent.get("device_id","?")
        if d == "PASS":
            c.publish(f"actuators/{dev}/cmd", json.dumps({"setpoint":intent.get("setpoint")}), qos=1)
            self.stats["admitted"]+=1; self.audit.append("ADMIT", {"intent_id":intent.get("intent_id")})
        else:
            self.stats["throttled" if d=="THROTTLE" else "rejected"]+=1
            self.audit.append(d, {"intent_id":intent.get("intent_id"),"gate":gate})
        self.lat.append((time.perf_counter()-t0)*1000)
    def start(self): self.c.connect("127.0.0.1",1883,30); self.c.loop_start()
    def stop(self): self.c.loop_stop(); self.c.disconnect()

class Actuator:
    def __init__(self):
        self.applied=[]; self.state={"setpoint":22.0}
        self.c=make_client("actuator")
        self.c.on_connect=lambda c,u,f,rc:c.subscribe("actuators/room-1/cmd",qos=1)
        self.c.on_message=lambda c,u,m:(self.applied.append(json.loads(m.payload.decode())),
                                        self.state.update(setpoint=json.loads(m.payload.decode()).get("setpoint") or self.state["setpoint"]))
        self.c.connect("127.0.0.1",1883,30); self.c.loop_start()
    def stop(self): self.c.loop_stop(); self.c.disconnect()

class AI:
    def __init__(self):
        self.c=make_client("ai"); self.c.connect("127.0.0.1",1883,30); self.c.loop_start()
    _M=object()
    def send(self, setpoint, action="set_temperature", timestamp=_M, source="ai_1"):
        ts=time.time() if timestamp is AI._M else timestamp
        self.c.publish("intents/room-1", json.dumps({
            "intent_id":str(uuid.uuid4())[:8],"device_id":"room-1","setpoint":setpoint,
            "action":action,"timestamp":ts,"source":source,"queue_depth":0}), qos=1)
    def stop(self): self.c.loop_stop(); self.c.disconnect()

# ---------- embedded real MQTT broker (amqtt) ----------
def start_broker():
    from amqtt.broker import Broker
    cfg={"listeners":{"default":{"type":"tcp","bind":"127.0.0.1:1883","max_connections":300}},
         "sys_interval":0,"auth":{"allow_anonymous":True},"topic_check":{"enabled":False}}
    loop=asyncio.new_event_loop(); stop=None
    def th():
        nonlocal stop; asyncio.set_event_loop(loop); stop=asyncio.Event()
        async def run():
            b=Broker(cfg); await b.start(); await stop.wait(); await b.shutdown()
        loop.run_until_complete(run())
    t=threading.Thread(target=th,daemon=True); t.start(); time.sleep(1.5)
    return loop, lambda: loop.call_soon_threadsafe(stop.set)

def main():
    d=tempfile.mkdtemp(); audit=os.path.join(d,"audit.jsonl")
    loop, stop_broker = start_broker()
    gov=Governance(audit); gov.start(); time.sleep(0.4)
    act=Actuator(); time.sleep(0.4); ai=AI(); time.sleep(0.3)

    cases=[("safe",dict(setpoint=22.0)),("setback",dict(setpoint=17.0)),
           ("out of band",dict(setpoint=41.0)),("bad action",dict(setpoint=22.0,action="reboot_grid")),
           ("no timestamp",dict(setpoint=22.0,timestamp=None)),("forged source",dict(setpoint=22.0,source=None))]
    t0=time.perf_counter()
    for _ in range(40):
        for _,kw in cases: ai.send(**kw); time.sleep(0.002)
    time.sleep(2.0); wall=time.perf_counter()-t0; ai.stop(); time.sleep(0.2)

    lat=sorted(gov.lat); p50=lat[len(lat)//2]; p90=lat[int(len(lat)*0.9)]
    ok,n=Audit.verify(audit)
    print("="*70)
    print("END-TO-END ON A REAL MQTT STACK (broker + governance + actuator)")
    print("="*70)
    print(f"  intents published        : 240")
    print(f"  admitted / rejected      : {gov.stats['admitted']} / {gov.stats['rejected']}")
    print(f"  actuator commands applied: {len(act.applied)}")
    print(f"  MEASURED decision latency: median {p50:.3f} ms   P90 {p90:.3f} ms   (real stack)")
    print(f"  wall clock (240 intents) : {wall:.2f} s")
    print(f"  audit records            : {n}   chain intact: {ok}")

    lines=open(audit).readlines(); r=json.loads(lines[5]); r["detail"]["intent_id"]="TAMPER"
    lines[5]=json.dumps(r)+"\n"; tp=os.path.join(d,"tampered.jsonl"); open(tp,"w").writelines(lines)
    print(f"  after tampering one record on disk, chain verifies: {Audit.verify(tp)[0]}")

    print("\n  --- live policy-gap diagnosis (same running stack) ---")
    gov.stop(); time.sleep(0.3)
    for label, cs in [("SHIPPED policy ", False), ("CORRECTED policy", True)]:
        g=Governance(os.path.join(d,f"a_{cs}.jsonl"), check_source=cs); g.start(); time.sleep(0.3)
        a=AI()
        for _ in range(40): a.send(setpoint=22.0, source=None)
        time.sleep(1.2); a.stop()
        print(f"    forged-provenance admitted under {label}: {g.stats['admitted']}/40"
              + ("   <-- the G2 gap" if not cs else "   <-- gap closed"))
        g.stop()

    act.stop(); stop_broker(); time.sleep(0.5)
    print("\n  (all numbers measured on a real MQTT stack; no simulated timing)")

if __name__=="__main__":
    main()
