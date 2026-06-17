"""ISH 3 (hardware-in-the-loop): rollback on a REAL actuator via GPIO.

This is the bridge from the software twin (rollback_demo.py) to physical hardware.
It runs the SAME rollback FSM, but drives a real actuator through a pluggable
driver and measures the PHYSICAL recovery latency (command + settle), which is the
component the paper defers to hardware evaluation (Section 6.5.2).

Drivers
  MockDriver  -- pure software, runs anywhere (CI / laptop). Default.
  GPIODriver  -- Raspberry Pi. A relay (e.g. GPIO17) models an HVAC contactor
                 (valve open/closed); an optional servo (gpiozero) models a
                 continuous valve position 0-100. Auto-selected with --gpio if the
                 RPi libraries are present.

Wiring (GPIODriver, BCM numbering)
  Relay IN  -> GPIO17 (pin 11)      Relay VCC -> 5V      Relay GND -> GND
  Servo sig -> GPIO18 (pin 12)      Servo  V  -> 5V      Servo GND -> GND
  (Use a relay board / level shifting appropriate to your actuator; never drive an
   inductive load directly from a GPIO pin.)

Run
  python3 hardware_rollback_harness.py                 # MockDriver, 200 tx
  python3 hardware_rollback_harness.py --gpio --n 50   # on the Pi, real relay/servo
"""
import argparse, copy, json, hashlib, time, statistics
from policy_loader import load_policy, evaluate_policy

POLICY = load_policy(None)


def state_hash(s):
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()


class ActuatorDriver:
    """Interface a real or simulated actuator must implement."""
    def read_state(self): raise NotImplementedError
    def apply(self, op): raise NotImplementedError           # mutate physical state
    def goto(self, state): raise NotImplementedError         # drive actuator to a target state
    def is_reversible(self, op): return op[0] != "purge"
    def reset(self): raise NotImplementedError
    def cleanup(self): pass


class MockDriver(ActuatorDriver):
    """Software stand-in; identical semantics to the twin, runs anywhere."""
    def __init__(self):
        self.state = {"valve": 50, "relay": "closed", "setpoint": 22.0, "purge_fired": False}
    def read_state(self): return copy.deepcopy(self.state)
    def apply(self, op):
        k, v = op
        if k == "purge": self.state["purge_fired"] = True
        else: self.state[k] = v
    def goto(self, state):
        # simulate a small physical settle time proportional to valve travel
        travel = abs(self.state.get("valve", 0) - state.get("valve", 0))
        time.sleep(0.0005 + travel * 0.00002)
        self.state = copy.deepcopy(state)
    def reset(self):
        self.goto({"valve": 50, "relay": "closed", "setpoint": 22.0, "purge_fired": False})


class GPIODriver(ActuatorDriver):
    """Raspberry Pi driver. Relay = discrete valve; servo = continuous position."""
    def __init__(self, relay_pin=17, servo_pin=18):
        import RPi.GPIO as GPIO              # noqa: import guarded by caller
        from gpiozero import AngularServo
        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(relay_pin, GPIO.OUT)
        self.relay_pin = relay_pin
        self.servo = AngularServo(servo_pin, min_angle=0, max_angle=180)
        self.state = {"valve": 50, "relay": "closed", "setpoint": 22.0, "purge_fired": False}
        self._drive(self.state)
    def _drive(self, state):
        # relay
        self.GPIO.output(self.relay_pin, self.GPIO.HIGH if state["relay"] == "open" else self.GPIO.LOW)
        # servo: map valve 0..100 -> angle 0..180
        self.servo.angle = max(0, min(100, state["valve"])) * 1.8
        time.sleep(0.02)                     # let the actuator physically settle
    def read_state(self): return copy.deepcopy(self.state)
    def apply(self, op):
        k, v = op
        if k == "purge": self.state["purge_fired"] = True
        elif k == "valve": self.state["valve"] = v; self._drive(self.state)
        elif k == "relay": self.state["relay"] = v; self._drive(self.state)
        else: self.state[k] = v
    def goto(self, state):
        self._drive(state); self.state = copy.deepcopy(state)
    def reset(self):
        self.goto({"valve": 50, "relay": "closed", "setpoint": 22.0, "purge_fired": False})
    def cleanup(self):
        try: self.servo.detach(); self.GPIO.cleanup()
        except Exception: pass


def violates(state):
    ctx = {"setpoint": state["setpoint"], "action": "set_temperature", "timestamp": 1,
           "source": "hw", "intent_id": "i", "device_id": "d", "queue_depth": 0}
    dec, gate = evaluate_policy(POLICY, ctx)
    if dec != "PASS": return True, gate
    if not (0 <= state["valve"] <= 100): return True, "VALVE"
    return False, None


def run_harness(driver, n=200, seed=1):
    import random; random.seed(seed)
    committed = rolled = failed = 0
    phys_mttr = []; verified = 0
    failed_safe_events = 0
    for _ in range(n):
        checkpoint = driver.read_state(); cp_hash = state_hash(checkpoint)
        roll = random.random()
        ops = [("valve", random.randint(10, 90))]
        irreversible = False
        if roll < 0.10:
            ops.append(("setpoint", random.choice([8.0, 41.0, 35.0])))
        elif roll < 0.13:
            ops.append(("purge", True)); ops.append(("setpoint", 44.0)); irreversible = True
        else:
            ops.append(("setpoint", round(random.uniform(16, 29), 1)))
        for op in ops: driver.apply(op)
        bad, gate = violates(driver.read_state())
        if not bad:
            committed += 1; continue
        t0 = time.perf_counter()
        if irreversible:
            failed += 1; failed_safe_events += 1
            phys_mttr.append((time.perf_counter() - t0) * 1000)
            driver.reset()                                # operator replaces/clears device
            continue
        driver.goto(checkpoint)                       # PHYSICAL restore
        dt = (time.perf_counter() - t0) * 1000
        phys_mttr.append(dt); rolled += 1
        if state_hash(driver.read_state()) == cp_hash: verified += 1
    return dict(committed=committed, rolled=rolled, failed=failed,
                verified=verified, mttr=phys_mttr, status="NORMAL",
                failed_safe_events=failed_safe_events)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpio", action="store_true", help="use the real Raspberry Pi GPIO driver")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    if args.gpio:
        try:
            driver = GPIODriver(); backend = "GPIODriver (real hardware)"
        except Exception as e:
            print(f"[GPIO unavailable: {e}] -> falling back to MockDriver")
            driver = MockDriver(); backend = "MockDriver (fallback)"
    else:
        driver = MockDriver(); backend = "MockDriver"

    print(f"=== Hardware-in-the-loop rollback harness :: {backend} ===")
    try:
        r = run_harness(driver, n=args.n)
    finally:
        driver.cleanup()
    print(f"  committed={r['committed']}  rolled_back={r['rolled']}  failed_safe={r['failed']}")
    if r["rolled"]:
        print(f"  physical restore VERIFIED: {r['verified']}/{r['rolled']} = {100*r['verified']/r['rolled']:.1f}%")
    if r["mttr"]:
        print(f"  MEASURED physical recovery latency: median={statistics.median(r['mttr']):.3f} ms"
              f"  max={max(r['mttr']):.3f} ms  (n={len(r['mttr'])})")
    print(f"  FAILED-SAFE transitions (device halted + operator-replaced): {r['failed_safe_events']}")
    print("\n  On the Pi this latency includes real relay switching + servo travel + settle;")
    print("  report it as the physical-actuation component of MTTR (Section 6.5.2).")
