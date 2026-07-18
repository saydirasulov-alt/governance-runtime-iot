import argparse, time
from .actuator_sim import ActuatorSim
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="127.0.0.1"); ap.add_argument("--device", default="room-1")
    a = ap.parse_args()
    act = ActuatorSim(broker=a.broker, device=a.device)
    print(f"[actuator] {a.device} listening", flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: act.stop()
