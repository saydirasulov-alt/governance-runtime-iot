"""Raspberry Pi actuator: the SAME MQTT contract as the simulator, driving real hardware.

Relay on GPIO17 (discrete valve / contactor), servo on GPIO18 (continuous valve position).
Run on the Pi:  python -m devices.gpio_actuator --device room-1 --broker <broker-ip>
The governance plane is unchanged: it publishes to actuators/<device>/cmd either way.
"""
import argparse
from .actuator_sim import ActuatorSim


class GpioActuator(ActuatorSim):
    def __init__(self, relay_pin=17, servo_pin=18, **kw):
        import RPi.GPIO as GPIO
        from gpiozero import AngularServo
        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM); GPIO.setup(relay_pin, GPIO.OUT)
        self.relay_pin = relay_pin
        self.servo = AngularServo(servo_pin, min_angle=0, max_angle=180)
        super().__init__(**kw)

    def apply(self, cmd):
        super().apply(cmd)
        sp = self.state["setpoint"]
        self.GPIO.output(self.relay_pin, self.GPIO.HIGH if sp >= 20 else self.GPIO.LOW)
        self.servo.angle = max(0, min(100, (sp - 15) / 15 * 100)) * 1.8


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="127.0.0.1"); ap.add_argument("--device", default="room-1")
    a = ap.parse_args()
    act = GpioActuator(broker=a.broker, device=a.device)
    print(f"GPIO actuator running for {a.device}; Ctrl-C to stop")
    try:
        import time
        while True: time.sleep(1)
    except KeyboardInterrupt:
        act.stop()
