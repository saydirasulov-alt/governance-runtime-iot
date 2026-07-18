"""Actuator adapter. Subscribes to admitted commands and applies them.

On a workstation this keeps state in memory (simulator). On a Raspberry Pi the same
class is subclassed by gpio_actuator.py to drive a relay/servo. The governance plane
does not know or care which one is running: same MQTT contract.
"""
import json
from runtime.mqtt_compat import make_client


class ActuatorSim:
    def __init__(self, broker="127.0.0.1", port=1883, device="room-1"):
        self.device = device
        self.state = {"setpoint": 22.0}
        self.applied = []
        self.c = make_client(f"act-{device}")
        self.c.on_connect = lambda c, u, f, rc: c.subscribe(f"actuators/{device}/cmd", qos=1)
        self.c.on_message = self._on
        self.c.connect(broker, port, keepalive=30)
        self.c.loop_start()

    def _on(self, c, u, msg):
        cmd = json.loads(msg.payload.decode())
        self.apply(cmd)
        self.applied.append(cmd)

    def apply(self, cmd):
        if cmd.get("setpoint") is not None:
            self.state["setpoint"] = cmd["setpoint"]

    def stop(self):
        self.c.loop_stop(); self.c.disconnect()
