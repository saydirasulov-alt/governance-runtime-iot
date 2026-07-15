"""A REAL AI inference service: publishes control intents over MQTT.

Loads the model trained on the real UCI occupancy dataset and publishes the setpoint it
recommends for each live sensor reading. Its genuine errors -- not injected faults -- are
what the governance plane must gate.
"""
import json, time, uuid
from runtime.mqtt_compat import make_client


class AIService:
    def __init__(self, broker="127.0.0.1", port=1883, device="room-1"):
        self.device = device
        self.c = make_client(f"ai-{device}")
        self.c.connect(broker, port, keepalive=30)
        self.c.loop_start()

    _MISSING = object()

    def publish_intent(self, setpoint, source="ai_service_1", action="set_temperature",
                       timestamp=_MISSING, intent_id=None):
        ts = time.time() if timestamp is AIService._MISSING else timestamp
        intent = {"intent_id": intent_id or str(uuid.uuid4())[:8],
                  "device_id": self.device, "room_id": self.device,
                  "setpoint": setpoint, "action": action,
                  "timestamp": ts,
                  "source": source, "sequence_id": int(time.time() * 1000) % 100000}
        self.c.publish(f"intents/{self.device}", json.dumps(intent), qos=1)
        return intent

    def stop(self):
        self.c.loop_stop(); self.c.disconnect()
