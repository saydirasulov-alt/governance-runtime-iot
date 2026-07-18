"""paho-mqtt 1.x / 2.x compatibility.

paho-mqtt 2.0 made CallbackAPIVersion a required first argument. This shim lets the
same code run on both, so the service works on whatever the user already has.
"""
import paho.mqtt.client as mqtt


def make_client(client_id: str):
    try:                                  # paho-mqtt >= 2.0
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                           client_id=client_id, protocol=mqtt.MQTTv311)
    except AttributeError:                # paho-mqtt 1.x
        return mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
