# Run it on your PC (2 commands)

1) Install the three extra packages (you already have the rest):
```
pip install paho-mqtt amqtt pyyaml
```

2) Run the end-to-end demo:
```
python experiments/e2e_demo.py
```
(or open `experiments/e2e_demo.py` in VS Code and press Run — it works from any folder)

Expected: a real MQTT broker starts, the governance plane gates 240 real intents,
the actuator applies the admitted ones, and the persistent audit chain is verified
and then shown to detect tampering. Finally it demonstrates the G2 policy gap live
and closes it with the corrected policy.

To run the tests: `pytest -q`  (9 tests)
