# Hardware-in-the-loop rollback harness — wiring & run guide

`hardware_rollback_harness.py` runs the same rollback FSM as the software twin but
drives a **real actuator** and measures **physical** recovery latency — the
component the paper defers to hardware (Section 6.5.2).

## Parts
- Raspberry Pi (any model with GPIO; tested target: Pi 4 / Pi 5)
- 1-channel relay board (models a discrete HVAC contactor / valve open-close)
- Optional hobby servo (models a continuous valve position 0–100)

## Wiring (BCM numbering)
| Signal      | GPIO (BCM) | Header pin | Notes                          |
|-------------|-----------:|-----------:|--------------------------------|
| Relay IN    | GPIO17     | 11         | via relay board, not direct    |
| Relay VCC   | 5V         | 2          |                                |
| Relay GND   | GND        | 6          |                                |
| Servo sig   | GPIO18     | 12         | hardware PWM pin               |
| Servo V+    | 5V         | 4          | external 5V if servo is large  |
| Servo GND   | GND        | 9          | common ground with the Pi      |

> Never drive an inductive/AC load directly from a GPIO pin — always use the relay
> board (opto-isolated recommended).

## Install (on the Pi)
```bash
pip install RPi.GPIO gpiozero pyyaml
```

## Run
```bash
# Anywhere (no hardware) — logic smoke test:
python3 hardware_rollback_harness.py --n 2000

# On the Pi with the real relay/servo:
python3 hardware_rollback_harness.py --gpio --n 100
```
If `--gpio` is set but the Pi libraries are missing, it prints a notice and falls
back to the MockDriver, so the script never hard-fails.

## What to report from the Pi
- `physical restore VERIFIED: N/N = 100.0%` — the state machine actually returns the
  actuator to the checkpointed position.
- `MEASURED physical recovery latency: median=… ms` — this is the **physical**
  actuation component of MTTR (relay switch + servo travel + settle). Report it
  alongside the software control-path latency from `rollback_demo.py`.
- `FAILED-SAFE transitions` — count of irreversible-action events that correctly
  halted the device.
