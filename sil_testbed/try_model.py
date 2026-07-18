"""
Train the model, then interrogate it yourself.

    python try_model.py

This trains the exact model the paper uses, then drops you into a prompt where you
type sensor readings and watch what setpoint the model returns. There is nothing to
take on trust: you pick the inputs, you see the output, and you decide whether it is
behaving like a model that learned occupancy from the data or like a hand-tuned fake.

Type four numbers separated by spaces:  Temperature Humidity CO2 HumidityRatio
  e.g.   21.5 27 1200 0.0045     (looks occupied -> expect a warm setpoint, ~22)
         20   25 450  0.0037     (looks empty    -> expect a cool setpoint, ~17)
Type  q  to quit.

Reference ranges from the real training data (so your inputs are realistic):
  Temperature   ~19-24 degC
  Humidity      ~16-40 %
  CO2           ~400-2000 ppm   (higher = more likely someone is breathing in the room)
  HumidityRatio ~0.003-0.006
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from gsim.aimodel import SETPOINT_OCCUPIED, SETPOINT_VACANT, load_uci, train_setpoint_model

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "ds")


def main():
    print("training the model (a few seconds)...")
    model, info = train_setpoint_model(DS)
    print(f"  trained on {info['n_train']} real rows, training error {info['train_mae_c']:.3f} degC")
    print(f"  setpoints it was taught: occupied {SETPOINT_OCCUPIED}, vacant {SETPOINT_VACANT}\n")

    # show the real data ranges so the user's guesses are grounded
    tr = load_uci(DS, "datatraining.txt")
    print("  real training-data ranges (min / mean / max):")
    for col in ("Temperature", "Humidity", "CO2", "HumidityRatio"):
        c = tr[col]
        w = 4 if col == "HumidityRatio" else 1
        print(f"    {col:14s} {c.min():8.{w}f} / {c.mean():8.{w}f} / {c.max():8.{w}f}")

    print("\n  Type: Temperature Humidity CO2 HumidityRatio   (or 'q' to quit)")
    print("  Example: 21.5 27 1200 0.0045\n")

    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye")
            return
        if raw.lower() in ("q", "quit", "exit", ""):
            print("  bye")
            return
        parts = raw.split()
        if len(parts) != 4:
            print("    need exactly 4 numbers: Temperature Humidity CO2 HumidityRatio")
            continue
        try:
            t, h, co2, hr = (float(p) for p in parts)
        except ValueError:
            print("    those weren't all numbers -- try again")
            continue

        sp = model.predict(t, h, co2, hr)
        verdict = "OCCUPIED-ish (near 22)" if sp > 19.5 else "VACANT-ish (near 17)"
        # a little context: how the model reads this
        note = ""
        if co2 >= 800:
            note = "  (high CO2 -> reads as occupied)"
        elif co2 <= 500:
            note = "  (low CO2 -> reads as empty)"
        print(f"    setpoint = {sp:6.2f} degC   -> {verdict}{note}\n")


if __name__ == "__main__":
    main()
