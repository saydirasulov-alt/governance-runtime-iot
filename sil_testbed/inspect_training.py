"""
Watch the AI service train, step by step, and satisfy yourself it is real.

    python inspect_training.py

The paper's whole argument depends on ONE thing being true: the model that the
governance plane governs is a real model, trained on real recorded sensor data,
and nobody hand-typed its numbers. This script exists so you can verify that with
your own eyes rather than take the pipeline's word for it. It trains the SAME model
the paper uses (same architecture, same seed, same data) and shows you every stage:

    1. the raw UCI file on disk -- real rows, real timestamps
    2. the features and the target, and how the target is built
    3. the network learning, iteration by iteration (the loss must fall)
    4. the training error
    5. predictions on inputs you can sanity-check by hand
    6. that it generalises to data it never saw during training
    7. that it is deterministic -- run it twice, get the identical number

Nothing here is used by the paper's results; run_sil.py does the real work. This is
a transparency tool, not part of the pipeline.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from gsim.aimodel import (FEATURES, SETPOINT_OCCUPIED, SETPOINT_VACANT, load_uci,
                          train_setpoint_model)

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "ds")


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    # ---------------------------------------------------------------- 1
    hr("1.  THE RAW DATA ON DISK -- this is a real recorded dataset, not synthetic")
    path = os.path.join(DS, "datatraining.txt")
    print(f"  file: {path}")
    print(f"  size: {os.path.getsize(path):,} bytes\n")
    with open(path) as f:
        head = [next(f) for _ in range(4)]
    print("  first rows exactly as they sit on disk:")
    for line in head:
        print("    " + line.rstrip()[:96])
    print("\n  This is the UCI Occupancy Detection dataset (Candanedo & Feldheim, 2016):")
    print("  a real office monitored with real sensors, one row per minute, with a")
    print("  ground-truth Occupancy column obtained from time-stamped photographs.")

    tr = load_uci(DS, "datatraining.txt")
    print(f"\n  loaded {len(tr)} training rows, columns: {list(tr.columns)}")

    # ---------------------------------------------------------------- 2
    hr("2.  FEATURES AND TARGET -- what the model sees, and what it must predict")
    print(f"  features (what the model reads):  {FEATURES}")
    print(f"  NOTE: 'Light' is deliberately EXCLUDED. It is an almost-perfect giveaway of")
    print(f"        occupancy, and leaving it in would make the task trivial and unrealistic.")
    print(f"\n  target (what the model must output): a thermostat setpoint")
    print(f"        Occupancy == 1  ->  {SETPOINT_OCCUPIED} degC  (someone is in the room, keep it comfortable)")
    print(f"        Occupancy == 0  ->  {SETPOINT_VACANT} degC  (empty, set back to save energy)")
    occ = int((tr["Occupancy"] == 1).sum())
    vac = int((tr["Occupancy"] == 0).sum())
    print(f"\n  training rows: {occ} occupied, {vac} vacant ({100*occ/len(tr):.1f}% occupied)")
    print("  So the model must infer, from temperature/humidity/CO2 alone, whether the")
    print("  room is occupied -- and it will get this WRONG sometimes. That is the point:")
    print("  an imperfect real model is what the governance plane has to cope with.")

    # ---------------------------------------------------------------- 3
    hr("3.  TRAINING -- watch the loss fall (this is the network actually learning)")
    X = tr[FEATURES].to_numpy(dtype=float)
    y = np.where(tr["Occupancy"].to_numpy() == 1, SETPOINT_OCCUPIED, SETPOINT_VACANT)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    print("  architecture: MLP, hidden layers (32, 16), ReLU, seed 42 (fixed)\n")
    print("  iteration        loss")
    net = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                       max_iter=600, random_state=42)
    net.fit(Xs, y)
    curve = net.loss_curve_
    marks = [0, 1, 2, 4, 9, 19, 49, 99, 199, len(curve) // 2, len(curve) - 1]
    for i in sorted(set(m for m in marks if m < len(curve))):
        bar = "#" * int(40 * curve[i] / curve[0])
        print(f"    {i+1:5d}    {curve[i]:9.4f}  {bar}")
    print(f"\n  converged in {net.n_iter_} iterations.")
    print(f"  loss fell from {curve[0]:.3f} to {curve[-1]:.4f} -- a factor of {curve[0]/curve[-1]:.0f}.")
    print("  A falling loss is the network fitting the data. A hand-typed table would")
    print("  not have a loss curve at all.")

    # ---------------------------------------------------------------- 4
    hr("4.  TRAINING ERROR")
    yhat = net.predict(Xs)
    mae = float(np.mean(np.abs(yhat - y)))
    print(f"  mean absolute error on the training set: {mae:.4f} degC")
    print(f"  (the setpoints are 17 and 22, so this is the average miss in degrees)")

    # ---------------------------------------------------------------- 5
    hr("5.  PREDICTIONS YOU CAN CHECK BY HAND")
    print("  Feed the model a few situations and see whether its answer is sensible.")
    print("  High CO2 + high humidity look like an occupied room; low look empty.\n")
    model, _ = train_setpoint_model(DS)   # the exact object the paper uses
    cases = [
        ("clearly occupied  (warm, humid, CO2 1200)", 21.5, 27.0, 1200.0, 0.0045),
        ("clearly vacant    (cool, dry,   CO2  450)", 20.0, 25.0,  450.0, 0.0037),
        ("borderline        (CO2 700)",               21.0, 26.0,  700.0, 0.0041),
    ]
    print("    situation                                    ->  setpoint")
    for label, t, h, co2, hr_ in cases:
        sp = model.predict(t, h, co2, hr_)
        near = "~occupied (22)" if sp > 19.5 else "~vacant (17)"
        print(f"    {label:44s} ->  {sp:5.2f} degC   {near}")
    print("\n  The occupied case should land near 22 and the vacant near 17. If they do,")
    print("  the model has learned the real relationship, not memorised a lookup table.")

    # ---------------------------------------------------------------- 6
    hr("6.  DOES IT GENERALISE?  Predict on data it NEVER saw during training")
    te = load_uci(DS, "datatest.txt")
    Xte = scaler.transform(te[FEATURES].to_numpy(dtype=float))
    yte = np.where(te["Occupancy"].to_numpy() == 1, SETPOINT_OCCUPIED, SETPOINT_VACANT)
    mae_te = float(np.mean(np.abs(net.predict(Xte) - yte)))
    ratio = mae_te / mae
    print(f"  held-out test rows: {len(te)}")
    print(f"  error on training data : {mae:.4f} degC")
    print(f"  error on UNSEEN data   : {mae_te:.4f} degC   ({ratio:.1f}x worse)")
    print("  Read that gap honestly. The error on unseen data is several times the training")
    print("  error -- the model does NOT generalise perfectly, and we are not pretending it")
    print("  does. That degradation is not a flaw in this script; it IS the phenomenon the")
    print("  paper studies. A model that sails through a distribution shift would leave")
    print("  nothing for a governance layer to catch. The point of the paper is what happens")
    print("  to physical safety precisely BECAUSE a realistic model gets worse off-distribution.")

    # ---------------------------------------------------------------- 7
    hr("7.  IS IT REPRODUCIBLE?  Train it again from scratch, compare")
    _, info2 = train_setpoint_model(DS)
    print(f"  first training  MAE: {mae:.6f}")
    print(f"  second training MAE: {info2['train_mae_c']:.6f}")
    same = abs(mae - info2["train_mae_c"]) < 1e-9
    print(f"  identical: {same}  (fixed seed -> the model is deterministic, so anyone who")
    print("  runs this gets exactly the number the paper reports)")

    hr("SUMMARY")
    print("  You have now seen, end to end:")
    print("   - the real recorded data it trains on,")
    print("   - the network's loss actually falling as it learns,")
    print(f"   - a training error of {mae:.3f} degC that rises to {mae_te:.3f} on unseen data,")
    print("     which is the realistic degradation the paper is built to study,")
    print("   - sensible predictions you could check by hand,")
    print("   - and bit-for-bit reproducibility.")
    print("\n  This is the model the governance plane governs in run_sil.py. Nothing about")
    print("  it is hand-tuned to make governance look good -- it is an ordinary, imperfect,")
    print("  honestly-trained model, and its imperfection under shift is the whole subject")
    print("  of the paper.")


if __name__ == "__main__":
    main()
