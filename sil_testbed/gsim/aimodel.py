"""
The AI service under governance.

A real model, trained on a real dataset, making a real control decision. Nothing
here is hand-injected, and no failure is planted: the model's errors under
distribution shift are its own.

Dataset
    UCI Occupancy Detection (Candanedo & Feldheim, 2016).
      datatraining.txt  training regime
      datatest.txt      same regime as training (in-distribution)
      datatest2.txt     a different occupancy/ventilation regime (distribution shift)

Features
    Temperature, Humidity, CO2, HumidityRatio.
    Light is deliberately EXCLUDED. Light is a near-perfect proxy for occupancy in
    this dataset, so including it makes the task trivial and the model never fails.
    A model that never fails cannot be used to study what governance does when a
    model fails. Excluding Light is therefore not a handicap we imposed to make a
    point; it is what makes the study possible at all, and it is stated openly.

Target
    The control target, not the label: 22 degC when occupied, 17 degC when vacant.
    The model is a setpoint regressor, so its errors arrive at the governance plane
    in exactly the form a real HVAC AI service would produce -- a number, with no
    confidence, no flag, and no indication that anything is wrong.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = ["Temperature", "Humidity", "CO2", "HumidityRatio"]
SETPOINT_OCCUPIED = 22.0
SETPOINT_VACANT = 17.0


def load_uci(ds_dir: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ds_dir, name))
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


@dataclass
class SetpointModel:
    """MLP setpoint regressor. This is the component the governance plane governs."""

    net: MLPRegressor
    scaler: StandardScaler
    co2_vacant_p95: float     # 95th pct of CO2 among VACANT *training* samples

    def predict(self, temperature: float, humidity: float, co2: float,
                humidity_ratio: float) -> float:
        x = np.array([[temperature, humidity, co2, humidity_ratio]], dtype=float)
        return float(self.net.predict(self.scaler.transform(x))[0])


def train_setpoint_model(ds_dir: str, seed: int = 42) -> tuple[SetpointModel, dict]:
    tr = load_uci(ds_dir, "datatraining.txt")
    X = tr[FEATURES].to_numpy(dtype=float)
    y = np.where(tr["Occupancy"].to_numpy() == 1, SETPOINT_OCCUPIED, SETPOINT_VACANT)

    scaler = StandardScaler().fit(X)
    net = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=600,
        random_state=seed,
    ).fit(scaler.transform(X), y)

    # The CO2 threshold used by the CORRECTED gate. It is derived ONLY from the
    # training split, never from the evaluation splits. Deriving it from the test
    # data would make the corrected gate look better than it is, which is exactly
    # the kind of circularity this study exists to avoid.
    vac = tr.loc[tr["Occupancy"] == 0, "CO2"].to_numpy(dtype=float)
    co2_p95 = float(np.percentile(vac, 95))

    yhat = net.predict(scaler.transform(X))
    info = {
        "n_train": int(len(tr)),
        "train_mae_c": float(np.mean(np.abs(yhat - y))),
        "co2_vacant_p95_ppm": co2_p95,
    }
    return SetpointModel(net, scaler, co2_p95), info


def sensor_trace(ds_dir: str, name: str) -> list[tuple[float, float, float, float, int]]:
    """
    (Temperature, Humidity, CO2, HumidityRatio, Occupancy) per minute.

    The first four are the real recorded sensor stream the AI service consumes. The
    fifth is the ground-truth occupancy, which neither the AI service nor the
    governance gates ever see: it goes only to the plant, as an internal heat gain,
    and to the independent safety oracle. Keeping it strictly out of the governed
    path is what makes the safety measurements non-circular.
    """
    df = load_uci(ds_dir, name)
    return list(zip(
        df["Temperature"].astype(float),
        df["Humidity"].astype(float),
        df["CO2"].astype(float),
        df["HumidityRatio"].astype(float),
        df["Occupancy"].astype(int),
    ))
