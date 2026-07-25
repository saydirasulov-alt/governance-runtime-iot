"""
What does the learned controller buy, and what does it cost?

Deterministic rule-based setpoint controllers are compared against the learned MLP
under IDENTICAL governance (shipped gate, rollback, safe_state fallback), on the
same recorded sensor traces, scored by the same independent safety oracle.

Controllers
    mlp        the paper's MLP setpoint regressor (4 features)
    rule_co2   deterministic thermostat: CO2 > p95(vacant CO2, training split)
               -> occupied setpoint 22 C, else vacant setback 17 C.
               Same threshold provenance as the corrected gate; single feature.
    const22    constant 22 C, no setback (comfort-maximal, energy-maximal reference)

Metrics (per split: in-distribution datatest, shift datatest2)
    unsafe exposure (C*min)   comfort/safety risk, oracle-scored
    peak excursion (C)
    setback fraction (%)      share of minutes commanded <= 17.5 C (energy-saving action)
    heating proxy (kWh)       steady-state heating power to hold the commanded setpoint:
                              q(t) = max(0, (sp - T_out)/R - (P_base + n_occ*P_person)),
                              integrated over commanded minutes. A proxy, not a meter.

Run:  python experiments/rule_vs_ai_baseline.py
Writes results/rule_vs_ai.json and prints the table.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIL = os.path.dirname(HERE)
sys.path.insert(0, SIL)

from gsim.aimodel import sensor_trace, train_setpoint_model, SETPOINT_OCCUPIED, SETPOINT_VACANT
from gsim.loop import run_closed_loop
from gsim.plant import PlantParams
from gsim import gates as G

DS, RES = os.path.join(SIL, "ds"), os.path.join(SIL, "results")
os.makedirs(RES, exist_ok=True)


class RuleCO2Thermostat:
    """Deterministic single-rule controller. No learning anywhere."""
    def __init__(self, co2_threshold_ppm: float):
        self.thr = co2_threshold_ppm
    def predict(self, temperature, humidity, co2, humidity_ratio) -> float:
        return SETPOINT_OCCUPIED if co2 > self.thr else SETPOINT_VACANT


class Const22:
    def predict(self, temperature, humidity, co2, humidity_ratio) -> float:
        return SETPOINT_OCCUPIED


def heating_kwh(res, p: PlantParams) -> float:
    """Steady-state heating power (W) needed to hold each commanded setpoint, integrated."""
    kwh = 0.0
    for sp, occ in zip(res.trace_sp, res.trace_occ):
        gains = p.P_base + occ * p.P_person
        q = max(0.0, (sp - p.T_out) / p.R_th - gains)     # W
        kwh += q * 60.0 / 3.6e6                           # one minute per sample
    return kwh


def setback_frac(res) -> float:
    n = len(res.trace_sp)
    return 100.0 * sum(1 for s in res.trace_sp if s <= 17.5) / max(1, n)


def main():
    model, info = train_setpoint_model(DS, seed=42)
    controllers = {
        "mlp": model,
        "rule_co2": RuleCO2Thermostat(model.co2_vacant_p95),
        "const22": Const22(),
    }
    shipped = G.shipped_policy()
    p = PlantParams()
    out = []
    print(f"CO2 threshold (train vacant p95): {model.co2_vacant_p95:.0f} ppm")
    for regime, fname in [("in-distribution", "datatest.txt"), ("shift", "datatest2.txt")]:
        tr = sensor_trace(DS, fname)
        for cname, ctrl in controllers.items():
            r = run_closed_loop(ctrl, tr, shipped, arm=f"{cname}/{regime}",
                                enable_rollback=True, fallback="safe_state",
                                seed=7, keep_trace=True)
            rec = dict(regime=regime, controller=cname,
                       unsafe_exposure_c_min=round(r.unsafe_exposure_c_min, 1),
                       peak_excursion_c=round(r.peak_excursion_c, 2),
                       setback_pct=round(setback_frac(r), 1),
                       heating_kwh=round(heating_kwh(r, p), 1),
                       rollbacks=r.rollbacks, admitted=r.admitted, rejected=r.rejected)
            out.append(rec)
            print(rec)
    with open(os.path.join(RES, "rule_vs_ai.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/rule_vs_ai.json")


if __name__ == "__main__":
    main()
