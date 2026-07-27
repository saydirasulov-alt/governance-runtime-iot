"""
verify_paper_numbers.py -- regenerate every host-independent number reported in the
manuscript and check it against the expected value. One command, one scoreboard.

    python verify_paper_numbers.py            fast set (~3-4 min): model, E1, E4,
                                              Table 16, E6, E3, E7, floor, real-AI,
                                              gap finder
    python verify_paper_numbers.py --sweep    + E8 plant sweep      (~7-8 min)
    python verify_paper_numbers.py --robust   + E9 robustness audit (~14 min)
    python verify_paper_numbers.py --all      everything            (~25 min)

Host-dependent latencies (0.44 ms MQTT decision, 0.94 ms OPA HTTP, gate microseconds,
Tier-1 backend milliseconds) are intentionally NOT pass/fail checked here: they are
measured quantities of a given machine, reproduced by run_governance_demo.py,
experiments/opa_http_latency.py and reproduce_all.py.

Exit code = number of FAILed checks.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIL = os.path.join(HERE, "sil_testbed")
sys.path.insert(0, SIL)

from gsim import gates as G
from gsim.aimodel import (SETPOINT_OCCUPIED, SETPOINT_VACANT, load_uci,
                          sensor_trace, train_setpoint_model)
from gsim.loop import run_closed_loop
from gsim.plant import PlantParams

DS = os.path.join(SIL, "ds")
OPEN = {"name": "UNGOVERNED", "gates": []}

PASS, FAIL, checks = 0, 0, []
def check(name, got, want, tol=0.15):
    global PASS, FAIL
    ok = abs(float(got) - float(want)) <= tol
    checks.append((ok, name, want, got))
    PASS, FAIL = PASS + ok, FAIL + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:58s} want {want:>9} got {round(float(got),3):>9}")
def check_int(name, got, want): check(name, got, want, tol=0.0)


class Perfect:
    """Ground-truth-occupancy controller: the floor of Sec. 10.1."""
    def __init__(self, trace): self.t, self.i = trace, 0
    def predict(self, *_):
        occ = self.t[min(self.i, len(self.t) - 1)][4]; self.i += 1
        return SETPOINT_OCCUPIED if occ else SETPOINT_VACANT

class Const22:
    def predict(self, *_): return SETPOINT_OCCUPIED

class RuleCO2:
    def __init__(self, thr): self.thr = thr
    def predict(self, t, h, co2, hr):
        return SETPOINT_OCCUPIED if co2 > self.thr else SETPOINT_VACANT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all: a.sweep = a.robust = True

    print("=" * 78); print("0. MODEL / TRAINING  (Table 10)"); print("=" * 78)
    model, info = train_setpoint_model(DS, seed=42)
    tr_df = load_uci(DS, "datatraining.txt")
    check_int("training rows", len(tr_df), 8143)
    check_int("occupied rows", int((tr_df["Occupancy"] == 1).sum()), 1729)
    check_int("vacant rows", int((tr_df["Occupancy"] == 0).sum()), 6414)
    check_int("training iterations", model.net.n_iter_, 327)
    n_par = sum(w.size for w in model.net.coefs_) + sum(b.size for b in model.net.intercepts_)
    check_int("learned parameters", n_par, 705)
    check("train MAE (degC)", info["train_mae_c"], 0.402, 0.02)
    check("CO2 vacant p95 (ppm)", model.co2_vacant_p95, 755, 1.0)
    check("loss curve, first", model.net.loss_curve_[0], 178.7, 2.0)
    check("loss curve, last", model.net.loss_curve_[-1], 0.34, 0.05)
    import numpy as np
    for split, want in (("datatest.txt", 1.69), ("datatest2.txt", 3.32)):
        df = load_uci(DS, split)
        X = df[["Temperature", "Humidity", "CO2", "HumidityRatio"]].to_numpy(float)
        y = np.where(df["Occupancy"].to_numpy() == 1, SETPOINT_OCCUPIED, SETPOINT_VACANT)
        mae = float(np.mean(np.abs(model.net.predict(model.scaler.transform(X)) - y)))
        check(f"MAE on {split} (degC)", mae, want, 0.02)
    yh = model.net.predict(model.scaler.transform(
        tr_df[["Temperature", "Humidity", "CO2", "HumidityRatio"]].to_numpy(float)))
    occ_pred = yh >= 19.5; occ_true = tr_df["Occupancy"].to_numpy() == 1
    check("occupancy recovery on train (%)", 100 * float((occ_pred == occ_true).mean()), 96.4, 0.15)

    shipped = G.shipped_policy()
    corrected = G.corrected_policy(info["co2_vacant_p95_ppm"])
    oracle = G.oracle_policy()
    tr_sh = sensor_trace(DS, "datatest2.txt")
    tr_in = sensor_trace(DS, "datatest.txt")

    def run(ctrl, tr, pol, rb=True, fb="safe_state", budget="default", irrev=None):
        kw = dict(enable_rollback=rb, fallback=fb, seed=7, keep_trace=True)
        if budget != "default": kw["rollback_budget"] = budget
        if irrev is not None: kw["irreversible_above_c"] = irrev
        return run_closed_loop(ctrl, tr, pol, arm="verify", **kw)

    print("=" * 78); print("E1. CLOSED-LOOP UNDER SHIFT  (Table 14)"); print("=" * 78)
    exp = [("ungoverned", OPEN, False, 1185.3, 3.75, 0),
           ("shipped no-rb", shipped, False, 954.8, 3.75, 0),
           ("shipped +rb", shipped, True, 953.2, 3.81, 9),
           ("corrected +rb", corrected, True, 727.7, 3.81, 5),
           ("oracle +rb", oracle, True, 1.2, 0.16, 1)]
    r_or = None
    for tag, pol, rb, e, pk, nrb in exp:
        r = run(model, tr_sh, pol, rb=rb)
        if tag == "oracle +rb": r_or = r
        check(f"E1 {tag} exposure", r.unsafe_exposure_c_min, e)
        check(f"E1 {tag} peak", r.peak_excursion_c, pk, 0.02)
        check_int(f"E1 {tag} rollbacks", r.rollbacks, nrb)
    check("oracle strict exposure (floor table)", r_or.strict_exposure_c_min, 130.1, 0.2)

    print("=" * 78); print("E4. IN-DISTRIBUTION  (Sec. 8.4)"); print("=" * 78)
    for tag, pol, rb, e in [("ungoverned", OPEN, False, 222.4),
                            ("shipped no-rb", shipped, False, 222.4),
                            ("shipped +rb", shipped, True, 221.5),
                            ("corrected +rb", corrected, True, 0.5),
                            ("oracle +rb", oracle, True, 0.5)]:
        check(f"E4 {tag} exposure", run(model, tr_in, pol, rb=rb).unsafe_exposure_c_min, e)

    print("=" * 78); print("T16. LEARNED vs RULE vs CONST  (Sec. 8.5)"); print("=" * 78)
    p = PlantParams()
    def kwh(r):
        s = 0.0
        for sp, occ in zip(r.trace_sp, r.trace_occ):
            s += max(0.0, (sp - p.T_out) / p.R_th - (p.P_base + occ * p.P_person)) * 60 / 3.6e6
        return s
    ctrls = {"mlp": model, "rule_co2": RuleCO2(model.co2_vacant_p95), "const22": Const22()}
    t16 = {("in", "mlp"): (221.5, 27.7, None), ("in", "rule_co2"): (200.4, 23.4, None),
           ("in", "const22"): (0.0, 38.8, None), ("sh", "mlp"): (953.2, 115.6, 306),
           ("sh", "rule_co2"): (1427.7, 83.6, 0), ("sh", "const22"): (0.0, 144.7, 0)}
    for reg, tr in (("in", tr_in), ("sh", tr_sh)):
        for cn in ctrls:
            r = run(ctrls[cn], tr, shipped)
            e, kw_, rej = t16[(reg, cn)]
            check(f"T16 {reg}/{cn} exposure", r.unsafe_exposure_c_min, e)
            check(f"T16 {reg}/{cn} heating kWh", kwh(r), kw_)
            if rej is not None: check_int(f"T16 {reg}/{cn} rejected", r.rejected, rej)

    print("=" * 78); print("E6. REJECTION SEMANTICS  (Table 17)"); print("=" * 78)
    for pol, pn, vals in [(oracle, "oracle", {"hold": 924.8, "checkpoint": 918.7, "safe_state": 1.2}),
                          (corrected, "corrected", {"hold": 734.0, "checkpoint": 727.9, "safe_state": 727.7})]:
        for fb, e in vals.items():
            check(f"E6 {pn}/{fb} exposure", run(model, tr_sh, pol, fb=fb).unsafe_exposure_c_min, e)

    print("=" * 78); print("E3. IRREVERSIBLE ACTUATION  (Sec. 8.7)"); print("=" * 78)
    r = run(model, tr_sh, shipped, irrev=28.0)
    check("E3 shipped/latch exposure", r.unsafe_exposure_c_min, 4951.7, 0.2)
    check_int("E3 shipped/latch FAILED_SAFE", int(r.failed_safe), 1)
    check_int("E3 shipped/latch minute", r.failed_safe_minute, 1050)
    check("E3 shipped/latch availability (%)", 100 * r.governed_minutes / r.minutes, 10.8, 0.15)
    check("E3 corrected/latch exposure", run(model, tr_sh, corrected, irrev=28.0).unsafe_exposure_c_min, 727.7)

    print("=" * 78); print("E7. ROLLBACK BUDGET  (Sec. 8.8)"); print("=" * 78)
    for b in (10, 5, 3):
        r = run(model, tr_sh, shipped, budget=b)
        check(f"E7 budget={b} exposure", r.unsafe_exposure_c_min, 953.2)
        check_int(f"E7 budget={b} rollbacks", r.rollbacks, 9)

    print("=" * 78); print("FLOOR  (Table 20)"); print("=" * 78)
    rp = run(Perfect(tr_sh), tr_sh, OPEN, rb=False)
    check("floor perfect grace", rp.unsafe_exposure_c_min, 0.0, 0.05)
    check("floor perfect strict", rp.strict_exposure_c_min, 337.0, 0.5)
    rc = run(Const22(), tr_sh, OPEN, rb=False)
    check("floor const22 grace", rc.unsafe_exposure_c_min, 0.0, 0.05)
    check("floor const22 strict", rc.strict_exposure_c_min, 0.0, 0.05)

    print("=" * 78); print("REAL-AI RESIDUALS + AUC  (Table 11, via real_ai_governance.py)"); print("=" * 78)
    out = subprocess.run([sys.executable, "real_ai_governance.py"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout
    import re
    rng_rows, fn_rows = [], []
    for ln in out.splitlines():
        if ln.strip().startswith(("in-distribution (datatest)", "distribution shift (datatest2)")):
            if "[" in ln:
                m = re.search(r"\[\s*([\d.]+),\s*([\d.]+)\]\s*([\d.]+)%", ln)
                rng_rows.append(tuple(float(x) for x in m.groups()))
            elif "%" in ln:
                nums = re.findall(r"([\d.]+)%", ln) + [ln.split()[-1]]
                fn_rows.append(tuple(float(x) for x in nums))
    (ilo, ihi, ioob), (slo, shi, soob) = rng_rows
    check("realai in setpoint lo", ilo, 16.7, 0.05); check("realai in setpoint hi", ihi, 24.5, 0.05)
    check("realai in out-of-band (%)", ioob, 0.0, 0.05)
    check("realai sh setpoint lo", slo, 13.8, 0.05); check("realai sh setpoint hi", shi, 40.9, 0.05)
    check("realai sh out-of-band (%)", soob, 7.1, 0.1)
    (ibl, ico, ifp, iauc), (sbl, sco, sfp, sauc) = fn_rows
    check("realai in blind residual (%)", ibl, 100.0, 0.1)
    check("realai in +CO2 residual (%)", ico, 15.7, 0.1)
    check("realai in +CO2 FP cost (%)", ifp, 2.7, 0.1)
    check("realai in gate AUC", iauc, 0.553, 0.003)
    check("realai sh blind residual (%)", sbl, 48.0, 0.1)
    check("realai sh +CO2 residual (%)", sco, 44.2, 0.1)
    check("realai sh gate AUC", sauc, 0.863, 0.003)

    print("=" * 78); print("GAP FINDER  (Table 9 / Fig 4)"); print("=" * 78)
    gf = os.path.join(HERE, "gap_finder")
    subprocess.run([sys.executable, "policy_gap_experiment.py"], cwd=gf,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    md = open(os.path.join(gf, "results", "gap_finder_results.md")).read()
    def md_val(row):
        for ln in md.splitlines():
            if ln.startswith(f"| {row} "): return float(ln.split("|")[2])
        raise KeyError(row)
    check("gap: manual injection recall", md_val("manual injection (1 hand-picked)"), 0.17, 0.005)
    check("gap: no-oracle recall", md_val("policy-consistency testing (no oracle)"), 0.00, 0.001)
    check("gap: random+oracle recall", md_val("random + physical oracle"), 1.00, 0.001)
    check("gap: coverage+oracle recall", md_val("OURS: coverage-guided + oracle"), 1.00, 0.001)
    check("gap: SHIPPED policy gap (%)", md_val("SHIPPED policy (no mutation)"), 20.2, 0.35)
    check("gap: drop G3/queue_depth (%)", md_val("drop G3/queue_depth"), 48.6, 0.45)
    check("gap: drop G1/setpoint (%)", md_val("drop G1/setpoint"), 30.6, 0.35)

    if a.sweep:
        print("=" * 78); print("E8. PLANT SWEEP  (Table 18, ~7-8 min)"); print("=" * 78)
        subprocess.run([sys.executable, "run_sweep.py"], cwd=SIL, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        want = {(0.015, 1): 99.0, (0.015, 5): 95.4, (0.015, 30): 53.2,
                (0.030, 1): 98.9, (0.030, 5): 43.3, (0.030, 30): 56.1,
                (0.060, 1): 94.8, (0.060, 5): 95.0, (0.060, 30): 30.4,
                (0.090, 1): 98.3, (0.090, 5): 96.9, (0.090, 30): 26.2,
                (0.150, 1): 34.5, (0.150, 5): 33.5, (0.150, 30): 16.9,
                (0.225, 1): 33.2, (0.225, 5): 29.6, (0.225, 30): 15.3,
                (0.300, 1): 28.7, (0.300, 5): 24.9, (0.300, 30): 15.2,
                (0.450, 1): 4.8, (0.450, 5): 2.2, (0.450, 30): 14.7,
                (0.750, 1): 0.9, (0.750, 5): 1.4, (0.750, 30): 2.7,
                (1.500, 1): 0.3, (1.500, 5): 0.3, (1.500, 30): 0.3,
                (3.000, 1): 0.2, (3.000, 5): 0.1, (3.000, 30): 0.4}
        with open(os.path.join(SIL, "results", "sweep_results.csv")) as f:
            for rowd in csv.DictReader(f):
                key = (round(float(rowd["tau_h"]), 3), int(float(rowd["monitor_period_s"])))
                if key in want:
                    check(f"E8 tau={key[0]}h mon={key[1]}s benefit(%)",
                          float(rowd["rollback_benefit_pct"]), want[key], 0.15)

    if a.robust:
        print("=" * 78); print("E9. ROBUSTNESS  (Sec. 10, ~14 min)"); print("=" * 78)
        subprocess.run([sys.executable, "run_robustness.py"], cwd=SIL, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        d = json.load(open(os.path.join(SIL, "results", "robustness.json")))
        for c in ("C1", "C2", "C3"):
            check_int(f"E9 {c} holds", d["verdict"][c]["passed"], 18)
        fl = d["floor"]
        check("E9 floor perfect grace", fl["perfect_model_grace"], 0.0, 0.05)
        check("E9 floor perfect strict", fl["perfect_model_strict"], 337.0, 0.5)

    print("=" * 78)
    print(f"RESULT: {PASS} PASS / {FAIL} FAIL  ({PASS + FAIL} checks)")
    if FAIL:
        for ok, name, want, got in checks:
            if not ok: print(f"  FAILED: {name}  want {want}  got {got}")
    print("Host-dependent latencies (0.44 ms / 0.94 ms / gate us / backend ms) are not")
    print("checked here; reproduce with run_governance_demo.py, experiments/opa_http_latency.py.")
    sys.exit(FAIL)


if __name__ == "__main__":
    main()

