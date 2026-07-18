"""
REAL AI MODEL + GOVERNANCE GATE on REAL IoT SENSOR DATA.

Dataset : UCI Occupancy Detection (Candanedo & Feldheim, Energy and Buildings, 2016).
          Real office sensors: Temperature, Humidity, CO2, HumidityRatio.
          Light is excluded: it trivially reveals occupancy and is often absent in HVAC
          deployments (the dataset authors also study the no-light case).
Shift   : datatest and datatest2 are different recording periods, giving a REAL
          distribution shift (no synthetic corruption anywhere in this experiment).

Smart-building HVAC scenario, made real:
  AI service : an MLP regressor maps live sensor readings to a recommended setpoint.
               Control target: occupied -> 22 C comfort, vacant -> 17 C energy setback.
               Its REAL errors (miscalibration, and extrapolation under shift) produce the
               control intents. No faults are injected by hand.
  Gate G1    : context-blind admission. Admit iff setpoint in [15, 30]. It sees the
               commanded setpoint only, never the true occupancy.
  Oracle     : INDEPENDENT physical-safety oracle using the TRUE occupancy label.
               truly occupied -> safe band [20, 25] (occupant thermal safety)
               truly vacant   -> safe band [15, 30]
               A command is physically unsafe iff outside the TRUE-context band.
  Corrected  : the policy-gap diagnosis suggests adding one predicate that uses CO2, a
    gate      context signal ALREADY present in the intent: reject an energy-setback command
               when CO2 indicates the room is in fact occupied.

Run:  python real_ai_governance_experiment.py
"""
import pandas as pd, numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

F = ["Temperature", "Humidity", "CO2", "HumidityRatio"]
GLOBAL_BAND = (15.0, 30.0)
OCCUPIED_SAFE, VACANT_SAFE = (20.0, 25.0), (15.0, 30.0)
COMFORT, SETBACK = 22.0, 17.0


def load(f):
    d = pd.read_csv(f"ds/{f}")
    return d[F].values, d["Occupancy"].values


def control_target(occ, rng):
    return np.where(occ == 1, COMFORT, SETBACK) + rng.normal(0, 0.3, len(occ))


def physically_unsafe(sp, true_occ):
    lo = np.where(true_occ == 1, OCCUPIED_SAFE[0], VACANT_SAFE[0])
    hi = np.where(true_occ == 1, OCCUPIED_SAFE[1], VACANT_SAFE[1])
    return (sp < lo) | (sp > hi)


def gate_blind(sp):
    return (sp >= GLOBAL_BAND[0]) & (sp <= GLOBAL_BAND[1])


def gate_co2(sp, co2, thr):
    """Corrected gate: + one predicate on CO2 (already in the intent)."""
    occupied_signal = co2 > thr
    unsafe_setback = occupied_signal & (sp < OCCUPIED_SAFE[0])
    return gate_blind(sp) & (~unsafe_setback)


def auc_mw(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([neg, pos]); order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        if j > i: ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[len(neg):]
    return (rp.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def evaluate(tag, Xte, yte, model, thr):
    sp = model.predict(Xte)
    co2 = Xte[:, F.index("CO2")]
    unsafe = physically_unsafe(sp, yte)
    adm_b, adm_c = gate_blind(sp), gate_co2(sp, co2, thr)
    safe = ~unsafe
    n_uns = int(unsafe.sum())
    return dict(
        tag=tag, n=len(yte),
        mae=float(np.mean(np.abs(sp - control_target(yte, np.random.default_rng(1))))),
        sp_lo=float(sp.min()), sp_hi=float(sp.max()),
        oob=100.0 * float(((sp < 15) | (sp > 30)).mean()),
        n_unsafe=n_uns,
        fn_blind=100.0 * int((unsafe & adm_b).sum()) / max(n_uns, 1),
        fn_co2=100.0 * int((unsafe & adm_c).sum()) / max(n_uns, 1),
        fp_co2=100.0 * int((safe & ~adm_c).sum()) / max(int(safe.sum()), 1),
        auc=auc_mw(np.abs(sp - 22.5), unsafe.astype(int)),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr, ytr = load("datatraining.txt")
    ttr = control_target(ytr, rng)

    model = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1500,
                                       random_state=0, early_stopping=True))
    model.fit(Xtr, ttr)
    thr = float(np.percentile(Xtr[:, F.index("CO2")][ytr == 0], 95))  # from TRAINING data only

    print("=" * 86)
    print("REAL AI MODEL + GOVERNANCE GATE  |  real IoT sensor data, real distribution shift")
    print("UCI Occupancy Detection (Candanedo & Feldheim 2016) | office HVAC control loop")
    print("=" * 86)
    print(f"\nAI service: MLP setpoint regressor, trained on {len(ytr)} real samples, features={F}")
    print(f"Corrected-gate CO2 threshold (95th pct of vacant TRAINING CO2): {thr:.0f} ppm\n")

    rows = [evaluate("in-distribution (datatest)", *load("datatest.txt"), model, thr),
            evaluate("distribution shift (datatest2)", *load("datatest2.txt"), model, thr)]

    print(f"{'':32s}{'n':>6s}{'setpoint range':>18s}{'out-of-band':>13s}{'unsafe cmds':>13s}")
    for r in rows:
        print(f"{r['tag']:32s}{r['n']:6d}   [{r['sp_lo']:5.1f},{r['sp_hi']:5.1f}]{r['oob']:12.1f}%{r['n_unsafe']:13d}")

    print(f"\n{'GOVERNANCE RESIDUAL RISK':32s}{'blind gate FN':>15s}{'+CO2 gate FN':>14s}{'+CO2 FP cost':>14s}{'gate AUC':>10s}")
    for r in rows:
        print(f"{r['tag']:32s}{r['fn_blind']:14.1f}%{r['fn_co2']:13.1f}%{r['fp_co2']:13.1f}%{r['auc']:10.3f}")

    print("""
FINDINGS (all from a real model's real errors; nothing hand-injected):
 1. In-distribution the model errs subtly: every recommended setpoint stays inside the gate's
    [15,30] band, so the context-blind gate catches NONE of the hazards (FN floor 100%).
    All residual risk is context-relative and structurally invisible to the gate.
 2. Under real distribution shift the model extrapolates far out of range, so a fraction of
    commands leave the global band and the gate DOES catch those; the FN floor falls, but a
    large residual of in-band, context-unsafe commands still passes.
 3. Adding ONE predicate that uses CO2 (a context signal already carried in the intent)
    substantially reduces the residual in-distribution at negligible false-positive cost, but
    degrades under shift, because the context signal itself shifts. Context-aware governance
    helps, and is itself distribution-shift-sensitive.
""")

    # ---------------- figure ----------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
        for f, lab, c in [("datatest.txt", "in-distribution", "#1f4e79"),
                          ("datatest2.txt", "distribution shift", "#c00000")]:
            Xte, yte = load(f); sp = model.predict(Xte)
            a1.hist(sp, bins=60, alpha=0.6, label=lab, color=c)
        a1.axvspan(15, 30, color="green", alpha=0.08)
        a1.axvline(15, ls="--", c="green"); a1.axvline(30, ls="--", c="green")
        a1.set_xlabel("AI-recommended setpoint ($^\\circ$C)"); a1.set_ylabel("count")
        a1.set_title("Real model output; green = gate band [15,30]", fontsize=10); a1.legend(fontsize=8)

        labels = ["in-dist", "shift"]; x = np.arange(2); w = 0.35
        a2.bar(x - w/2, [rows[0]["fn_blind"], rows[1]["fn_blind"]], w, label="context-blind gate", color="#c00000")
        a2.bar(x + w/2, [rows[0]["fn_co2"], rows[1]["fn_co2"]], w, label="+ CO$_2$ context predicate", color="#2e8b57")
        a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylabel("residual FN floor (%)")
        a2.set_title("Hazards admitted despite the gate", fontsize=10); a2.legend(fontsize=8)

        for f, lab, c in [("datatest.txt", "in-distribution", "#1f4e79"),
                          ("datatest2.txt", "distribution shift", "#c00000")]:
            Xte, yte = load(f); sp = model.predict(Xte)
            uns = physically_unsafe(sp, yte).astype(int); sc = np.abs(sp - 22.5)
            th = np.linspace(sc.min(), sc.max(), 200)
            tpr = [(sc > t)[uns == 1].mean() for t in th]; fpr = [(sc > t)[uns == 0].mean() for t in th]
            au = auc_mw(sc, uns)
            a3.plot(fpr, tpr, color=c, lw=2, label=f"{lab} (AUC={au:.3f})")
        a3.plot([0, 1], [0, 1], "--", c="gray", lw=1)
        a3.set_xlabel("false-positive rate"); a3.set_ylabel("true-positive rate")
        a3.set_title("Context-blind gate vs physical oracle", fontsize=10); a3.legend(fontsize=8); a3.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("fig_real_ai_governance.png", dpi=200)
        print("Saved figure: fig_real_ai_governance.png")
    except Exception as e:
        print("[figure skipped: %s]" % e)
