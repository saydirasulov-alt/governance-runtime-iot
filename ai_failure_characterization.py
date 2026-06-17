"""Realistic AI-failure characterization with an INDEPENDENT physical-safety oracle.

This replaces the earlier circular version (in which ground truth was defined by the
same predicate the gate checks, forcing detection to 100%). Here the ground-truth
hazard is a physical property the governance gate CANNOT directly observe, so the
fixed deterministic gate necessarily makes errors and we can measure a real
detection / false-positive tradeoff (ROC, AUC, confusion matrix).

SETTING (multi-zone HVAC / actuator governance)
  * Each control intent targets a zone whose TRUE safe setpoint band [L_z, U_z]
    depends on the zone type (server room, clinic, greenhouse, ...). This band is a
    latent physical property; the gate does not see it.
  * An upstream AI inference service emits a commanded setpoint x. Depending on the
    failure mode, x is a noisy/biased/out-of-distribution version of an intended
    value. Many x land near a boundary or remain plausible -- it is NOT always
    cleanly broken.
  * GROUND TRUTH (independent of the gate): the command is physically unsafe iff
    x falls outside that zone's TRUE band [L_z, U_z].
  * GOVERNANCE GATE (context-blind G1): admits iff x in the global fixed band
    [15, 30]. It cannot know the per-zone band, so it both MISSES hazards (x in
    [15,30] but outside a tighter zone band -> false negative) and OVER-BLOCKS
    (x outside [15,30] but inside a tolerant zone band -> false positive).

We report:
  (A) Deterministic-catch modes (missing timestamp / bad action / missing IDs):
      governed exactly, ~100% by construction -- reported honestly, not the focus.
  (B) Continuous safety detection: ROC + AUC over a swept decision threshold, the
      gate's fixed operating point ([15,30], theta=7.5) marked, and its confusion
      matrix with false-negative rate. A context-aware detector (knows the zone
      centre, not its width) is included to quantify how much a richer, config-
      defined policy could recover -- still below a perfect detector.

Run from the repo root:  python3 ai_failure_characterization.py
"""
import numpy as np

rng_master = np.random.default_rng(0)

# Zone types: (true safe band [L,U], selection weight). Latent to the gate.
ZONES = {
    "office":      ((18.0, 27.0), 0.30),
    "server_room": ((16.0, 23.0), 0.20),   # cooling-critical: high setpoint is hot/unsafe
    "clinic":      ((20.0, 25.0), 0.15),   # tight tolerance
    "greenhouse":  ((19.0, 31.0), 0.20),   # tolerates warmth -> some [15,30] rejects are SAFE
    "warehouse":   ((15.0, 29.0), 0.15),
}
ZONE_NAMES = list(ZONES.keys())
ZONE_BANDS = {k: v[0] for k, v in ZONES.items()}
ZONE_W = np.array([v[1] for v in ZONES.values()]); ZONE_W /= ZONE_W.sum()

GLOBAL_BAND = (15.0, 30.0)            # fixed governance gate band (context-blind)
GLOBAL_CENTRE = sum(GLOBAL_BAND) / 2  # 22.5
THETA_GATE = (GLOBAL_BAND[1] - GLOBAL_BAND[0]) / 2  # 7.5 -> the gate's operating point

# Setpoint-error failure modes (continuous-hazard part). Probabilities sum to 1.
MODE_P = {
    "valid":               0.50,   # small noise around the zone centre
    "miscalibrated_drift": 0.18,   # biased regressor: offset +/- a few degrees
    "distribution_shift":  0.12,   # OOD: wide setpoint, ignores intent
    "boundary_ambiguous":  0.12,   # lands right around a gate edge (15 or 30)
    "moderate_noise":      0.08,   # plausible but noisy
}
MODE_NAMES = list(MODE_P.keys())
MODE_W = np.array(list(MODE_P.values())); MODE_W /= MODE_W.sum()


def sample_intents(rng, n):
    """Return arrays: x (commanded setpoint), centre_z, true unsafe label, mode idx, zone idx."""
    zi = rng.choice(len(ZONE_NAMES), size=n, p=ZONE_W)
    mi = rng.choice(len(MODE_NAMES), size=n, p=MODE_W)
    L = np.array([ZONE_BANDS[ZONE_NAMES[z]][0] for z in zi])
    U = np.array([ZONE_BANDS[ZONE_NAMES[z]][1] for z in zi])
    centre = (L + U) / 2.0
    intended = rng.normal(centre, 1.2)            # AI intends something near the zone centre
    x = intended.copy()
    for k, name in enumerate(MODE_NAMES):
        m = mi == k
        c = m.sum()
        if c == 0:
            continue
        if name == "valid":
            x[m] = intended[m] + rng.normal(0, 0.4, c)
        elif name == "miscalibrated_drift":
            bias = rng.uniform(3.0, 8.0, c) * rng.choice([-1, 1], c)
            x[m] = intended[m] + bias + rng.normal(0, 1.5, c)
        elif name == "distribution_shift":
            x[m] = rng.uniform(8.0, 38.0, c)      # OOD, ignores intent
        elif name == "boundary_ambiguous":
            edge = rng.choice(GLOBAL_BAND, c)     # cluster around 15 or 30
            x[m] = edge + rng.normal(0, 0.8, c)
        elif name == "moderate_noise":
            x[m] = intended[m] + rng.normal(0, 2.5, c)
    # INDEPENDENT ground truth: unsafe iff outside the zone's TRUE band
    unsafe = (x < L) | (x > U)
    return x, centre, unsafe.astype(int), mi, zi


def auc_mw(scores, labels):
    """ROC AUC via the Mann-Whitney statistic (exact, handles ties = 0.5)."""
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([neg, pos])
    sidx = np.argsort(allv, kind="mergesort"); sv = allv[sidx]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[sidx[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[len(neg):]
    auc = (r_pos.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return auc


def roc_curve(scores, labels, n_thr=200):
    pos = labels == 1; neg = labels == 0
    P = pos.sum(); N = neg.sum()
    thrs = np.linspace(scores.min() - 1e-6, scores.max() + 1e-6, n_thr)
    tpr = []; fpr = []
    for t in thrs:
        flag = scores > t           # flag as unsafe (block) when score exceeds t
        tpr.append((flag & pos).sum() / P)
        fpr.append((flag & neg).sum() / N)
    return np.array(fpr), np.array(tpr), thrs


def bootstrap_ci(vals, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    bs = [np.mean(rng.choice(vals, len(vals), True)) for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def operating_point(x, unsafe):
    """Gate confusion matrix at the fixed [15,30] band (block = outside band)."""
    block = (x < GLOBAL_BAND[0]) | (x > GLOBAL_BAND[1])
    tp = int((block & (unsafe == 1)).sum())
    fp = int((block & (unsafe == 0)).sum())
    fn = int((~block & (unsafe == 1)).sum())
    tn = int((~block & (unsafe == 0)).sum())
    return tp, fp, fn, tn


def run(seeds=30, n=2000):
    blind_auc, aware_auc = [], []
    rec, fnr, fpr_op, prec = [], [], [], []
    decomp_rows = []
    agg = np.zeros(4, dtype=int)  # tp,fp,fn,tn
    pooled_blind = ([], [])  # scores, labels
    for s in range(1, seeds + 1):
        rng = np.random.default_rng(1000 + s)
        x, centre, unsafe, mi, zi = sample_intents(rng, n)
        s_blind = np.abs(x - GLOBAL_CENTRE)      # context-blind detector score
        s_aware = np.abs(x - centre)             # context-aware (knows zone centre, not width)
        blind_auc.append(auc_mw(s_blind, unsafe))
        aware_auc.append(auc_mw(s_aware, unsafe))
        tp, fp, fn, tn = operating_point(x, unsafe)
        agg += np.array([tp, fp, fn, tn])
        rec.append(tp / (tp + fn) if tp + fn else np.nan)
        fnr.append(fn / (tp + fn) if tp + fn else np.nan)
        fpr_op.append(fp / (fp + tn) if fp + tn else np.nan)
        prec.append(tp / (tp + fp) if tp + fp else np.nan)
        pooled_blind[0].append(s_blind); pooled_blind[1].append(unsafe)
        inb = (x >= GLOBAL_BAND[0]) & (x <= GLOBAL_BAND[1])
        gross_tot = int(((~inb) & (unsafe==1)).sum()); gross_caught = gross_tot  # outside band -> blocked
        ctx_tot = int((inb & (unsafe==1)).sum()); ctx_caught = 0                  # inside band -> admitted
        decomp_rows.append((gross_caught, gross_tot, ctx_caught, ctx_tot))
    pooled_scores = np.concatenate(pooled_blind[0]); pooled_labels = np.concatenate(pooled_blind[1])
    dr = np.array(decomp_rows)
    decomp = dict(gross_caught=int(dr[:,0].sum()), gross_tot=int(dr[:,1].sum()),
                  ctx_caught=int(dr[:,2].sum()), ctx_tot=int(dr[:,3].sum()))
    return {
        "blind_auc": bootstrap_ci(blind_auc), "aware_auc": bootstrap_ci(aware_auc),
        "recall": bootstrap_ci(rec), "fnr": bootstrap_ci(fnr),
        "fpr": bootstrap_ci(fpr_op), "precision": bootstrap_ci(prec),
        "confusion": agg, "pooled": (pooled_scores, pooled_labels), "decomp": decomp,
    }


def part_a_deterministic(seeds=30, n=2000):
    """Modes the gate is DESIGNED to catch exactly (metadata/action/identity).
    Ground truth here is the injected defect; recall is ~100% by construction.
    Reported honestly so the continuous-hazard ROC is not mistaken for the whole story."""
    from governance_runtime import GovernanceRuntime
    import time
    modes = {
        "stale_inference (no timestamp)": ("timestamp", None),
        "hallucinated_action":            ("action", "reboot_grid"),
        "missing_identity (no intent_id)":("intent_id", None),
    }
    out = {}
    for label, (field, bad) in modes.items():
        caught = total = 0
        for s in range(1, seeds+1):
            import random as _r; _r.seed(s)
            rt = GovernanceRuntime()
            for i in range(n):
                intent = {"setpoint":22.0,"action":"set_temperature","timestamp":time.time(),
                          "source":f"ai_{i%4}","intent_id":f"i{i}","device_id":f"d{i}"}
                intent[field] = bad
                dec,_g = rt._evaluate_gates(intent, 0)
                total += 1
                if dec in ("REJECT","THROTTLE"): caught += 1
        out[label] = (caught, total)
    return out


def youden_point(scores, labels):
    fpr, tpr, thr = roc_curve(scores, labels, n_thr=400)
    j = tpr - fpr
    k = int(np.argmax(j))
    return tpr[k], fpr[k], thr[k]


def fmt(ci): return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"


if __name__ == "__main__":
    import os
    print("=== (A) Deterministic-catch modes (gate designed to see these) ===")
    for lab,(c,t) in part_a_deterministic().items():
        print(f"  {lab:34s}: caught {c}/{t} = {c/t*100:.1f}%")
    print()
    R = run()
    tp, fp, fn, tn = R["confusion"]
    d = R["decomp"]
    sc, lb = R["pooled"]
    centre_blind = R  # alias
    print("=== (B) Continuous safety detection vs INDEPENDENT physical oracle ===")
    print(f"  Context-blind gate ROC AUC : {fmt(R['blind_auc'])}")
    print(f"  Context-aware  ROC AUC     : {fmt(R['aware_auc'])}   (knows zone centre)")
    print()
    print(f"  Gate operating point [15,30] (theta=7.5):")
    print(f"    recall / TPR            : {fmt(R['recall'])}")
    print(f"    FALSE-NEGATIVE rate     : {fmt(R['fnr'])}   <-- hazards admitted")
    print(f"    false-positive rate     : {fmt(R['fpr'])}")
    print(f"    precision               : {fmt(R['precision'])}")
    print(f"  Confusion (pooled): TP={tp} FP={fp} FN={fn} TN={tn}")
    print()
    print("  Hazard decomposition (what the blind spot actually is):")
    gr = d['gross_caught']/d['gross_tot'] if d['gross_tot'] else 0
    cr = d['ctx_caught']/d['ctx_tot'] if d['ctx_tot'] else 0
    share = d['ctx_tot']/(d['ctx_tot']+d['gross_tot'])
    print(f"    in-scope (global-band) hazards : caught {d['gross_caught']}/{d['gross_tot']} = {gr*100:.1f}%")
    print(f"    context-relative hazards       : caught {d['ctx_caught']}/{d['ctx_tot']} = {cr*100:.1f}%")
    print(f"    -> context-relative are {share*100:.1f}% of all hazards and are STRUCTURALLY")
    print(f"       invisible to a context-blind gate (this is the localized gap).")
    print()
    btpr, bfpr, bthr = youden_point(sc, lb)
    print(f"  ROC-optimal (Youden) for the SAME blind score: recall={btpr:.3f} at FPR={bfpr:.3f} (theta={bthr:.2f})")
    print("    i.e. retuning the band helps but cannot recover context hazards; only a")
    print("    context-aware (config-defined) policy raises AUC to ~0.98.")

    # ---- Figure: ROC with gate operating point + Youden + context-aware ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # recompute aware pooled curve
        rng = np.random.default_rng(1000+1)
        # build pooled aware scores across seeds for the curve
        aw_s, aw_l = [], []
        for s in range(1, 31):
            r = np.random.default_rng(1000+s)
            x, centre, unsafe, mi, zi = sample_intents(r, 2000)
            aw_s.append(np.abs(x-centre)); aw_l.append(unsafe)
        aw_s = np.concatenate(aw_s); aw_l = np.concatenate(aw_l)
        f1,t1,_ = roc_curve(sc, lb, 300)
        f2,t2,_ = roc_curve(aw_s, aw_l, 300)
        # gate operating point on blind curve
        g_tpr = R['recall'][0]; g_fpr = R['fpr'][0]
        plt.figure(figsize=(5.2,5.0))
        plt.plot(f1,t1,'-',color='#1f4e79',lw=2,label=f"Context-blind gate (AUC={R['blind_auc'][0]:.3f})")
        plt.plot(f2,t2,'-',color='#2e8b57',lw=2,label=f"Context-aware policy (AUC={R['aware_auc'][0]:.3f})")
        plt.plot([0,1],[0,1],'--',color='gray',lw=1,label='Chance')
        plt.scatter([g_fpr],[g_tpr],color='#c00000',zorder=5,s=70,
                    label=f"Deployed band [15,30]\n(recall={g_tpr:.2f}, FPR={g_fpr:.2f})")
        plt.scatter([bfpr],[btpr],color='#ed7d31',marker='D',zorder=5,s=55,
                    label=f"ROC-optimal blind\n(recall={btpr:.2f}, FPR={bfpr:.2f})")
        plt.xlabel("False-positive rate (safe intents blocked)")
        plt.ylabel("True-positive rate (hazards caught)")
        plt.title("AI-failure detection vs independent physical oracle")
        plt.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
        plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig("results/fig_ai_failure_roc.png", dpi=200)
        print("\nSaved figure: results/fig_ai_failure_roc.png")
    except Exception as e:
        print(f"\n[figure skipped: {e}]")

    # ---- Table 12 replacement (markdown) ----
    with open("results/table12_ai_failure.md","w") as fh:
        fh.write("# Table 12 (replacement): AI-failure characterization vs independent physical oracle\n\n")
        fh.write("All values are means over 30 seeds (2000 intents/seed); 95% bootstrap CIs in brackets.\n\n")
        fh.write("| Metric | Context-blind gate [15,30] | Context-aware policy |\n")
        fh.write("|---|---|---|\n")
        fh.write(f"| ROC AUC | {fmt(R['blind_auc'])} | {fmt(R['aware_auc'])} |\n")
        fh.write(f"| Recall (hazards caught) | {fmt(R['recall'])} | (Youden) {btpr:.3f} |\n")
        fh.write(f"| False-negative rate | {fmt(R['fnr'])} | -- |\n")
        fh.write(f"| False-positive rate | {fmt(R['fpr'])} | {bfpr:.3f} |\n")
        fh.write(f"| Precision | {fmt(R['precision'])} | -- |\n\n")
        fh.write(f"Hazard decomposition: in-scope global-band hazards caught "
                 f"{d['gross_caught']}/{d['gross_tot']} ({gr*100:.1f}%); context-relative hazards caught "
                 f"{d['ctx_caught']}/{d['ctx_tot']} ({cr*100:.1f}%). Context-relative hazards are "
                 f"{share*100:.1f}% of all hazards and are structurally invisible to a context-blind gate.\n")
    print("Saved Table 12 markdown: results/table12_ai_failure.md")
