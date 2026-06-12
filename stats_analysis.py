"""Statistical analysis — reads real CSV results."""
import csv, os
import numpy as np
np.random.seed(42)
from scipy import stats

def bootstrap_ci(data, n_boot=10000):
    data = np.array([x for x in data if x is not None and not np.isnan(x)])
    if len(data) < 2: return np.nan, np.nan, np.nan
    medians = [np.median(np.random.choice(data, len(data), True)) for _ in range(n_boot)]
    return np.median(data), np.percentile(medians, 2.5), np.percentile(medians, 97.5)

def cliffs_delta(x, y):
    n = d = 0
    for xi in x:
        for yj in y:
            if xi > yj: d += 1
            elif xi < yj: d -= 1
            n += 1
    return d / n if n else 0

if __name__ == "__main__":
    if not os.path.exists("results"):
        print("Run experiment_runner.py first."); exit(1)

    print("=== ADVERSARIAL ===")
    rows = list(csv.DictReader(open("results/05_adversarial.csv")))
    faults = {}
    for r in rows:
        f = r["fault"]
        if f and f != "None":
            faults.setdefault(f, {"t": 0, "ua": 0})
            faults[f]["t"] += 1
            if r["outcome"] == "ADMITTED" and r["ground_truth_unsafe"] == "True":
                faults[f]["ua"] += 1
    for f in sorted(faults):
        v = faults[f]
        print(f"  {f:25s}: det={1-v['ua']/v['t']:.3f} ua={v['ua']}/{v['t']}")

    print("\n=== SCALABILITY ===")
    rows = list(csv.DictReader(open("results/06_scalability.csv")))
    for nv in ["n10","n100","n500","n1000","n2000","n5000"]:
        vals = [float(r["sim_e2e_ms"]) for r in rows
                if r["variant"]==nv and r.get("sim_e2e_ms") not in (None,"None","")]
        if vals:
            m, lo, hi = bootstrap_ci(vals)
            print(f"  {nv:8s}: P90={np.percentile(vals,90):7.1f}  med={m:.1f} [{lo:.1f},{hi:.1f}]")

    print("\n=== ABLATION ===")
    rows = list(csv.DictReader(open("results/02_ablation.csv")))
    ga = [float(r["sim_e2e_ms"]) for r in rows if r["variant"]=="GA-RT" and r.get("sim_e2e_ms") not in (None,"None","")]
    for v in ["GA-RT","NoPolicy","NoRollback","NoCheckpoint","NoAudit"]:
        vr = [r for r in rows if r["variant"]==v]
        ua = sum(1 for r in vr if r["outcome"]=="ADMITTED" and r.get("ground_truth_unsafe")=="True")
        rb = sum(1 for r in vr if r.get("rollback_success")=="1")
        # Rb success denominator = admitted intents where rollback is relevant
        rb_denom = sum(1 for r in vr if r.get("rollback_success") in ("0","1"))
        vals = [float(r["sim_e2e_ms"]) for r in vr if r.get("sim_e2e_ms") not in (None,"None","")]
        p90 = np.percentile(vals, 90) if vals else 0
        rb_rate = rb/rb_denom if rb_denom else 0
        print(f"  {v:15s}: P90={p90:.1f}ms ua={ua}/{len(vr)} rb_success={rb}/{rb_denom}={rb_rate:.3f}")

    print("\n=== BACKEND ===")
    rows = list(csv.DictReader(open("results/01_backend_comparison.csv")))
    for b in ["inline","http_rest","subprocess_cli"]:
        vr = [r for r in rows if r["variant"]==b]
        pol = [float(r["policy_ms"]) for r in vr if r.get("policy_ms") not in (None,"None","")]
        e2e = [float(r["sim_e2e_ms"]) for r in vr if r.get("sim_e2e_ms") not in (None,"None","")]
        if pol and e2e:
            print(f"  {b:18s}: policy P90={np.percentile(pol,90):.3f}ms  E2E P90={np.percentile(e2e,90):.1f}ms")

    # === TABLE 7: per-metric Wilcoxon + Cliff's delta (vs GA-RT) ===
    print("\n=== TABLE 7: ABLATION SIGNIFICANCE (vs GA-RT) ===")
    rows = list(csv.DictReader(open("results/02_ablation.csv")))

    def per_seed(variant, metric):
        """Aggregate a metric per seed for a variant."""
        seeds = {}
        for r in rows:
            if r["variant"] != variant: continue
            s = r["seed"]
            seeds.setdefault(s, [])
            if metric == "unsafe":
                seeds[s].append(1 if (r["outcome"]=="ADMITTED" and r.get("ground_truth_unsafe")=="True") else 0)
            elif metric == "rb_success":
                if r["outcome"] == "ADMITTED":
                    seeds[s].append(1 if r.get("rollback_success")=="1" else 0)
            elif metric == "mttr":
                v = r.get("sim_mttr_ms")
                if v not in (None,"None",""): seeds[s].append(float(v))
            elif metric == "e2e":
                v = r.get("sim_e2e_ms")
                if v not in (None,"None",""): seeds[s].append(float(v))
        return [np.mean(seeds[s]) if seeds[s] else 0 for s in sorted(seeds)]

    # Each row of Table 7: (comparison, metric)
    table7 = [
        ("NoPolicy", "unsafe", "Unsafe adm."),
        ("NoRollback", "rb_success", "Rb. success"),
        ("NoRollback", "mttr", "MTTR P90"),
        ("NoCheckpoint", "e2e", "E2E P90"),
        ("NoAudit", "e2e", "E2E P90"),
    ]
    for variant, metric, label in table7:
        ga = per_seed("GA-RT", metric)
        other = per_seed(variant, metric)
        try:
            if np.array_equal(ga, other):
                print(f"  vs {variant:12s} {label:12s}: identical (delta=0.000)")
            else:
                stat, p = stats.wilcoxon(ga, other)
                d = cliffs_delta(ga, other)
                print(f"  vs {variant:12s} {label:12s}: Wilcoxon p={p:.2e}, Cliff's d={d:+.3f}")
        except Exception:
            print(f"  vs {variant:12s} {label:12s}: no variance")
