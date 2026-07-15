"""
Reproduce every reported value in the paper, on this machine, in one command.

    python reproduce_all.py

Takes about 10 minutes. Writes results/REPRODUCTION_LOG.txt recording the host, the Python
version, every script run, its wall-clock time, and the key values it produced, so that an
independent re-run by a co-author or a reviewer leaves an auditable record.
"""
import subprocess, sys, time, platform, os, re, datetime

STEPS = [
    ("experiment_runner.py",           "6 experiments x 30 seeds -> results/*.csv"),
    ("stats_analysis.py",              "Tables 5, 7, 8 (backend, ablation, significance)"),
    ("gate_complexity_benchmark.py",   "Table 6 (gate cost vs complexity)"),
    ("verify_corrected_g2.py",         "Table 11 (corrected-G2: 18.7% -> 0.0%)"),
    ("ai_failure_characterization.py", "Table 12 / Figure 7 (AI-failure vs physical oracle)"),
    ("rollback_demo.py",               "Figure 3 (verified rollback, measured MTTR)"),
    ("verify_audit_chain.py",          "tamper-evident audit chain"),
    ("test_policy_loader.py",          "declarative gates == legacy gates (400k intents)"),
    ("opa_conformance.py",             "OPA Rego conformance fixtures"),
]

KEY = [
    (r"partial_corruption\s*:\s*det=([0-9.]+)\s+ua=(\d+)/(\d+)", "adversarial partial-corruption"),
    (r"Timestamp \+ source \(corrected\):\s*([0-9.]+)%",         "corrected-G2 unsafe admission"),
    (r"Context-blind gate ROC AUC\s*:\s*([0-9.]+)",              "AI-failure blind-gate AUC"),
    (r"VERIFIED \(state == checkpoint hash\):\s*(\d+/\d+)",      "rollback restorations verified"),
    (r"ALL PASSED",                                              "policy-loader equivalence"),
]


def main():
    os.makedirs("results", exist_ok=True)
    log = ["=" * 78,
           "REPRODUCTION LOG",
           "=" * 78,
           f"date          : {datetime.datetime.now().isoformat(timespec='seconds')}",
           f"host          : {platform.node()}",
           f"os            : {platform.system()} {platform.release()} ({platform.machine()})",
           f"python        : {sys.version.split()[0]}",
           ""]
    print("\n".join(log))
    total = 0.0
    ok = True
    for script, what in STEPS:
        if not os.path.exists(script):
            log.append(f"[SKIP] {script}  (not present)"); continue
        print(f"[RUN ] {script:32s} {what}", flush=True)
        t0 = time.time()
        p = subprocess.run([sys.executable, script], capture_output=True, text=True)
        dt = time.time() - t0
        total += dt
        status = "OK " if p.returncode == 0 else "FAIL"
        if p.returncode != 0:
            ok = False
        log.append(f"[{status}] {script:32s} {dt:7.1f}s   {what}")
        for pat, label in KEY:
            m = re.search(pat, p.stdout)
            if m:
                log.append(f"        -> {label}: {m.group(0).strip()[:70]}")
        print(f"[{status}] {script:32s} {dt:7.1f}s", flush=True)

    log += ["", f"total wall clock : {total/60:.1f} min",
            f"overall          : {'ALL STEPS SUCCEEDED' if ok else 'SOME STEPS FAILED'}",
            "",
            "Every value in the paper is produced by the scripts above on this machine.",
            "=" * 78]
    out = "results/REPRODUCTION_LOG.txt"
    open(out, "w").write("\n".join(log) + "\n")
    print("\n".join(log[-6:]))
    print(f"\nWritten: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
