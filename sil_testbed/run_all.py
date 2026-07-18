"""
Everything, in one command.

    python run_all.py

Runs, in order, and stops at the first failure:

    0  the 25 tests            5 s     nothing below this is trustworthy if these fail
    1  E1-E7, the main study   60 s
    2  the figures             5 s
    3  E8, the plant sweep     9 min   seed-averaged; this is the slow one
    4  E9, the robustness      23 min  72 full-trace runs across 18 perturbations
    5  a summary of everything the paper needs

Total: about 33 minutes on a laptop. Almost all of it is E8 and E9 integrating the
9752-minute plant thousands of times; the tests and the main study are quick. The
runtime is dominated by physics, not by the model, which is the paper's point in
miniature: the governance decision is microseconds, the plant is what takes the time.

Writes a complete transcript to results/RUN_ALL_LOG.txt, stamped with the host, OS and
Python version, so the reproducibility claim in the paper is a fact rather than a promise.

It stops at the first failure on purpose. In this project a wrong number that carried
forward was, every single time, harder to find than one that stopped the run.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
LOG = os.path.join(RES, "RUN_ALL_LOG.txt")

STEPS = [
    ("tests",      [sys.executable, "-m", "pytest", "tests/", "-q"],   "5 s"),
    ("E1-E7",      [sys.executable, "run_sil.py"],                     "60 s"),
    ("figures",    [sys.executable, "make_figures.py"],                "5 s"),
    ("E8 sweep",   [sys.executable, "run_sweep.py"],                   "9 min"),
    ("E9 audit",   [sys.executable, "run_robustness.py"],              "23 min"),
]


def main() -> None:
    t0 = time.time()
    header = (
        "=" * 78 + "\n"
        "RUN ALL\n" + "=" * 78 + "\n"
        f"  host    {platform.node()}\n"
        f"  os      {platform.system()} {platform.release()}\n"
        f"  python  {platform.python_version()}\n"
        f"  cwd     {HERE}\n\n"
        "  No physical hardware is operated. Every physical quantity below is produced by\n"
        "  the digital twin, whose model and parameters are given in full in gsim/plant.py.\n"
    )
    print(header)
    log = open(LOG, "w", encoding="utf-8")
    log.write(header)

    for i, (name, cmd, eta) in enumerate(STEPS, 1):
        banner = f"\n{'-' * 78}\n[{i}/{len(STEPS)}]  {name}   (about {eta})\n{'-' * 78}"
        print(banner, flush=True)
        log.write(banner + "\n")
        log.flush()

        t = time.time()
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()
        dt = time.time() - t

        if proc.returncode != 0:
            msg = (f"\n*** {name} FAILED (exit {proc.returncode}) after {dt:.0f} s.\n"
                   f"*** Stopping here rather than carrying a bad result forward.\n"
                   f"*** The full transcript is in {LOG} -- send it as-is.\n")
            print(msg)
            log.write(msg)
            log.close()
            sys.exit(1)

        done = f"\n[{i}/{len(STEPS)}] {name} OK  ({dt:.0f} s)\n"
        print(done)
        log.write(done)
        log.flush()

    # ---------------------------------------------------------------- summary
    summary = summarise()
    print(summary)
    log.write(summary)
    tail = f"\nTOTAL {time.time() - t0:.0f} s\nTranscript: {LOG}\n"
    print(tail)
    log.write(tail)
    log.close()


def summarise() -> str:
    """Pull the numbers the paper actually needs out of the result files."""
    import csv
    import json

    out = ["\n" + "=" * 78, "WHAT THE PAPER NEEDS", "=" * 78]

    try:
        with open(os.path.join(RES, "sil_results.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        shift = [r for r in rows if r.get("regime") == "shift"
                 and r["arm"] in ("ungoverned", "shipped, no rollback", "shipped + rollback",
                                  "corrected + rollback", "oracle + rollback")]
        out.append("\nE1  unsafe physical exposure under distribution shift (degC-min)")
        for r in shift:
            out.append(f"      {r['arm']:24s} {float(r['unsafe_exposure_c_min']):8.1f}   "
                       f"peak {float(r['peak_excursion_c']):5.2f}   "
                       f"rollbacks {r['rollbacks']}")
        ung = next(float(r["unsafe_exposure_c_min"]) for r in shift if r["arm"] == "ungoverned")
        nrb = next(float(r["unsafe_exposure_c_min"]) for r in shift
                   if r["arm"] == "shipped, no rollback")
        rb = next(float(r["unsafe_exposure_c_min"]) for r in shift
                  if r["arm"] == "shipped + rollback")
        out.append(f"\n      admission control:  {ung:.0f} -> {nrb:.0f}")
        out.append(f"      adding rollback:    {nrb:.0f} -> {rb:.0f}   "
                   f"= {100*(nrb-rb)/nrb:.1f}%   <-- the headline")
    except Exception as e:
        out.append(f"  (could not read sil_results.csv: {e})")

    try:
        with open(os.path.join(RES, "sweep_results.csv"), encoding="utf-8") as f:
            sw = [r for r in csv.DictReader(f) if r["monitor_period_s"] == "1.0"]
        out.append("\nE8  when rollback is worth having  (1 s monitor, seed-averaged)")
        out.append(f"      {'tau (h)':>8} {'slew':>8} {'benefit':>10} {'spread':>8}")
        for r in sw:
            out.append(f"      {float(r['tau_h']):8.3f} {float(r['slew_c_per_min']):8.1f} "
                       f"{float(r['rollback_benefit_pct']):9.1f}% "
                       f"{float(r['rollback_benefit_spread_pct']):7.1f}")
    except Exception as e:
        out.append(f"  (could not read sweep_results.csv: {e})")

    try:
        with open(os.path.join(RES, "robustness.json"), encoding="utf-8") as f:
            rb = json.load(f)
        # READ the verdict; do not recompute it. This block used to count the
        # perturbations itself, over a different set of groups than run_robustness.py
        # used, so one run printed 15/15 here and 14/14 there. The verdict now has a
        # single source -- the same discipline the figures already follow.
        v = rb["verdict"]
        out.append("\nE9  robustness")
        out.append(f"      {v['n_perturbations']} perturbations")
        for i, nm in enumerate(("C1  rollback buys < 10%",
                                "C2  corrected beats shipped",
                                "C3  oracle removes ~all of it"), start=1):
            c = v[f"C{i}"]
            out.append(f"      {nm:32s} {c['passed']}/{c['total']} "
                       f"{'HOLDS' if not c['fails'] else '*** BROKEN: ' + ', '.join(c['fails'])}")
        fl = rb.get("floor", {})
        if fl:
            out.append(f"      floor: a perfect model scores "
                       f"{fl.get('perfect_model_grace')} under the grace window "
                       f"(must be 0.0)")
    except Exception as e:
        out.append(f"  (could not read robustness.json: {e})")

    out.append("\n  Send me this block, plus results/RUN_ALL_LOG.txt.")
    return "\n".join(out)


if __name__ == "__main__":
    main()
