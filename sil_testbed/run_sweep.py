"""
When is runtime rollback worth having?

The headline result of the main study -- rollback recovers essentially none of the
unsafe exposure -- is true OF OUR PLANT, and a careful reviewer will immediately ask the
right question: what if the plant were faster? A room with a three-hour time constant is
not a valve, a fan, or a robot arm. If physical recovery took seconds instead of an
hour, rollback would surely work. Stated as a fact about governance, our result would be
wrong. It is a fact about governance ON A SLOW PLANT.

So rather than defend one set of plant parameters we cannot validate against a real
building, we sweep them and let the sweep draw the boundary. That turns the study's
biggest vulnerability -- "your plant model is invented" -- into its most useful output:
a criterion an engineer can apply to their OWN system, before building a rollback
mechanism that will not help them.

Two knobs, and both are necessary
---------------------------------
PLANT SPEED. We scale the zone's thermal capacitance C. This is the honest single knob,
because it scales exactly the two things that matter and nothing else:

    tau   = R * C          the recovery time constant      ~ C
    dT/dt = (...) / C      the maximum slew rate            ~ 1/C

The equilibrium temperature (T_out + R*Q) does not involve C, so the GEOMETRY of the
hazard -- which setpoints are unsafe, and by how much -- is identical across the entire
sweep. Only the speed of the physical world changes.

MONITOR PERIOD. The first version of this sweep used a fixed 5 s monitor and came out
non-monotone: the very fastest plants scored WORSE than merely fast ones. That is not a
thing physics does, and it was not a bug either. It is a real effect, and missing it
would have been the more serious error. A plant slewing at 100 degC/min moves 8 degC
between two 5 s samples -- it is already deep out of band by the time the runtime first
looks at it. The monitor, not the plant, had become the binding constraint.

Sweeping both separates the two, and the answer has a shape worth reporting: rollback
has a USABLE BAND, bounded below by how fast you can sample and above by how fast you
can restore.

RUN IT ON THE FULL TRACE, AND AVERAGE OVER SEEDS. Both of those exist because we got
this wrong twice.

First we ran the sweep on a 2000-minute slice. The curves came out visibly non-monotone,
deterministically so, and we wrote up a tidy physical mechanism for it -- a phase resonance
between the monitor's sampling period and the limit cycle that rollback-then-re-admit
creates. It was small-sample noise in a ratio. On the full trace it vanished.

Then we fixed a real bug: the runtime monitor had been reading the TRUE plant state, which
made it an oracle rather than a monitor. It now reads a noisy sensor, as a real one does.
That fix is correct, and it made the rollback trigger stochastic -- so the single-seed
sweep started showing jitter again, and this time it was seed noise rather than sample
noise. Different cause, identical appearance, same temptation to explain it.

So the sweep now averages each point over several seeds and reports the spread. The
threshold survives both corrections, which is the point; the wiggles never did.

    python run_sweep.py
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gsim import gates as G
from gsim.aimodel import sensor_trace, train_setpoint_model
from gsim.loop import CONTROL_PERIOD_S, run_closed_loop
from gsim.plant import PlantParams, RoomPlant

HERE = os.path.dirname(os.path.abspath(__file__))
DS, RES, FIG = (os.path.join(HERE, d) for d in ("ds", "results", "figures"))
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SCALES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0]  # 1.0 = the paper's building
MONITORS = [1.0, 5.0, 30.0]                               # seconds
SEEDS = [7, 23, 101]         # the monitor reads a noisy sensor, so a single seed jitters
MINUTES = None                                            # None = the full trace
COLORS = {1.0: "#2e8b57", 5.0: "#2f6fb2", 30.0: "#c0392b"}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=int, default=MINUTES)
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 78)
    print("WHEN IS RUNTIME ROLLBACK WORTH HAVING?")
    print("=" * 78)
    print(f"  host {platform.node()}  {platform.system()}  python {platform.python_version()}")
    print("\n  Same model, same trace, same policy, same governance code. Two things vary:")
    print("  how fast the plant is, and how often the runtime looks at it.\n")

    model, info = train_setpoint_model(DS)
    trace = sensor_trace(DS, "datatest2.txt")
    shipped = G.shipped_policy()

    if args.minutes is not None and args.minutes < len(trace):
        print(f"  *** WARNING: running on {args.minutes} of {len(trace)} minutes.")
        print("  *** Truncated traces contain too few excursions for the exposure RATIO")
        print("  *** to be stable, and the curves come out spuriously non-monotone. Do")
        print("  *** not report these numbers. Use the full trace for anything real.\n")

    rows = []
    for sc in SCALES:
        pp = PlantParams(C_th=1.2e6 * sc)
        probe = RoomPlant(params=pp)

        # The no-rollback baseline does not depend on the monitor: nothing is watching.
        off = run_closed_loop(model, trace, shipped, arm=f"C={sc} no rb",
                              enable_rollback=False, plant_params=pp,
                              minutes=args.minutes)
        e_off = off.unsafe_exposure_c_min

        for mp in MONITORS:
            # Average over seeds. The monitor samples a noisy sensor, so whether a given
            # marginal excursion trips the deadband is a coin flip, and a single seed
            # produces jitter that looks exactly like structure. We were about to publish
            # that jitter as a resonance. Twice.
            bens, recs, rbs = [], [], []
            for sd in SEEDS:
                on = run_closed_loop(model, trace, shipped, arm=f"C={sc} rb m={mp} s={sd}",
                                     enable_rollback=True, plant_params=pp,
                                     monitor_period_s=mp, minutes=args.minutes, seed=sd)
                rec = [e.physical_recovery_s for e in on.rollback_events
                       if e.physical_recovery_s is not None]
                if rec:
                    recs.append(statistics.median(rec))
                rbs.append(on.rollbacks)
                bens.append(100.0 * (e_off - on.unsafe_exposure_c_min) / e_off
                            if e_off > 1e-9 else 0.0)
            med = statistics.median(recs) if recs else None
            rows.append({
                "C_scale": sc,
                "tau_h": round(probe.tau_thermal_h, 4),
                "slew_c_per_min": round(abs(probe.max_cooling_rate_c_per_min()), 2),
                "monitor_period_s": mp,
                "n_seeds": len(SEEDS),
                "median_recovery_s": round(med, 1) if med is not None else None,
                "exposure_no_rollback": round(e_off, 1),
                "rollback_benefit_pct": round(statistics.mean(bens), 1),
                "rollback_benefit_min_pct": round(min(bens), 1),
                "rollback_benefit_max_pct": round(max(bens), 1),
                "rollback_benefit_spread_pct": round(max(bens) - min(bens), 1),
                "rollbacks_mean": round(statistics.mean(rbs), 1),
                "peak_no_rollback_c": round(off.peak_excursion_c, 2),
            })

    # ------------------------------------------------------------------ table
    print(f"  {'tau':>7} {'slew':>9} |" + "".join(
        f"{'monitor ' + str(int(m)) + 's':>16}" for m in MONITORS))
    print(f"  {'h':>7} {'C/min':>9} |" + "".join(
        f"{'benefit':>9}{'  +-sd':>7}" for _ in MONITORS))
    print("  " + "-" * 68)
    for sc in SCALES:
        rs = [r for r in rows if r["C_scale"] == sc]
        line = f"  {rs[0]['tau_h']:7.3f} {rs[0]['slew_c_per_min']:9.1f} |"
        for mp in MONITORS:
            r = next(x for x in rs if x["monitor_period_s"] == mp)
            line += (f"{r['rollback_benefit_pct']:8.1f}%"
                     f" +-{r['rollback_benefit_spread_pct']/2:4.1f}")
        print(line)

    # -------------------------------------------------------------- criterion
    print("\n" + "=" * 78)
    print("WHAT THE SWEEP SAYS")
    print("=" * 78)

    best = max(rows, key=lambda r: r["rollback_benefit_pct"])
    building = [r for r in rows if r["C_scale"] == 1.0 and r["monitor_period_s"] == 5.0][0]

    m1 = sorted([r for r in rows if r["monitor_period_s"] == 1.0],
                key=lambda r: -r["slew_c_per_min"])
    hi = [r for r in m1 if r["rollback_benefit_pct"] >= 90]
    lo = [r for r in m1 if r["rollback_benefit_pct"] <= 5]

    print("  It is a THRESHOLD, not a gradient. That is the useful part.\n")
    if hi and lo:
        print(f"    plants slewing at {min(r['slew_c_per_min'] for r in hi):.1f} degC/min "
              f"or faster (tau <= {max(r['tau_h'] for r in hi)*60:.0f} min):")
        print(f"        rollback removes at least "
              f"{min(r['rollback_benefit_pct'] for r in hi):.0f}% of the unsafe exposure")
        print(f"    plants slewing at {max(r['slew_c_per_min'] for r in lo):.1f} degC/min "
              f"or slower (tau >= {min(r['tau_h'] for r in lo)*60:.0f} min):")
        print(f"        rollback removes at most "
              f"{max(r['rollback_benefit_pct'] for r in lo):.1f}%\n")
    print("  The system either wins the race against the next bad command on every cycle")
    print("  or it loses on every cycle; the transition between the two is narrow. Rollback")
    print("  undoes ONE command, and a systematically wrong model simply issues it again on")
    print("  the next cycle, so the runtime has to be able to restore the plant well inside")
    print(f"  the {CONTROL_PERIOD_S} s between decisions or it never catches up at all.\n")
    print("  A NOTE ON THE RECOVERY-TIME COLUMN. Do not use it as the predictor. It is a")
    print("  median over excursions that DID recover, so on slow plants -- where many")
    print("  excursions never recover before the trace ends -- it is biased low and can even")
    print("  fall as the plant gets slower. The slew rate is the honest independent")
    print("  variable, because it is a property of the plant rather than of the outcome.\n")
    print(f"  UPPER BOUND -- the plant is too slow. The building in this paper")
    print(f"  (tau = {building['tau_h']:.1f} h, {building['slew_c_per_min']:.2f} degC/min) "
          f"sits far above the threshold:")
    print(f"  rollback removes {building['rollback_benefit_pct']:.1f}%. Sampling faster does "
          f"not help, because the")
    print("  bottleneck is the room, not the sensor.\n")
    print("  LOWER BOUND -- the monitor is too slow. On the fastest plants a 30 s monitor is")
    print("  worth a fraction of a 1 s one: the plant leaves the safe band between samples,")
    print("  so the runtime only ever sees the aftermath. Here the bottleneck IS the sensor,")
    print("  and unlike the room it is fixable.\n")
    print(f"  BEST CASE in this sweep: {best['rollback_benefit_pct']:.1f}% "
          f"(tau = {best['tau_h']*60:.0f} min, monitor {best['monitor_period_s']:.0f} s).\n")
    print("  So the honest claim is not 'rollback does not work'. It is:\n")
    print("      Runtime rollback delivers safety when the plant can be restored faster")
    print("      than the next bad command arrives, and sampled faster than it can leave")
    print("      the safe set. Outside that band it degrades into a bookkeeping mechanism:")
    print("      a correct audit trail and a defined terminal state, but not safety.")
    print("      Safety there has to come from admission control, and admission control is")
    print("      only as good as the policy's estimate of context.\n")
    print("  Every quantity in that criterion is measurable on a real system before any")
    print("  code is written: the actuator's recovery time, the control period, and the")
    print("  monitor's sampling rate.")

    # ---------------------------------------------------------------- outputs
    with open(os.path.join(RES, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "sweep_results.json"), "w") as f:
        json.dump({"host": platform.node(), "os": platform.system(),
                   "python": platform.python_version(),
                   "control_period_s": CONTROL_PERIOD_S, "model": info,
                   "rows": rows}, f, indent=2)
    figure(rows)
    print(f"\n  {time.time()-t0:.1f} s")
    print("  results/sweep_results.csv   figures/fig_sil_5_when_rollback_works.png")


def figure(rows) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for mp in MONITORS:
        rs = sorted([r for r in rows if r["monitor_period_s"] == mp],
                    key=lambda r: r["slew_c_per_min"])
        x = [r["slew_c_per_min"] for r in rs]
        ax.fill_between(x, [r["rollback_benefit_min_pct"] for r in rs],
                        [r["rollback_benefit_max_pct"] for r in rs],
                        color=COLORS[mp], alpha=0.15, lw=0)
        ax.plot(x, [r["rollback_benefit_pct"] for r in rs], "-o",
                color=COLORS[mp], lw=2, ms=5,
                label=f"monitor samples every {mp:.0f} s  (mean of {len(SEEDS)} seeds)")

    ax.set_xscale("log")
    ax.invert_xaxis()          # slow plants on the right, where the building is
    ax.set_xlabel("Plant slew rate  ($^{\\circ}$C/min, log scale)   "
                  "$\\longleftarrow$ faster        slower $\\longrightarrow$", fontsize=10)
    ax.set_ylabel("Unsafe exposure removed\nby rollback  (%)", fontsize=10)
    ax.set_ylim(-5, 105)

    b = [r for r in rows if r["C_scale"] == 1.0 and r["monitor_period_s"] == 5.0][0]
    ax.plot([b["slew_c_per_min"]], [b["rollback_benefit_pct"]], "o", ms=13,
            mfc="none", mec="#1a1a1a", mew=2, zorder=5)
    ax.annotate(f"the building in this paper\n(tau = {b['tau_h']:.1f} h): "
                f"{b['rollback_benefit_pct']:.1f}%",
                xy=(b["slew_c_per_min"], b["rollback_benefit_pct"]),
                xytext=(-8, 55), textcoords="offset points", fontsize=8.5,
                color="#1a1a1a", ha="center",
                arrowprops=dict(arrowstyle="->", color="#1a1a1a", lw=1))

    ax.text(0.015, 0.56, "a slow monitor cannot see it:\nthe plant leaves the band\n"
                         "between samples",
            transform=ax.transAxes, fontsize=8.5, color="#c0392b", va="top")
    ax.text(0.60, 0.34, "the plant cannot be restored\nbefore the next bad command\n"
                        "arrives, whatever the monitor does",
            transform=ax.transAxes, fontsize=8.5, color="#1a1a1a", va="top")
    ax.text(0.30, 0.985, "rollback works here", transform=ax.transAxes,
            fontsize=9, color="#2e8b57", va="top", fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False, loc=(0.02, 0.06))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Rollback has a usable band: bounded below by how fast you sample, "
                 "above by how\nfast you can restore. The building in this paper sits "
                 "outside it.",
                 fontsize=10.5, color="#1a1a1a", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sil_5_when_rollback_works.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
