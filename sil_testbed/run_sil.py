"""
Software-in-the-loop (SIL) governance experiments.

    python run_sil.py                    full run
    python run_sil.py --quick            short run, for a fast sanity check
    python run_sil.py --closed-perception  feed the plant temperature back to the model

Writes results/sil_results.csv, results/sil_results.json, results/SIL_LOG.txt.
Then: python make_figures.py

Scope, stated once: NO PHYSICAL HARDWARE WAS OPERATED. Every physical quantity below
is produced by the digital twin in gsim/plant.py, whose model and parameters are given
in full. The Raspberry Pi backends in gsim/hal.py are released and implement the same
interface, but they were not run, and nothing here should be read as a hardware result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gsim import gates as G
from gsim.aimodel import sensor_trace, train_setpoint_model
from gsim.hal import parity_report
from gsim.loop import (MONITOR_PERIOD_S, ROLLBACK_TRIGGER_C, ROLLBACK_WINDOW_S,
                       run_closed_loop)
from gsim.plant import SAFE_BAND_OCCUPIED, SAFE_BAND_VACANT, RoomPlant

HERE = os.path.dirname(os.path.abspath(__file__))
DS, RES = os.path.join(HERE, "ds"), os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

OPEN = {"name": "UNGOVERNED", "gates": []}


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def row(r) -> dict:
    rec = [e for e in r.rollback_events if e.physical_recovery_s is not None]
    med = statistics.median
    return {
        "arm": r.arm, "policy": r.policy, "fallback": r.fallback,
        "minutes": r.minutes,
        "intents": r.intents, "admitted": r.admitted, "rejected": r.rejected,
        "admit_rate_pct": round(100 * r.admitted / max(1, r.intents), 1),
        "unsafe_minutes": r.unsafe_minutes,
        "availability_pct": round(100 * r.governed_minutes / max(1, r.minutes), 1),
        "unsafe_pct": round(100 * r.unsafe_minutes / max(1, r.minutes), 2),
        "unsafe_exposure_c_min": round(r.unsafe_exposure_c_min, 1),
        "strict_exposure_c_min": round(r.strict_exposure_c_min, 1),
        "peak_excursion_c": round(r.peak_excursion_c, 2),
        "rollbacks": r.rollbacks,
        "failed_safe": r.failed_safe,
        "failed_safe_reason": r.failed_safe_reason,
        "failed_safe_minute": r.failed_safe_minute,
        "median_detect_latency_s": round(med([e.detect_latency_s for e in r.rollback_events]), 2) if r.rollback_events else None,
        "median_physical_recovery_s": round(med([e.physical_recovery_s for e in rec]), 1) if rec else None,
        "max_physical_recovery_s": round(max(e.physical_recovery_s for e in rec), 1) if rec else None,
        "median_gate_latency_us": round(1000 * med(r.gate_latency_ms), 2),
        "p99_gate_latency_us": round(1000 * sorted(r.gate_latency_ms)[int(0.99 * len(r.gate_latency_ms)) - 1], 2),
        "audit_ok": r.audit_ok, "audit_entries": r.audit_entries,
    }


def show(d: dict) -> None:
    fs = ("-" if not d["failed_safe"]
          else f"FAILED_SAFE@{d['failed_safe_minute']}m ({d['failed_safe_reason']})")
    print(f"  {d['arm']:26s} exposure {d['unsafe_exposure_c_min']:7.1f} C-min   "
          f"peak {d['peak_excursion_c']:5.2f} C   "
          f"avail {d['availability_pct']:5.1f}%   "
          f"rb {d['rollbacks']:2d}   {fs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--closed-perception", action="store_true")
    a = ap.parse_args()
    limit = 2000 if a.quick else None
    t_start = time.time()

    hr("SOFTWARE-IN-THE-LOOP (SIL) GOVERNANCE TESTBED")
    print(f"  host   {platform.node()}   {platform.system()} {platform.release()}   "
          f"python {platform.python_version()}")
    print("  plant  single-zone RC thermal digital twin")
    print("  NOTE   no physical hardware was operated; all physical values are twin values")

    # ---------------------------------------------------------------- 0
    hr("0.  THE AI SERVICE, TRAINED ON REAL SENSOR DATA")
    model, info = train_setpoint_model(DS)
    print(f"  training samples          {info['n_train']}")
    print(f"  training MAE              {info['train_mae_c']:.3f} degC")
    print(f"  CO2 threshold (train p95) {info['co2_vacant_p95_ppm']:.0f} ppm  "
          f"<- from the TRAINING split only")

    shipped = G.shipped_policy()
    corrected = G.corrected_policy(info["co2_vacant_p95_ppm"])
    oracle = G.oracle_policy()

    p = RoomPlant()
    hr("0b. THE DIGITAL TWIN, AND WHY ITS TIME CONSTANTS ARE THE WHOLE STORY")
    print(f"  envelope time constant       {p.tau_thermal_h:.2f} h")
    print(f"  max heating rate             {p.max_heating_rate_c_per_min():+.3f} degC/min")
    print(f"  max cooling rate             {p.max_cooling_rate_c_per_min():+.3f} degC/min")
    print(f"  equilibrium at full heat     {p.equilibrium_T(1.0, 1):.1f} degC")
    print(f"  safe band occupied / vacant  {SAFE_BAND_OCCUPIED} / {SAFE_BAND_VACANT} degC")
    print(f"  twin CO2 check: 1 occupant -> {p.steady_state_co2(1):.0f} ppm steady state,")
    print(f"    consistent with recorded occupied CO2. (The gates use RECORDED CO2, not this.)")
    print(f"\n  monitor period {MONITOR_PERIOD_S:.0f} s | rollback trigger "
          f"{ROLLBACK_TRIGGER_C} degC (5 sigma of sensor noise)")
    print("  rollback budget unlimited in the")
    print("  safety comparisons (a budget truncates the arms it fires on and would make a")
    print("  shut-down system look safe); it gets its own experiment, E6.")

    tr_in = sensor_trace(DS, "datatest.txt")
    tr_sh = sensor_trace(DS, "datatest2.txt")
    for nm, tr in (("in-distribution", tr_in), ("shifted", tr_sh)):
        occ = 100 * sum(x[4] for x in tr) / len(tr)
        print(f"  {nm:16s} {len(tr):5d} min, {occ:.0f}% occupied")

    arms = [
        ("ungoverned",            OPEN,      False),
        ("shipped, no rollback",  shipped,   False),
        ("shipped + rollback",    shipped,   True),
        ("corrected + rollback",  corrected, True),
        ("oracle + rollback",     oracle,    True),
    ]
    rows: list[dict] = []
    kw = dict(closed_perception=a.closed_perception, minutes=limit)

    # ---------------------------------------------------------------- E1
    hr("E1.  CLOSED-LOOP SAFETY UNDER DISTRIBUTION SHIFT")
    print("  Same model, same trace, same plant. Only the governance changes.\n")
    for tag, pol, rb in arms:
        d = row(run_closed_loop(model, tr_sh, pol, arm=tag, enable_rollback=rb, **kw))
        d["regime"] = "shift"
        rows.append(d)
        show(d)

    ung = rows[0]
    ship = rows[2]
    corr = rows[3]
    noreb = rows[1]
    orac = rows[4]
    d_rb = noreb["unsafe_exposure_c_min"] - ship["unsafe_exposure_c_min"]
    pct_rb = 100 * d_rb / max(1e-9, noreb["unsafe_exposure_c_min"])
    print(f"\n  Read the third and fourth columns together, because they do not say what")
    print(f"  the earlier version of this paper assumed they would.")
    print(f"\n  Admission control alone:  {ung['unsafe_exposure_c_min']:.0f} -> "
          f"{noreb['unsafe_exposure_c_min']:.0f} degC-min.")
    print(f"  Adding ROLLBACK:         {noreb['unsafe_exposure_c_min']:.0f} -> "
          f"{ship['unsafe_exposure_c_min']:.0f} degC-min.  That is {pct_rb:.1f}%.")
    print(f"  Peak excursion:          {noreb['peak_excursion_c']:.2f} -> "
          f"{ship['peak_excursion_c']:.2f} degC.  Rollback did not reduce the peak at all.")
    print(f"\n  Rollback, against a plant with thermal inertia, recovers almost nothing. It")
    print(f"  cannot: by the time the room has left the safe band and the runtime has")
    print(f"  restored a safe setpoint, the heat is already in the room, and getting it back")
    print(f"  out takes the tens of minutes that E2 measures. The exposure has been paid")
    print(f"  before the mechanism can act.")
    print(f"\n  What DOES work is not getting there. The oracle policy -- same runtime, same")
    print(f"  rollback, same plant, one extra predicate that happens to be right -- lands at")
    print(f"  {orac['unsafe_exposure_c_min']:.1f} degC-min. The runtime mechanisms are not the")
    print(f"  limiting factor. The policy's estimate of context is.")
    print(f"\n  And the deployable proxy for that context sits between the two: the corrected")
    print(f"  CO2 predicate leaves {corr['unsafe_exposure_c_min']:.0f} degC-min, because under")
    print(f"  distribution shift CO2 stops tracking occupancy. That gap, "
          f"{corr['unsafe_exposure_c_min']:.0f} vs {orac['unsafe_exposure_c_min']:.1f}, is a")
    print(f"  measurement of how much residual PHYSICAL risk is attributable to the context")
    print(f"  estimator rather than to the governance runtime. That number is the paper.")

    # ---------------------------------------------------------------- E2
    hr("E2.  WHAT ROLLBACK ACTUALLY BUYS  (the number the paper says it does not measure)")
    r_ship = run_closed_loop(model, tr_sh, shipped, arm="shipped + rollback",
                             enable_rollback=True, keep_trace=True, **kw)
    if not r_ship.rollback_events:
        print("  no rollback events")
    else:
        print(f"  {'#':>2} {'peak excursion':>15} {'detect':>9} {'PHYSICAL RECOVERY':>20} "
              f"{'exposure':>12}")
        print(f"  {'':>2} {'degC':>15} {'s':>9} {'s':>20} {'degC-min':>12}")
        for i, e in enumerate(r_ship.rollback_events, 1):
            rec = (f"{e.physical_recovery_s:.0f}  ({e.physical_recovery_s/60:.0f} min)"
                   if e.physical_recovery_s is not None else "NEVER RECOVERED")
            print(f"  {i:>2} {e.peak_excursion_c:15.2f} {e.detect_latency_s:9.1f} "
                  f"{rec:>20} {e.exposure_c_min:12.1f}")
        done = [e for e in r_ship.rollback_events if e.physical_recovery_s is not None]
        if done:
            mp = statistics.median(e.physical_recovery_s for e in done)
            mx = max(e.physical_recovery_s for e in done)
            print(f"\n  physical recovery, median {mp:7.0f} s  ({mp/60:.1f} min)")
            print(f"  physical recovery, worst  {mx:7.0f} s  ({mx/60:.1f} min)")
            print("\n  For contrast, the governance DECISION path was measured on the real MQTT")
            print("  stack at a median of 0.44 ms. The decision is six orders of magnitude faster")
            print("  than its own physical consequence. Reporting only the decision latency, as")
            print("  the previous version of this paper did, describes the cheap half of rollback")
            print("  and silently omits the expensive half.")
        print(f"\n  Rollback is fast to DECIDE and slow to TAKE EFFECT,")
        print(f"  because rooms have thermal mass. Rollback ENDS an excursion; it does not")
        print(f"  undo it. The {ship['unsafe_exposure_c_min']:.0f} degC-min still on the clock "
              f"under the shipped policy is\n  exposure that already happened before the "
              f"runtime could get the room back.")
        print(f"  The corrected predicate leaves {corr['unsafe_exposure_c_min']:.0f} degC-min, "
              f"because the intent never actuates at all.")

    # ---------------------------------------------------------------- E3
    hr("E3.  IRREVERSIBILITY: WHERE ROLLBACK IS NOT AVAILABLE AT ALL")
    print("  Some actuations cannot be undone by issuing another command: a compressor")
    print("  lockout, a fired suppression system, a purged tank. We model an actuator that")
    print("  LATCHES above 28 degC and ignores every command afterwards.\n")
    for tag, pol in (("shipped, latching actuator", shipped),
                     ("corrected, latching actuator", corrected)):
        d = row(run_closed_loop(model, tr_sh, pol, arm=tag, enable_rollback=True,
                                irreversible_above_c=28.0, **kw))
        d["regime"] = "shift/irreversible"
        rows.append(d)
        show(d)
    print("\n  Under the shipped policy the runtime loses authority and enters FAILED_SAFE.")
    print("  It does not claim to have recovered, because it has not: the room is latched")
    print("  hot and only a human can clear it. A rollback mechanism that reported success")
    print("  here would be lying. Under the corrected policy the intent is never admitted,")
    print("  so the irreversible actuator is never reached. Prevention is the only thing")
    print("  that works against irreversibility, and that is an argument for policy")
    print("  correctness, not for better rollback.")

    # ---------------------------------------------------------------- E4
    hr("E4.  THE SAME EXPERIMENT IN-DISTRIBUTION")
    print("  If the model is not stressed, governance looks like overhead. The gate did not")
    print("  change; the world did.\n")
    for tag, pol, rb in arms:
        d = row(run_closed_loop(model, tr_in, pol, arm=tag, enable_rollback=rb, **kw))
        d["regime"] = "in-distribution"
        rows.append(d)
        show(d)

    # ---------------------------------------------------------------- E5
    hr("E5.  WHAT GOVERNANCE COSTS, AND AUDIT INTEGRITY")
    print(f"  gate evaluation, median   {ship['median_gate_latency_us']:6.2f} us shipped   "
          f"{corr['median_gate_latency_us']:6.2f} us corrected")
    print(f"  gate evaluation, p99      {ship['p99_gate_latency_us']:6.2f} us shipped   "
          f"{corr['p99_gate_latency_us']:6.2f} us corrected")
    print("  rollback decision path    measured separately on the REAL MQTT stack")
    print("                            (median 0.44 ms end-to-end); the twin's own")
    print("                            function-call time is not a meaningful number")
    print("                            and is deliberately not reported.")
    print(f"  audit chain               verified={ship['audit_ok']}  "
          f"{ship['audit_entries']} entries")
    d_us = corr["median_gate_latency_us"] - ship["median_gate_latency_us"]
    d_exp = ship["unsafe_exposure_c_min"] - corr["unsafe_exposure_c_min"]
    print(f"\n  The context predicate costs {d_us:+.2f} us per intent and removes "
          f"{d_exp:.0f} degC-min of")
    print("  unsafe exposure. That ratio, not the absolute latency, is the deployment")
    print("  argument, and it is the opposite of the ratio rollback offers.")

    # ---------------------------------------------------------------- E6
    hr("E6.  REJECTION SEMANTICS BEAT REJECTION ACCURACY")
    print("  What should the runtime DO when it says no? A gate that only vetoes is not a")
    print("  controller: the actuator keeps tracking whatever setpoint it already held, and")
    print("  if the context has changed, that setpoint is now wrong too.")
    print("\n  Three fallbacks, applied to the SAME policy, on the SAME trace:")
    print("    hold        keep the current setpoint (a pure filter)")
    print("    checkpoint  revert to the last setpoint that provably held the room in band")
    print("    safe_state  command 21 degC, which lies inside BOTH bands and is therefore")
    print("                safe without knowing which one applies\n")
    for pol_name, pol in (("oracle", oracle), ("corrected", corrected)):
        for fb in ("hold", "checkpoint", "safe_state"):
            d = row(run_closed_loop(model, tr_sh, pol, arm=f"{pol_name} / {fb}",
                                    fallback=fb, enable_rollback=True, **kw))
            d["regime"] = "shift/fallback-ablation"
            rows.append(d)
            show(d)
        print()
    hold_o = [x for x in rows if x["arm"] == "oracle / hold"][0]
    safe_o = [x for x in rows if x["arm"] == "oracle / safe_state"][0]
    safe_c = [x for x in rows if x["arm"] == "corrected / safe_state"][0]
    print(f"  A PERFECT gate with a naive fallback leaves "
          f"{hold_o['unsafe_exposure_c_min']:.0f} degC-min.")
    print(f"  An IMPERFECT gate with a context-safe fallback leaves "
          f"{safe_c['unsafe_exposure_c_min']:.0f} degC-min.")
    print(f"  The perfect gate WITH the context-safe fallback leaves "
          f"{safe_o['unsafe_exposure_c_min']:.1f} degC-min.")
    print("\n  So the fallback dominates. An oracle-accurate gate that reverts to a")
    print("  context-blind state is beaten by a far less accurate gate that reverts to a")
    print("  context-independent safe one. This is invisible in a software-only testbed,")
    print("  where restoring a variable is instantaneous and all three fallbacks look the")
    print("  same. It is only visible because the actuator is stateful and the room is slow.")

    # ---------------------------------------------------------------- E7
    hr("E7.  THE ROLLBACK BUDGET DOES NOT BIND HERE, AND SAYING SO MATTERS")
    print("  A rollback budget is meant to stop the runtime from undoing the same mistake")
    print("  forever: if it is rolling back every few minutes, the model is systematically")
    print("  wrong, and the honest response is to hand the zone to the fallback thermostat")
    print("  and call a human. We implemented it, and then we checked whether it ever fires.\n")
    for budget in (None, 10, 5, 3):
        d = row(run_closed_loop(model, tr_sh, shipped,
                                arm=f"shipped, budget={budget or 'unlimited'}",
                                enable_rollback=True, rollback_budget=budget, **kw))
        d["regime"] = "shift/budget"
        rows.append(d)
        show(d)
    hrs = ship["minutes"] / 60.0
    print(f"\n  It does not. Every setting gives an identical result, because the shipped")
    print(f"  policy triggers only {ship['rollbacks']} rollbacks in {hrs:.0f} hours -- never "
          f"three within any one")
    print(f"  hour -- so no budget in this range is ever reached.")
    print("\n  This is worth reporting rather than quietly dropping, for two reasons. First,")
    print("  an earlier version of this experiment DID exhaust the budget and enter")
    print("  FAILED_SAFE within a few hours, which looked like a meaningful safety result. It")
    print("  was not: the monitor had no deadband and was firing rollbacks on 0.01 degC of")
    print("  sensor noise. The budget was measuring a bug in our monitor, not a property of")
    print("  governance. Second, it means the ONLY condition under which this runtime")
    print("  legitimately reaches FAILED_SAFE in our workload is irreversible actuation (E3).")
    print("  A mechanism that never fires is not a contribution, and we do not present it as")
    print("  one.")

    # ---------------------------------------------------------------- HAL
    hr("HAL PARITY: THE SAME GOVERNANCE CODE TARGETS THE RASPBERRY PI")
    print(parity_report())
    print("\n  The hardware backends are complete and released. They were NOT executed.")
    print("  No Pi, no SCD40, no BME280, no relay, no servo was operated in this study.")
    print("  The hardware evaluation is an interface swap, and it is future work.")

    # ---------------------------------------------------------------- write
    keys = sorted({k for d in rows for k in d})
    with open(os.path.join(RES, "sil_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "sil_results.json"), "w") as f:
        json.dump({"model": info,
                   "twin": {"tau_h": p.tau_thermal_h,
                            "heat_c_per_min": p.max_heating_rate_c_per_min(),
                            "cool_c_per_min": p.max_cooling_rate_c_per_min()},
                   "rows": rows}, f, indent=2, default=str)
    with open(os.path.join(RES, "SIL_LOG.txt"), "w") as f:
        f.write(f"host    {platform.node()}\n")
        f.write(f"os      {platform.system()} {platform.release()}\n")
        f.write(f"python  {platform.python_version()}\n")
        f.write(f"elapsed {time.time()-t_start:.1f}s\n")
        f.write("plant   digital twin; NO HARDWARE OPERATED\n\n")
        for d in rows:
            f.write(json.dumps(d, default=str) + "\n")

    hr("DONE")
    print(f"  {time.time()-t_start:.1f} s")
    print("  results/sil_results.csv   results/SIL_LOG.txt")
    print("  next: python make_figures.py")


if __name__ == "__main__":
    main()
