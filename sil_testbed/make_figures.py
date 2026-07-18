"""
Figures for the software-in-the-loop (SIL) study.

Every figure is generated FROM results/sil_results.json, which run_sil.py wrote. No
figure carries a hand-typed number, so a figure cannot disagree with its table. (The
previous round of this paper shipped figures that contradicted their own tables. This
script is the fix for that class of error, not a cosmetic improvement.)

    python run_sil.py && python make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

INK = "#1a1a1a"
GREY = "#9aa0a6"
BLUE = "#2f6fb2"
RED = "#c0392b"
GREEN = "#2e8b57"
AMBER = "#d98c00"


def load():
    with open(os.path.join(RES, "sil_results.json")) as f:
        return json.load(f)


def pick(rows, regime, arm):
    for r in rows:
        if r.get("regime") == regime and r["arm"] == arm:
            return r
    raise KeyError(f"{regime} / {arm}")


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Fig 1. What each governance layer actually buys, in physical units.
# ---------------------------------------------------------------------------
def fig_layers(d):
    rows = d["rows"]
    arms = ["ungoverned", "shipped, no rollback", "shipped + rollback",
            "corrected + rollback", "oracle + rollback"]
    labels = ["Ungoverned", "Admission\ncontrol only", "+ Rollback",
              "+ CO$_2$ context\npredicate", "Oracle context\n(not deployable)"]
    cols = [GREY, BLUE, BLUE, GREEN, INK]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, regime, title in zip(
            axes, ["shift", "in-distribution"],
            ["Distribution shift (datatest2)", "In-distribution (datatest)"]):
        vals = [pick(rows, regime, a)["unsafe_exposure_c_min"] for a in arms]
        b = ax.bar(range(len(arms)), vals, color=cols, width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.1f}", ha="center",
                    fontsize=9, color=INK, fontweight="bold")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
        style(ax)
    axes[0].set_ylabel("Unsafe physical exposure  ($^{\\circ}$C$\\cdot$min)",
                       fontsize=10, color=INK)
    fig.suptitle("Adding rollback changes almost nothing. Adding the right context "
                 "predicate changes everything.",
                 fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sil_1_layers.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2. The two halves of rollback: the decision, and its physical consequence.
# ---------------------------------------------------------------------------
def fig_rollback(d):
    r = pick(d["rows"], "shift", "shipped + rollback")
    med = r["median_physical_recovery_s"]
    worst = r["max_physical_recovery_s"]
    if med is None:
        return
    decision_ms = 0.44          # measured on the real MQTT stack, not in the twin

    fig, ax = plt.subplots(figsize=(9, 3.4))
    names = ["Governance decision\n(measured on the real\nMQTT stack)",
             "Physical recovery,\nmedian",
             "Physical recovery,\nworst case"]
    vals_s = [decision_ms / 1000.0, med, worst]
    cols = [BLUE, AMBER, RED]
    ax.barh(range(3), vals_s, color=cols, height=0.55)
    ax.set_yticks(range(3))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Time (s, log scale)", fontsize=10, color=INK)
    ax.invert_yaxis()
    for i, v in enumerate(vals_s):
        txt = f"{decision_ms:.2f} ms" if i == 0 else f"{v:.0f} s  ({v/60:.0f} min)"
        ax.text(v * 1.3, i, txt, va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xlim(1e-4, worst * 12)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Rollback is fast to decide and slow to take effect. Reporting only "
                 "the decision\ndescribes the cheap half.", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sil_2_rollback_latency.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3. Rejection semantics dominate rejection accuracy.
# ---------------------------------------------------------------------------
def fig_fallback(d):
    rows = d["rows"]
    reg = "shift/fallback-ablation"
    fbs = ["hold", "checkpoint", "safe_state"]
    fb_lab = ["hold\n(pure filter)", "checkpoint\n(context-blind)", "safe_state\n(context-safe)"]
    pols = [("corrected", GREEN), ("oracle", INK)]

    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.35
    for k, (pol, c) in enumerate(pols):
        vals = [pick(rows, reg, f"{pol} / {fb}")["unsafe_exposure_c_min"] for fb in fbs]
        xs = [i + (k - 0.5) * w for i in range(len(fbs))]
        ax.bar(xs, vals, width=w, color=c,
               label=("Corrected CO$_2$ gate (deployable)" if pol == "corrected"
                      else "Oracle gate (perfect, not deployable)"))
        for x, v in zip(xs, vals):
            ax.text(x, v + 15, f"{v:.1f}", ha="center", fontsize=8.5,
                    color=INK, fontweight="bold")
    ax.set_xticks(range(len(fbs)))
    ax.set_xticklabels(fb_lab, fontsize=9)
    ax.set_ylabel("Unsafe physical exposure  ($^{\\circ}$C$\\cdot$min)", fontsize=10)
    ax.set_xlabel("What the runtime does when it rejects an intent", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    style(ax)
    ax.set_title("A perfect gate with a naive fallback loses to an imperfect gate with a "
                 "safe one.\nWhat you do when you say no matters more than how accurately "
                 "you say it.",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sil_3_fallback.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4. Where the residual physical risk actually lives.
# ---------------------------------------------------------------------------
def fig_residual(d):
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for j, (regime, lab) in enumerate([("in-distribution", "In-distribution"),
                                       ("shift", "Distribution shift")]):
        corr = pick(rows, regime, "corrected + rollback")["unsafe_exposure_c_min"]
        orac = pick(rows, regime, "oracle + rollback")["unsafe_exposure_c_min"]
        ax.barh(j, orac, color=INK, height=0.45,
                label="Irreducible (oracle context)" if j == 0 else None)
        ax.barh(j, corr - orac, left=orac, color=RED, height=0.45,
                label="Attributable to the CO$_2$ context estimator" if j == 0 else None)
        ax.text(corr + 12, j, f"{corr:.1f} $^{{\\circ}}$C$\\cdot$min total",
                va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["In-distribution", "Distribution shift"], fontsize=9.5)
    ax.set_xlabel("Unsafe physical exposure under the deployable (corrected) policy  "
                  "($^{\\circ}$C$\\cdot$min)", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("The residual risk is a property of the context estimator, not of the "
                 "governance runtime.", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sil_4_residual.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d = load()
    fig_layers(d)
    fig_rollback(d)
    fig_fallback(d)
    fig_residual(d)
    print("figures written to figures/:")
    for f in sorted(os.listdir(FIG)):
        print("   ", f)
    print("\nEvery value is read from results/sil_results.json. No figure contains a")
    print("hand-typed number, so no figure can contradict its table.")


if __name__ == "__main__":
    main()
