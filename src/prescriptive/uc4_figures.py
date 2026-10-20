"""Figures for UC4. Reads the CSV/JSON outputs of uc4_prescriptive.py.

python uc4_figures.py --res results/uc4
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BLUE, BLUE_LIGHT, GREY, GREY_LIGHT, ORANGE = "#2a78d6", "#a9c8ef", "#8a8986", "#d6d5d1", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150,
})


def panel_label(ax, s):
    ax.text(-0.13, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")


def fig_model_and_threshold(res: Path):
    cal = pd.read_csv(res / "calibration_deciles.csv")
    dca = pd.read_csv(res / "decision_curve.csv")
    risk = pd.read_csv(res / "crossfitted_risk.csv")
    summ = json.load(open(res / "summary.json"))

    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[3.2, 1], hspace=0.08, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); axh = fig.add_subplot(gs[1, 0], sharex=ax)
    lim = (0.0, 0.32)
    ax.plot(lim, lim, color=GREY_LIGHT, lw=1.2, zorder=1)
    ax.errorbar(cal.pred, cal.obs, yerr=1.96 * cal.se, fmt="o", ms=5, color=BLUE, ecolor=BLUE_LIGHT,
                elinewidth=1.4, capsize=0, zorder=3, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_ylabel("Observed mortality")
    ax.set_yticks([0, 0.1, 0.2, 0.3]); ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.tick_params(labelbottom=False)
    ax.set_title("Calibration", loc="left", color=INK)
    ax.text(0.02, 0.29, f"AUC {summ['auc_crossfit']:.2f} ({summ['auc_crossfit_ci'][0]:.2f}–{summ['auc_crossfit_ci'][1]:.2f})\n"
            f"slope {summ['calibration_slope']:.2f}", fontsize=8.5, color=INK2, va="top")
    axh.hist(risk.p, bins=np.arange(0, 0.33, 0.01), color=GREY_LIGHT, edgecolor="white", linewidth=0.6)
    axh.set_yticks([]); axh.spines["left"].set_visible(False)
    axh.set_xticks([0, 0.1, 0.2, 0.3]); axh.set_xticklabels(["0%", "10%", "20%", "30%"])
    axh.set_xlabel("Predicted risk of in-hospital death")
    panel_label(ax, "A")

    ax2 = fig.add_subplot(gs[:, 1])
    d = dca[dca.threshold <= 0.35]
    lo, hi = summ["dca_ci_range"]
    ax2.axvspan(lo, hi, color=BLUE, alpha=0.06, lw=0, label=f"Model better ({lo:.0%}–{hi:.0%})")
    ax2.axhline(0, color=GREY, lw=1.2, ls=(0, (1, 2)))
    ax2.plot(d.threshold, d.nb_all.clip(lower=-0.06), color=ORANGE, lw=1.6, ls=(0, (4, 2)), label="Monitor all")
    ax2.fill_between(d.threshold, d.nb_model_lo, d.nb_model_hi, color=BLUE, alpha=0.18, lw=0)
    ax2.plot(d.threshold, d.nb_model, color=BLUE, lw=2.2, label="Risk model (95% CI)")
    ax2.set_ylim(-0.03, 0.15); ax2.set_xlim(0.05, 0.35)
    ax2.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]); ax2.set_xticklabels([f"{t:.0%}" for t in ax2.get_xticks()])
    ax2.set_xlabel("Risk threshold for HDU care"); ax2.set_ylabel("Net benefit")
    ax2.set_title("Decision curve", loc="left", color=INK)
    ax2.grid(axis="y", color=GRID, lw=0.7); ax2.set_axisbelow(True)
    ax2.plot([], [], color=GREY, lw=1.2, ls=(0, (1, 2)), label="Monitor none")
    h, l = ax2.get_legend_handles_labels(); o = [l.index("Risk model (95% CI)"), l.index("Monitor all"), l.index("Monitor none"), [i for i, t in enumerate(l) if t.startswith("Model better")][0]]
    ax2.legend([h[i] for i in o], [l[i] for i in o], loc="upper right", fontsize=8.5, handlelength=2.2)
    panel_label(ax2, "B")
    fig.savefig(res / "fig_uc4_model_threshold.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_policies(res: Path):
    ps = pd.read_csv(res / "policy_summary.csv").set_index("policy")
    order = ["Oracle", "CP-SAT", "Risk ranking", "CP-SAT + equity", "Threshold 20%", "FCFS"]
    labels = {"Oracle": "Oracle (knows outcome)", "CP-SAT": "CP-SAT optimal", "Risk ranking": "Highest risk first",
              "CP-SAT + equity": "CP-SAT + equity rule", "Threshold 20%": "Threshold ≥ 20%", "FCFS": "First come, first served"}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(order))[::-1]
    ref = ps.loc["FCFS", "deaths_in_HDU_per_1000"]
    ax.axvline(ref, color=GREY, lw=1, ls=(0, (3, 3)), zorder=1)
    for yi, pol in zip(y, order):
        v = ps.loc[pol, "deaths_in_HDU_per_1000"]
        lo = ref + ps.loc[pol, "vs_FCFS_lo"]; hi = ref + ps.loc[pol, "vs_FCFS_hi"]
        emph = pol in ("CP-SAT", "Risk ranking", "CP-SAT + equity")
        col = BLUE if emph else GREY
        if pol == "Oracle":
            ax.plot([v], [yi], "o", ms=8, mfc="white", mec=GREY, mew=1.6, zorder=3)
        else:
            if pol != "FCFS":
                ax.plot([lo, hi], [yi, yi], color=BLUE_LIGHT if emph else GREY_LIGHT, lw=4, solid_capstyle="round", zorder=2)
            ax.plot([v], [yi], "o", ms=8, color=col, mec="white", mew=1.0, zorder=3)
        delta = ps.loc[pol, "vs_FCFS"]
        txt = f"{v:.1f}" if pol == "FCFS" else f"{v:.1f}  (+{delta:.1f})"
        ax.text(v + 2.6, yi, txt, va="center", fontsize=8.8, color=INK if emph else INK2)
        ax.text(124, yi, f"{ps.loc[pol, 'under65_share_%']:.0f}%", va="center", ha="right", fontsize=8.8, color=INK2)
    ax.text(124, len(order) - 0.35, "< 65 y in HDU", ha="right", fontsize=8.5, color=INK2, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([labels[o] for o in order]); ax.tick_params(axis="y", length=0)
    for t, pol in zip(ax.get_yticklabels(), order):
        t.set_color(INK if pol in ("CP-SAT", "Risk ranking", "CP-SAT + equity") else INK2)
    ax.set_xlim(40, 125); ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlabel("Patients placed in the HDU who died in hospital, per 1,000 sepsis admissions")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_title("Allocation policies, 200 simulated 90-day periods (6 HDU beds, 8 admissions/day)", loc="left", color=INK)
    fig.savefig(res / "fig_uc4_policies.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_sensitivity(res: Path):
    s = pd.read_csv(res / "sensitivity_capacity_demand.csv")
    m = s.pivot(index="hdu_beds", columns="admissions_per_day", values="cp_minus_fcfs_per_1000").sort_index(ascending=False)
    cmap = LinearSegmentedColormap.from_list("b", ["#eef4fc", "#2a78d6", "#123f75"])
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    im = ax.imshow(m.values, cmap=cmap, vmin=0, vmax=12, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"+{v:.1f}", ha="center", va="center", fontsize=9.5, color="white" if v > 6.5 else INK)
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index)
    ax.set_xlabel("Sepsis admissions per day"); ax.set_ylabel("HDU beds")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Gain of CP-SAT over FCFS\n(per 1,000 admissions)", loc="left", color=INK)
    fig.savefig(res / "fig_uc4_sensitivity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--res", type=Path, default=Path("results/uc4"))
    r = ap.parse_args().res
    fig_model_and_threshold(r); fig_policies(r); fig_sensitivity(r)
    print("figures written to", r)
