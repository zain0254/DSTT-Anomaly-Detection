"""Figures for the proposal analyses (D-28), from results/phase3/.

fig05_category_improvement: mean improvement of P1 over Base and over B3 per
  NAB category (point F1, composite F1, high-variance FPR); positive = P1 better.
fig06_detection_delay: mean relative detection delay over detected test
  windows, and missed test windows (of 25), per method; mean +/- sd over seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import plotting as P  # noqa: E402
from pipeline.methods import METHODS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results" / "phase3"
PHASE = "phase3_thresholds"
METHOD_COLOR = dict(zip(METHODS, P.SERIES))
NAMES = {"Base": "Base (Ahmad 2026)", "B3": "B3 / RePAD2", "R2": "R2 EVT (Yu 2024)"}


def fig_categories() -> None:
    ca = pd.read_csv(RES / "category_analysis.csv")
    cats = [c for c in P.CATEGORIES if c in set(ca.category)]
    panels = [("f1", "Point F1"), ("composite_f1", "Composite F1"), ("hv_fpr", "High-variance FPR (reduction)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    y = np.arange(len(cats))[::-1]
    h = 0.36
    for ax, (metric, title) in zip(axes, panels):
        for j, b in enumerate(["Base", "B3"]):
            d = ca[(ca.metric == metric) & (ca.baseline == b)].set_index("category").reindex(cats)
            ax.barh(y + (0.5 - j) * h, d.mean_improvement, height=h, color=METHOD_COLOR[b],
                    label=f"P1 vs {NAMES[b]}")
        ax.axvline(0, color=P.INK_2, lw=0.8)
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean improvement of P1")
    n = ca[(ca.metric == "f1") & (ca.baseline == "Base")].set_index("category").n_files
    nh = ca[(ca.metric == "hv_fpr") & (ca.baseline == "Base")].set_index("category").n_files
    short = {"realAWSCloudwatch": "AWS", "realAdExchange": "AdExchange", "realKnownCause": "KnownCause",
             "realTraffic": "Traffic", "realTweets": "Tweets"}
    labels = [f"{short[c]} (n={n[c]}" + (f", HV n={nh[c]})" if nh[c] != n[c] else ")") for c in cats]
    axes[0].set_yticks(y, labels)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    for ax in axes:
        ax.set_xlabel("Mean improvement of P1", fontsize=7)
    fig.text(0.5, 0.0, "Positive = P1 better (higher F1, lower high-variance FPR). Mean over files in the category, "
             "each file averaged over 3 seeds. Descriptive only.", ha="center", fontsize=6.5, color=P.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    P.save(fig, PHASE, "fig05_category_improvement")


def fig_delay() -> None:
    d = pd.read_csv(RES / "detection_delay.csv")
    g = d.groupby("method")[["mean_relative_delay", "missed"]].agg(["mean", "std"]).reindex(list(METHODS))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    y = np.arange(len(g))[::-1]
    cols = [METHOD_COLOR[m] for m in g.index]
    ek = {"lw": 0.8, "capsize": 2, "ecolor": P.INK}
    a1.barh(y, g[("mean_relative_delay", "mean")], xerr=g[("mean_relative_delay", "std")], color=cols, height=0.62,
            error_kw=ek)
    a2.barh(y, g[("missed", "mean")], xerr=g[("missed", "std")], color=cols, height=0.62, error_kw=ek)
    for ax, key, fmt in ((a1, "mean_relative_delay", "{:.2f}"), (a2, "missed", "{:.1f}")):
        for yi, v, sd in zip(y, g[(key, "mean")], g[(key, "std")].fillna(0)):
            ax.text(v + sd, yi, "  " + fmt.format(v), va="center", fontsize=6.5, color=P.INK)
        ax.grid(axis="y", visible=False)
    a1.set_xlim(0, 0.65)
    a2.set_xlim(0, 9)
    a1.set_yticks(y, [NAMES.get(m, m) for m in g.index])
    a1.set_title("Relative delay of first alarm (detected windows)", loc="left", fontsize=8)
    a1.set_xlabel("(first alarm - window start) / window length")
    a2.set_title("Missed test windows (of 25)", loc="left", fontsize=8)
    a2.set_xlabel("Windows without any alarm")
    fig.text(0.5, -0.02, "Test windows of the 18 evaluation files, clipped to the test segment. Mean over 3 seeds; "
             "error bars: sd over seeds.", ha="center", fontsize=6.5, color=P.INK_2)
    fig.tight_layout()
    P.save(fig, PHASE, "fig06_detection_delay")


if __name__ == "__main__":
    P.apply_style()
    fig_categories()
    fig_delay()
    print("written", P.FIG_DIR / PHASE)
