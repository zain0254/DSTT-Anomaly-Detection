"""Phase 3 result figures from the primary results (results/phase3/).

fig02_test_metrics:   main test metrics per method, mean over seeds, sd error bars
fig03_paired_per_file: per-file P1 vs Base and P1 vs B3 (mean over seeds) for
                       composite F1 and FPR
fig04_example_trace:  nyc_taxi, seed 15 (the file used in the Phase 2 figure):
                       test-segment errors with Base, B3 and P1 thresholds and
                       alarms, NAB windows shaded
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import plotting as P  # noqa: E402
from pipeline.data import clip_windows, load_series, split_bounds  # noqa: E402
from pipeline.methods import METHODS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results" / "phase3"
PHASE = "phase3_thresholds"
METHOD_COLOR = dict(zip(METHODS, P.SERIES))          # fixed order, follows the method
TRACE_FILE, TRACE_SEED = "realKnownCause/nyc_taxi.csv", 15


def fig_metrics() -> None:
    s = pd.read_csv(RES / "test_summary.csv").set_index("method").reindex(list(METHODS))
    panels = [("composite_f1", "Composite F1 (higher better)"), ("f1", "Point F1 (higher better)"),
              ("event_recall", "Event recall (higher better)"), ("fpr", "FPR (lower better)"),
              ("hv_fpr", "High-variance FPR (lower better)"),
              ("clean_false_alarms_per_1000", "Clean-stream false alarms / 1,000 (lower better)")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey=True)
    y = np.arange(len(s))[::-1]
    for ax, (col, title) in zip(axes.flat, panels):
        ax.barh(y, s[f"{col}_mean"], xerr=s[f"{col}_sd"], height=0.62,
                color=[METHOD_COLOR[m] for m in s.index], error_kw={"lw": 0.8, "capsize": 2, "ecolor": P.INK})
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.grid(axis="y", visible=False)
        for yi, v, sd in zip(y, s[f"{col}_mean"], s[f"{col}_sd"]):
            ax.text(v + sd, yi, f"  {v:.2f}" if v < 10 else f"  {v:.0f}", va="center", fontsize=6, color=P.INK)
        ax.set_xlim(0, (s[f"{col}_mean"] + s[f"{col}_sd"]).max() * 1.3)
    labels = [m if m != "Base" else "Base (Ahmad 2026)" for m in s.index]
    labels = [lab if lab != "B3" else "B3 / RePAD2" for lab in labels]
    labels = [lab if lab != "R2" else "R2 EVT (Yu 2024)" for lab in labels]
    for ax in axes[:, 0]:
        ax.set_yticks(y, labels)
    fig.text(0.5, -0.01, "Test segments, 18 evaluation files (clean stream: 18 clean files). Bars: mean over 3 seeds "
             "of the across-file mean; error bars: sd over seeds. B3-k selected k = 3 and equals B3.",
             ha="center", fontsize=6.5, color=P.INK_2)
    fig.tight_layout()
    P.save(fig, PHASE, "fig02_test_metrics")


def fig_paired() -> None:
    t = pd.read_csv(RES / "test_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    for i, (metric, name, higher) in enumerate((("composite_f1", "Composite F1", True), ("fpr", "FPR", False))):
        pf = t.groupby(["method", "file"])[metric].mean().unstack("method")
        for j, other in enumerate(("Base", "B3")):
            ax = axes[i, j]
            x, yv = pf[other], pf["P1"]
            lim = (0, max(x.max(), yv.max()) * 1.05)
            ax.plot(lim, lim, color=P.INK_2, lw=0.8, ls="--", zorder=1)
            ax.scatter(x, yv, s=28, color=METHOD_COLOR["P1"], edgecolor=P.SURFACE, linewidth=1.2, zorder=3)
            wins = int(((yv > x) if higher else (yv < x)).sum())
            losses = int(((yv < x) if higher else (yv > x)).sum())
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            other_name = "Base (Ahmad 2026)" if other == "Base" else "B3 / RePAD2"
            ax.set_xlabel(f"{other_name} {name}")
            ax.set_ylabel(f"P1 {name}")
            better = "above" if higher else "below"
            ax.set_title(f"{name}: P1 vs {other_name}\nP1 better {wins}, worse {losses} of 18 ({better} = P1 better)",
                         loc="left", fontsize=7)
    fig.text(0.5, -0.01, "One point per evaluation file, mean over 3 seeds. Dashed: equal performance.",
             ha="center", fontsize=6.5, color=P.INK_2)
    fig.tight_layout()
    P.save(fig, PHASE, "fig03_paired_per_file")


def fig_trace() -> None:
    s = load_series(TRACE_FILE)
    _, _, te = split_bounds(len(s.values))
    z = np.load(RES / "test_traces" / f"{TRACE_FILE.replace('.csv', '').replace('/', '__')}__s{TRACE_SEED}.npz")
    ts = s.timestamps[te]
    wins = clip_windows(s.windows, te)
    fig, (ax, axa) = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(ts, z["errors"], color=P.INK_2, lw=0.6, label="reconstruction error (window MAE)")
    show = [("Base", "Base (Ahmad 2026)"), ("B3", "B3 / RePAD2 (w = 400)"), ("P1", "P1 (lambda = 6, w = 50, plain)")]
    for m, lab in show:
        ax.plot(ts, z[f"{m}__threshold"], color=METHOD_COLOR[m], lw=1.0, label=f"{lab} threshold")
    for k, (m, lab) in enumerate(show):
        a = z[f"{m}__alarms"]
        axa.scatter(ts[a], np.full(a.sum(), k), marker="|", s=40, color=METHOD_COLOR[m], linewidths=0.8)
    for a_ in (ax, axa):
        P.shade_windows(a_, ts, wins, label=None)
    axa.set_yticks(range(len(show)), [m if m != "Base" else "Base" for m, _ in show])
    axa.set_ylim(-0.6, len(show) - 0.4)
    axa.grid(axis="y", visible=False)
    axa.set_title("Alarms", loc="left", fontsize=7.5)
    ax.set_ylabel("Window MAE (scaled)")
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(color=P.ANOMALY, alpha=0.18)], l + ["NAB window"], loc="upper left", fontsize=6, ncol=2)
    ax.set_title(f"{TRACE_FILE.split('/')[1]}, seed {TRACE_SEED}, test segment", loc="left", fontsize=8)
    fig.tight_layout()
    P.save(fig, PHASE, "fig04_example_trace")


if __name__ == "__main__":
    P.apply_style()
    fig_metrics()
    fig_paired()
    fig_trace()
    print("written", P.FIG_DIR / PHASE)
