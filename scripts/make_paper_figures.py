"""Paper versions of the figures, drawn at their exact print size.

The four PDF figures of the paper (the pipeline figure is TikZ in main.tex):
fig05_windows_vs_split, fig03_paired_per_file, fig04_example_trace and
fig05_category_improvement. Same saved data and plotting logic as
scripts/eda.py, plot_phase3_results.py and plot_proposal.py; only size,
layout and fonts differ. Output: paper/figures/*.pdf (figures/ is unchanged).

Print size: full width (figure*) 7.16 in. Fonts at
print size: axis labels and titles 8 pt, ticks and legends 7 pt, nothing below
6.5 pt; line widths >= 0.8 pt. The smallest font of each figure is printed and
written to paper/figures/font_report.csv. A PNG copy at 300 dpi is written to
--png-dir (default: not written) for visual checks.
"""
from __future__ import annotations

import argparse
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
from matplotlib.patches import Patch  # noqa: E402

OUT = ROOT / "paper" / "figures"
R3 = ROOT / "results" / "phase3"
FULL = 7.16
SMALL = 6.5                        # smallest allowed font (pt)
METHOD_COLOR = dict(zip(METHODS, P.SERIES))
NAMES = {"Base": "Base (Ahmad 2026)", "B3": "B3 / RePAD2", "R2": "R2 EVT (Yu 2024)"}
PNG_DIR = None
REPORT = []


def style() -> None:
    P.apply_style()
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "lines.linewidth": 1.0, "axes.linewidth": 0.8, "grid.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "patch.linewidth": 0.8,
        "legend.frameon": True, "legend.framealpha": 1.0, "legend.facecolor": P.SURFACE, "legend.edgecolor": P.SURFACE,
    })


def min_font(fig) -> float:
    sizes = [t.get_fontsize() for t in fig.findobj(lambda a: hasattr(a, "get_fontsize") and hasattr(a, "get_text"))
             if getattr(a := t, "get_text")() and a.get_visible()]
    return min(sizes)


def save(fig, name: str) -> None:
    f = min_font(fig)
    REPORT.append({"figure": name, "width_in": fig.get_figwidth(), "height_in": fig.get_figheight(), "min_font_pt": f})
    if f < SMALL - 1e-9:
        raise SystemExit(f"{name}: font {f} pt below {SMALL} pt")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")          # exact size: no bbox_inches="tight"
    if PNG_DIR:
        fig.savefig(Path(PNG_DIR) / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"{name}: {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in, smallest font {f:g} pt")


def note(fig, text: str, y: float = 0.005) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=SMALL, color=P.INK_2)


# ------------------------------------------------------------ Phase 1: windows vs split
def fig_windows_vs_split() -> None:
    eda = pd.read_csv(ROOT / "results/phase1_eda/file_summary.csv")
    wp = pd.read_csv(ROOT / "results/phase1_eda/window_parts.csv")
    s = eda.set_index("file")
    # All labelled files (same set and order as figures/phase1_eda), split into two halves side by side.
    files = eda[eda.n_windows > 0].sort_values(["category", "file"]).file.tolist()
    half = (len(files) + 1) // 2
    fig = plt.figure(figsize=(FULL, 4.5))
    # two panels: names on the left; the mark starts inside the (empty) test zone and runs into the margin
    axes = [fig.add_axes([0.262, 0.135, 0.15, 0.825]), fig.add_axes([0.753, 0.135, 0.15, 0.825])]
    for ax, part in zip(axes, (files[:half], files[half:])):
        n = len(part)
        for i, f in enumerate(part):
            yy = n - 1 - i
            ax.plot([0, 1], [yy, yy], color=P.GRID, lw=0.8, zorder=1)
            for w in wp[wp.file == f].itertuples():
                ax.plot([w.start_rel, w.end_rel], [yy, yy], color=P.ANOMALY, lw=3.0, solid_capstyle="butt", zorder=2)
            if s.loc[f, "windows_test"] == 0:
                ax.text(0.815, yy, "[no test window]", va="center", ha="left", fontsize=SMALL, color=P.INK_2,
                        clip_on=False, zorder=3)
        ax.axvline(0.6, color=P.INK_2, lw=0.8, ls="--")
        ax.axvline(0.8, color=P.INK_2, lw=0.8, ls="--")
        ax.set_yticks(range(n)[::-1], [f.split("/")[1].replace(".csv", "") for f in part], fontsize=SMALL)
        for tick, f in zip(ax.get_yticklabels(), part):
            tick.set_color(P.CAT_COLOR[s.loc[f, "category"]] if s.loc[f, "windows_test"] > 0 else P.INK_2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xticks([0, 0.6, 0.8, 1.0], ["0", "0.6", "0.8", "1"])
        ax.set_xlabel("Relative position in file")
        ax.grid(axis="y", visible=False)
        for x0, x1, name in ((0, 0.6, "train"), (0.6, 0.8, "val"), (0.8, 1.0, "test")):
            ax.text((x0 + x1) / 2, n - 0.25, name, ha="center", va="bottom", fontsize=7, color=P.INK_2)
    note(fig, f"All {len(files)} series with at least one NAB window, in two columns. Coloured names have a test-set window;\n"
              "grey names are marked [no test window].", y=0.0)
    save(fig, "fig05_windows_vs_split")


# ------------------------------------------------------------ Phase 3: test metrics

def fig_paired() -> None:
    t = pd.read_csv(R3 / "test_metrics.csv")
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.55))
    k = 0
    for metric, name, higher in (("composite_f1", "Composite F1", True), ("fpr", "FPR", False)):
        pf = t.groupby(["method", "file"])[metric].mean().unstack("method")
        for other in ("Base", "B3"):
            ax = axes[k]
            k += 1
            x, yv = pf[other], pf["P1"]
            lim = (0, max(x.max(), yv.max()) * 1.05)
            ax.plot(lim, lim, color=P.INK_2, lw=0.8, ls="--", zorder=1)
            ax.scatter(x, yv, s=14, color=METHOD_COLOR["P1"], edgecolor=P.SURFACE, linewidth=0.8, zorder=3)
            wins = int(((yv > x) if higher else (yv < x)).sum())
            losses = int(((yv < x) if higher else (yv > x)).sum())
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{NAMES.get(other, other)}")
            ax.set_ylabel(f"P1 {name}")
            ax.set_title(f"{name}: P1 vs {other}\nbetter {wins}, worse {losses} of 18", loc="center")
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0.025, 0.06, 0.965, 1))
    note(fig, "One point per evaluation file, mean over 3 seeds. Dashed: equal. P1 better = above the diagonal "
              "for composite F1, below it for FPR.")
    save(fig, "fig03_paired_per_file")


def fig_trace() -> None:
    name, seed = "realKnownCause/nyc_taxi.csv", 15
    s = load_series(name)
    _, _, te = split_bounds(len(s.values))
    z = np.load(R3 / "test_traces" / f"{name.replace('.csv', '').replace('/', '__')}__s{seed}.npz")
    ts = s.timestamps[te]
    wins = clip_windows(s.windows, te)
    fig, (ax, axa) = plt.subplots(2, 1, figsize=(FULL, 3.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(ts, z["errors"], color=P.INK_2, lw=0.8, label="reconstruction error (window MAE)")
    show = [("Base", "Base (Ahmad 2026)"), ("B3", "B3 / RePAD2 (w = 400)"), ("P1", "P1 (lambda = 6, w = 50, plain)")]
    for m, lab in show:
        ax.plot(ts, z[f"{m}__threshold"], color=METHOD_COLOR[m], lw=1.0, label=f"{lab} threshold")
    for k, (m, _) in enumerate(show):
        a = z[f"{m}__alarms"]
        axa.scatter(ts[a], np.full(a.sum(), k), marker="|", s=30, color=METHOD_COLOR[m], linewidths=0.8)
    for a_ in (ax, axa):
        P.shade_windows(a_, ts, wins, label=None)
    axa.set_yticks(range(len(show)), [m for m, _ in show])
    axa.set_ylim(-0.6, len(show) - 0.4)
    axa.grid(axis="y", visible=False)
    axa.set_title("Alarms", loc="left")
    ax.set_ylabel("Window MAE (scaled)")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(color=P.ANOMALY, alpha=0.18)], lab + ["NAB window"], loc="upper left", ncol=2)
    ax.set_title(f"{name.split('/')[1]}, seed {seed}, test segment", loc="left")
    fig.tight_layout(pad=0.3, h_pad=0.5)
    save(fig, "fig04_example_trace")


def fig_categories() -> None:
    ca = pd.read_csv(R3 / "category_analysis.csv")
    cats = [c for c in P.CATEGORIES if c in set(ca.category)]
    panels = [("f1", "Point F1"), ("composite_f1", "Composite F1"), ("hv_fpr", "High-variance FPR (reduction)")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.7), sharey=True)
    y = np.arange(len(cats))[::-1]
    h = 0.36
    for ax, (metric, title) in zip(axes, panels):
        for j, b in enumerate(["Base", "B3"]):
            d = ca[(ca.metric == metric) & (ca.baseline == b)].set_index("category").reindex(cats)
            ax.barh(y + (0.5 - j) * h, d.mean_improvement, height=h, color=METHOD_COLOR[b], label=f"P1 vs {NAMES[b]}")
        ax.axvline(0, color=P.INK_2, lw=0.8)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean improvement of P1")
    n = ca[(ca.metric == "f1") & (ca.baseline == "Base")].set_index("category").n_files
    nh = ca[(ca.metric == "hv_fpr") & (ca.baseline == "Base")].set_index("category").n_files
    short = {"realAWSCloudwatch": "AWS", "realAdExchange": "AdExchange", "realKnownCause": "KnownCause",
             "realTraffic": "Traffic", "realTweets": "Tweets"}
    axes[0].set_yticks(y, [f"{short[c]} (n={n[c]}" + (f", HV n={nh[c]})" if nh[c] != n[c] else ")") for c in cats])
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 0.9))
    note(fig, "Positive = P1 better (higher F1, lower high-variance FPR). Mean over files in the category, each "
              "averaged over 3 seeds. Descriptive only.")
    save(fig, "fig05_category_improvement")


def main() -> None:
    global PNG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", default=None)
    PNG_DIR = ap.parse_args().png_dir
    if PNG_DIR:
        Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
    style()
    fig_windows_vs_split()
    fig_paired()
    fig_trace()
    fig_categories()
    pd.DataFrame(REPORT).to_csv(OUT / "font_report.csv", index=False)


if __name__ == "__main__":
    main()
