"""Phase 1a EDA of the NAB corpus. Uses labels only through window positions;
no model output.

Outputs
  results/phase1_eda/file_summary.csv    one row per file
  results/phase1_eda/window_parts.csv    one row per NAB window: which split parts it touches
  results/phase1_eda/category_summary.csv
  results/phase1_eda/selection_candidates.csv  proposal inputs (not final)
  figures/phase1_eda/*.png|pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data import list_files, load_series, split_bounds  # noqa: E402
from pipeline import plotting as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "results" / "phase1_eda"
PHASE = "phase1_eda"
PARTS = ("train", "val", "test")


def file_row(s) -> tuple[dict, list[dict]]:
    n = len(s.values)
    ts = s.timestamps
    d = np.diff(ts.values).astype("timedelta64[s]").astype(np.int64)
    step = int(np.median(d)) if len(d) else 0
    gaps = d[d > step]
    missing = int(np.sum(np.round(gaps / step) - 1)) if step else 0
    dup = int(ts.duplicated().sum())
    y = s.labels
    bounds = dict(zip(PARTS, split_bounds(n)))
    row = {
        "file": s.name, "category": s.category, "n": n,
        "start": ts[0], "end": ts[-1], "step_s": step,
        "n_irregular_steps": int(np.sum(d != step)), "n_missing_steps": missing,
        "n_duplicate_ts": dup, "n_nan_values": int(np.isnan(s.values).sum()),
        "n_windows": len(s.windows), "pct_labelled": 100 * y.mean(),
        "train_std": float(np.std(s.values[bounds["train"]])),
        "train_n_unique": int(len(np.unique(s.values[bounds["train"]]))),
    }
    wrows = []
    for k, (a, b) in enumerate(s.windows):
        touch = {p: max(a, sl.start) <= min(b, sl.stop - 1) for p, sl in bounds.items()}
        wrows.append({"file": s.name, "window": k, "start_idx": a, "end_idx": b,
                      "width": b - a + 1, "start_rel": a / n, "end_rel": b / n, **touch,
                      "n_parts": sum(touch.values())})
    for p, sl in bounds.items():
        row[f"windows_{p}"] = sum(w[p] for w in wrows)
        row[f"labelled_{p}"] = int(y[sl].sum())
        row[f"len_{p}"] = sl.stop - sl.start
    row["windows_spanning"] = sum(w["n_parts"] > 1 for w in wrows)
    row["windows_span_val_test"] = sum(w["val"] and w["test"] for w in wrows)
    row["windows_span_train_val"] = sum(w["train"] and w["val"] for w in wrows)
    row["pct_train_contaminated"] = 100 * row["labelled_train"] / row["len_train"]
    return row, wrows


def fig_examples(series: dict, summary: pd.DataFrame) -> None:
    """One series per category: the file with most windows (ties: first name)."""
    picks = []
    for cat in P.CATEGORIES:
        sub = summary[summary.category == cat].sort_values(["n_windows", "file"], ascending=[False, True])
        picks.append(sub.iloc[0]["file"])
    fig, axes = plt.subplots(len(picks), 1, figsize=(7.2, 1.25 * len(picks)))
    for ax, name in zip(axes, picks):
        s = series[name]
        n = len(s.values)
        ax.plot(s.timestamps, s.values, color=P.CAT_COLOR[s.category], lw=0.6)
        P.shade_windows(ax, s.timestamps, s.windows)
        tr, va, _ = split_bounds(n)
        for b in (va.start, va.stop):
            ax.axvline(s.timestamps[b], color=P.INK_2, lw=0.7, ls="--")
        ax.set_title(f"{s.category} / {name.split('/')[1]}", loc="left", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=P.ANOMALY, alpha=0.18, label="NAB window")], loc="upper right")
    fig.text(0.5, -0.005, "Dashed lines: 60 % and 80 % split boundaries. Shaded: NAB anomaly windows.",
             ha="center", fontsize=7, color=P.INK_2)
    fig.tight_layout()
    P.save(fig, PHASE, "fig01_example_per_category")


def fig_label_share(cat: pd.DataFrame, total_pct: float) -> None:
    labels = ["All files"] + cat["category"].tolist()
    pct = [total_pct] + cat["pct_labelled"].tolist()
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, pct, color=[P.INK_2] + [P.CAT_COLOR[c] for c in cat["category"]], height=0.6)
    ax.barh(y, [100 - p for p in pct], left=pct, color=P.GRID, height=0.6)
    for yi, p in zip(y, pct):
        ax.text(p + 1, yi, f"{p:.1f} %", va="center", fontsize=7, color=P.INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of points inside NAB windows (%); grey = unlabelled")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    P.save(fig, PHASE, "fig02_labelled_share")


def fig_distributions(series: dict, summary: pd.DataFrame) -> None:
    """Values min-max scaled per file over the whole file (EDA only), one box
    per file, grouped and coloured by category."""
    order = summary.sort_values(["category", "file"])
    data = []
    for name in order.file:
        v = series[name].values
        span = v.max() - v.min()
        data.append((v - v.min()) / span if span > 0 else np.zeros_like(v))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops={"color": P.INK, "lw": 0.8}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
    for patch, c in zip(bp["boxes"], order.category):
        patch.set_facecolor(P.CAT_COLOR[c])
        patch.set_edgecolor(P.CAT_COLOR[c])
        patch.set_alpha(0.8)
    ax.set_xticks([])
    ax.set_ylabel("Value, min-max scaled per file")
    ax.set_xlabel("Files, grouped by category (whiskers 1.5 IQR, outliers hidden)")
    pos = 1
    for c in P.CATEGORIES:
        k = int((order.category == c).sum())
        if pos > 1:
            ax.axvline(pos - 0.5, color=P.GRID, lw=0.8)
        pos += k
    ax.set_ylim(-0.03, 1.03)
    handles = [plt.Rectangle((0, 0), 1, 1, color=P.CAT_COLOR[c]) for c in P.CATEGORIES]
    ax.legend(handles, P.CATEGORIES, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    P.save(fig, PHASE, "fig03_value_distributions")


def fig_lengths(summary: pd.DataFrame) -> None:
    order = summary.sort_values(["category", "n"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(order))
    a1.bar(x, order.n, color=[P.CAT_COLOR[c] for c in order.category], width=0.8)
    a1.set_xticks([])
    a1.set_ylabel("Points per file")
    a1.set_xlabel("Files, grouped by category")
    a1.grid(axis="x", visible=False)
    steps = order.groupby("category")["step_s"].agg(lambda s: sorted(set((s / 60).round(1))))
    cats = [c for c in P.CATEGORIES if c in steps.index]
    a2.axis("off")
    rows = [[c.replace("artificial", "art").replace("real", ""), ", ".join(f"{v:g}" for v in steps[c])]
            for c in cats]
    tab = a2.table(cellText=rows, colLabels=["Category", "Median step (min)"], loc="center",
                   cellLoc="left", colLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.5)
    tab.scale(1, 1.3)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(P.GRID)
        if r > 0 and c == 0:
            cell.get_text().set_color(P.CAT_COLOR[cats[r - 1]])
    handles = [plt.Rectangle((0, 0), 1, 1, color=P.CAT_COLOR[c]) for c in cats]
    a1.legend(handles, [c for c in cats], loc="upper left", fontsize=6, ncol=2)
    fig.tight_layout()
    P.save(fig, PHASE, "fig04_lengths_and_sampling")


def fig_split_timeline(summary: pd.DataFrame, wparts: pd.DataFrame) -> None:
    """Every file with at least one window: window positions relative to the
    60/20/20 split. Files without a test-set window are marked."""
    files = summary[summary.n_windows > 0].sort_values(["category", "file"]).file.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 0.17 * len(files) + 0.9))
    for i, f in enumerate(files):
        yy = len(files) - 1 - i
        ax.plot([0, 1], [yy, yy], color=P.GRID, lw=0.8, zorder=1)
        for w in wparts[wparts.file == f].itertuples():
            ax.plot([w.start_rel, w.end_rel], [yy, yy], color=P.ANOMALY, lw=3.2,
                    solid_capstyle="butt", zorder=2)
    ax.axvline(0.6, color=P.INK_2, lw=0.8, ls="--")
    ax.axvline(0.8, color=P.INK_2, lw=0.8, ls="--")
    s = summary.set_index("file")
    labels = []
    for f in files:
        mark = "" if s.loc[f, "windows_test"] > 0 else "  [no test window]"
        labels.append(f.split("/")[1].replace(".csv", "") + mark)
    ax.set_yticks(range(len(files))[::-1], labels, fontsize=5.5)
    for tick, f in zip(ax.get_yticklabels(), files):
        tick.set_color(P.CAT_COLOR[s.loc[f, "category"]] if s.loc[f, "windows_test"] > 0 else P.INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(files) - 0.2)
    ax.set_xlabel("Relative position in file")
    for x0, x1, name in ((0, 0.6, "train 60 %"), (0.6, 0.8, "val 20 %"), (0.8, 1.0, "test 20 %")):
        ax.text((x0 + x1) / 2, len(files) - 0.1, name, ha="center", va="bottom", fontsize=7, color=P.INK_2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    P.save(fig, PHASE, "fig05_windows_vs_split")


def fig_window_parts(summary: pd.DataFrame) -> None:
    """Per file: windows touching train / val / test, and training contamination."""
    d = summary[summary.n_windows > 0].sort_values(["category", "file"])
    names = [f.split("/")[1].replace(".csv", "") for f in d.file]
    y = np.arange(len(d))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 0.17 * len(d) + 0.9), sharey=True,
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    h = 0.26
    for j, p in enumerate(PARTS):
        a1.barh(y + (1 - j) * h, d[f"windows_{p}"], height=h, color=P.PART_COLOR[p], label=p)
    a1.set_yticks(y, names, fontsize=5.5)
    a1.set_xlabel("NAB windows touching the part")
    a1.legend(loc="lower right", ncol=3)
    a1.grid(axis="y", visible=False)
    a2.barh(y, d.pct_train_contaminated, height=0.6, color=P.PART_COLOR["train"])
    a2.set_xlabel("Training points inside NAB windows (%)")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    P.save(fig, PHASE, "fig06_windows_per_part_and_contamination")


def main() -> None:
    P.apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    series, rows, wrows = {}, [], []
    for name in list_files():
        s = load_series(name)
        series[name] = s
        r, w = file_row(s)
        rows.append(r)
        wrows += w
    summary = pd.DataFrame(rows)
    wparts = pd.DataFrame(wrows)
    summary.to_csv(OUT / "file_summary.csv", index=False)
    wparts.to_csv(OUT / "window_parts.csv", index=False)

    cat = summary.groupby("category").agg(
        n_files=("file", "size"), n_points=("n", "sum"), n_windows=("n_windows", "sum"),
        labelled=("labelled_train", "sum")).reset_index()
    lab_tot = summary.assign(lab=summary.pct_labelled * summary.n / 100).groupby("category")["lab"].sum()
    cat["n_labelled"] = cat.category.map(lab_tot).round().astype(int)
    cat["pct_labelled"] = 100 * cat.n_labelled / cat.n_points
    for p in PARTS:
        cat[f"windows_{p}"] = cat.category.map(summary.groupby("category")[f"windows_{p}"].sum())
    cat = cat.drop(columns="labelled")
    cat["category"] = pd.Categorical(cat.category, P.CATEGORIES, ordered=True)
    cat = cat.sort_values("category")
    cat["category"] = cat.category.astype(str)
    cat.to_csv(OUT / "category_summary.csv", index=False)
    total_pct = 100 * cat.n_labelled.sum() / cat.n_points.sum()

    # Proposal inputs: label positions only.
    sel = summary[["file", "category", "n", "n_windows", "windows_train", "windows_val", "windows_test",
                   "windows_span_train_val", "windows_span_val_test", "labelled_train",
                   "pct_train_contaminated", "labelled_val", "labelled_test", "train_std", "train_n_unique"]].copy()
    sel["has_val_window"] = sel.windows_val > 0
    sel["has_test_window"] = sel.windows_test > 0
    sel.to_csv(OUT / "selection_candidates.csv", index=False)

    fig_examples(series, summary)
    fig_label_share(cat, total_pct)
    fig_distributions(series, summary)
    fig_lengths(summary)
    fig_split_timeline(summary, wparts)
    fig_window_parts(summary)

    print(f"files {len(summary)}, windows {int(summary.n_windows.sum())}, points {int(summary.n.sum())}, "
          f"labelled {total_pct:.2f} %")
    print(cat.to_string(index=False))


if __name__ == "__main__":
    main()
