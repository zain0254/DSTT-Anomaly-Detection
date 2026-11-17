"""Phase 3 tuning figure from results/phase3/tuning_summary.csv: validation
objective (mean composite F1 over 24 tuning files x 3 seeds) against each
method's parameters. Selected parameter sets are ringed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import plotting as P  # noqa: E402
from pipeline.methods import WS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PHASE = "phase3_thresholds"


def main() -> None:
    P.apply_style()
    summ = pd.read_csv(ROOT / "results" / "phase3" / "tuning_summary.csv")
    sel = pd.read_csv(ROOT / "results" / "phase3" / "selected_params.csv")
    summ["p"] = summ.params.map(json.loads)
    chosen = dict(zip(sel.method, sel.params))
    base = float(summ.loc[summ.method == "Base", "composite_f1"].iloc[0])
    r2 = float(summ.loc[summ.method == "R2", "composite_f1"].iloc[0])

    def sub(m):
        return summ[summ.method == m]

    def ring(ax, x, y):
        ax.plot([x], [y], "o", ms=9, mfc="none", mec=P.INK, mew=1.2, zorder=5)

    def refs(ax):
        ax.axhline(base, color=P.INK_2, lw=0.8, ls="--")
        ax.axhline(r2, color=P.INK_2, lw=0.8, ls=":")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharey=True)
    ax = axes[0, 0]
    d = sub("B1").assign(x=lambda t: t.p.map(lambda q: q["p"])).sort_values("x")
    ax.plot(range(len(d)), d.composite_f1, "-o", ms=4, color=P.SERIES[0])
    ax.set_xticks(range(len(d)), [f"{v:g}" for v in d.x])
    c = d[d.params == chosen["B1"]].iloc[0]
    ring(ax, list(d.x).index(c.x), c.composite_f1)
    ax.set_title("B1 percentile", loc="left")
    ax.set_xlabel("p")

    ax = axes[0, 1]
    for m, key, col in (("B2", "k", P.SERIES[0]), ("S-robust", "lam", P.SERIES[1])):
        d = sub(m).assign(x=lambda t, k=key: t.p.map(lambda q: q[k])).sort_values("x")
        ax.plot(d.x, d.composite_f1, "-o", ms=4, color=col, label=f"{m} ({'k' if key == 'k' else 'lambda'})")
        c = d[d.params == chosen[m]].iloc[0]
        ring(ax, c.x, c.composite_f1)
    ax.set_title("Static: B2 vs S-robust", loc="left")
    ax.set_xlabel("multiplier")
    ax.legend(loc="lower right")

    ax = axes[0, 2]
    d = sub("B3").assign(x=lambda t: t.p.map(lambda q: q["w"])).sort_values("x")
    ax.plot(range(len(d)), d.composite_f1, "-o", ms=4, color=P.SERIES[0])
    ax.set_xticks(range(len(d)), [str(v) for v in d.x])
    c = d[d.params == chosen["B3"]].iloc[0]
    ring(ax, list(d.x).index(c.x), c.composite_f1)
    ax.set_title("B3 / RePAD2 (k = 3)", loc="left")
    ax.set_xlabel("w")

    panels = (("B3-k", "k", None, axes[1, 0], "B3-k rolling mean + k std"),
              ("P1", "lam", "plain", axes[1, 1], "P1 plain memory"),
              ("P1", "lam", "gated", axes[1, 2], "P1 gated memory"))
    for m, key, mem, ax, title in panels:
        d = sub(m)
        if mem:
            d = d[d.p.map(lambda q: q["memory"]) == mem]
        for j, w in enumerate(WS):
            dw = d[d.p.map(lambda q: q["w"]) == w].assign(x=lambda t, k=key: t.p.map(lambda q: q[k])).sort_values("x")
            ax.plot(dw.x, dw.composite_f1, "-o", ms=3, lw=1, color=P.SERIES[j], label=f"w = {w}")
            hit = dw[dw.params == chosen[m]]
            if len(hit):
                ring(ax, hit.x.iloc[0], hit.composite_f1.iloc[0])
        ax.set_title(title, loc="left")
        ax.set_xlabel("k" if key == "k" else "lambda")
    axes[1, 0].legend(loc="upper right", fontsize=6)
    for ax in axes.flat:
        refs(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation composite F1")
    fig.text(0.5, -0.01, "Mean over 24 tuning files x 3 seeds. Circle: selected. Dashed: Base (Ahmad 2026). "
             "Dotted: R2 EVT (Yu et al. 2024).", ha="center", fontsize=6.5, color=P.INK_2)
    fig.tight_layout()
    P.save(fig, PHASE, "fig01_tuning_objective")
    print("written", P.FIG_DIR / PHASE)


if __name__ == "__main__":
    main()
