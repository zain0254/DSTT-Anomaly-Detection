"""Phase 2 figures from saved training outputs (results/phase2_errors/).

fig01_loss_curves: train and early-stopping MSE per epoch for a few files
  (seed 15), best epoch marked.
fig02_reconstruction_error: one file (seed 15): series and per-window MAE,
  NAB windows shaded, split boundaries dashed.
No threshold is computed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import plotting as P  # noqa: E402
from pipeline.data import load_series, split_bounds  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ERR = ROOT / "results" / "phase2_errors"
PHASE = "phase2_model"
LOSS_FILES = ["realKnownCause/nyc_taxi.csv", "realKnownCause/machine_temperature_system_failure.csv",
              "realTweets/Twitter_volume_KO.csv", "realTraffic/speed_7578.csv"]
ERR_FILE = "realKnownCause/nyc_taxi.csv"
SEED = 15


def stem(name: str, seed: int) -> Path:
    return ERR / f"{name.replace('.csv', '').replace('/', '__')}__s{seed}"


def fig_loss() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for ax, name in zip(axes.flat, LOSS_FILES):
        m = json.loads(stem(name, SEED).with_suffix(".json").read_text())
        h = np.array(m["loss_history"])
        ep = np.arange(1, len(h) + 1)
        ax.plot(ep, h[:, 0], color=P.SERIES[0], lw=1.2, label="training MSE")
        ax.plot(ep, h[:, 1], color=P.SERIES[1], lw=1.2, label="early-stopping MSE (last 10 % of train)")
        b = m["best_epoch"]
        ax.plot([b], [h[b - 1, 1]], "o", ms=5, color=P.SERIES[1], mec=P.SURFACE, mew=1.5)
        ax.annotate(f"best epoch {b}", (b, h[b - 1, 1]), xytext=(4, 8), textcoords="offset points",
                    fontsize=6.5, color=P.INK)
        ax.set_yscale("log")
        ax.set_title(name.split("/")[1].replace(".csv", ""), loc="left", fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (scaled units)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    P.save(fig, PHASE, "fig01_loss_curves")


def fig_error() -> None:
    s = load_series(ERR_FILE)
    e = np.load(stem(ERR_FILE, SEED).with_suffix(".npz"))["errors"]
    tr, va, te = split_bounds(len(s.values))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    a1.plot(s.timestamps, s.values, color=P.CAT_COLOR[s.category], lw=0.5)
    a1.set_ylabel("Value")
    a2.plot(s.timestamps, e, color=P.INK_2, lw=0.5)
    a2.set_ylabel("Window MAE (scaled)")
    for ax in (a1, a2):
        P.shade_windows(ax, s.timestamps, s.windows, label=None)
        for b in (va.start, te.start):
            ax.axvline(s.timestamps[b], color=P.INK_2, lw=0.7, ls="--")
    y = a1.get_ylim()[1]
    for x0, x1, txt in ((0, tr.stop - 1, "train"), (va.start, va.stop - 1, "val"), (te.start, len(s.values) - 1, "test")):
        a1.text(s.timestamps[(x0 + x1) // 2], y, txt, ha="center", va="bottom", fontsize=7, color=P.INK_2)
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=P.ANOMALY, alpha=0.18, label="NAB window")], loc="upper left")
    a1.set_title(f"{ERR_FILE.split('/')[1]}, seed {SEED}", loc="left", fontsize=8, pad=12)
    fig.tight_layout()
    P.save(fig, PHASE, "fig02_reconstruction_error")


if __name__ == "__main__":
    P.apply_style()
    fig_loss()
    fig_error()
    print("figures written to", P.FIG_DIR / PHASE)
