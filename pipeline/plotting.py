"""Shared figure style. Every figure is saved as PNG (300 dpi) and PDF under
figures/<phase>/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
ANOMALY = "#e34948"          # shaded NAB windows (with alpha)

CATEGORIES = ["artificialNoAnomaly", "artificialWithAnomaly", "realAWSCloudwatch",
              "realAdExchange", "realKnownCause", "realTraffic", "realTweets"]
CAT_COLOR = dict(zip(CATEGORIES, SERIES))
PART_COLOR = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.0, "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def save(fig, phase: str, name: str) -> list[Path]:
    out = FIG_DIR / phase
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def shade_windows(ax, timestamps, windows, label: str | None = "NAB window") -> None:
    for k, (s, e) in enumerate(windows):
        ax.axvspan(timestamps[s], timestamps[e], color=ANOMALY, alpha=0.18, lw=0,
                   label=label if k == 0 else None)
