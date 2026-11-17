"""NAB loading, label windows and the chronological 60/20/20 split."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAB_DIR = ROOT / "data" / "nab"
SPLIT = (0.6, 0.2, 0.2)


@dataclass
class Series:
    name: str                      # e.g. "realKnownCause/nyc_taxi.csv"
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    windows: list[tuple[int, int]]  # inclusive row-index pairs, sorted

    @property
    def category(self) -> str:
        return self.name.split("/")[0]

    @property
    def labels(self) -> np.ndarray:
        return windows_to_labels(self.windows, len(self.values))


def list_files(nab_dir: Path = NAB_DIR) -> list[str]:
    with open(nab_dir / "labels" / "combined_windows.json") as f:
        return sorted(json.load(f))


def load_windows(nab_dir: Path = NAB_DIR) -> dict[str, list[tuple[str, str]]]:
    with open(nab_dir / "labels" / "combined_windows.json") as f:
        return {k: [tuple(w) for w in v] for k, v in json.load(f).items()}


def window_indices(timestamps: pd.DatetimeIndex, windows) -> list[tuple[int, int]]:
    """Map NAB window timestamps to row indices. NAB requires both edges to be
    existing timestamps (sweeper.py uses timestamps.index), so a missing edge
    is an error here too."""
    pos = {t: i for i, t in enumerate(timestamps)}
    out = []
    for start, end in windows:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s not in pos or e not in pos:
            raise ValueError(f"window edge not in timestamps: {start} .. {end}")
        out.append((pos[s], pos[e]))
    return sorted(out)


def load_series(name: str, nab_dir: Path = NAB_DIR, windows=None) -> Series:
    df = pd.read_csv(nab_dir / "data" / name)
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
    if windows is None:
        windows = load_windows(nab_dir)[name]
    return Series(name, ts, df["value"].to_numpy(dtype=float), window_indices(ts, windows))


def windows_to_labels(windows, n: int) -> np.ndarray:
    y = np.zeros(n, dtype=bool)
    for s, e in windows:
        y[s:e + 1] = True
    return y


def split_bounds(n: int, fractions=SPLIT) -> tuple[slice, slice, slice]:
    """Chronological train/val/test slices. Boundaries use floor, so test gets
    the remainder."""
    a = int(np.floor(fractions[0] * n))
    b = int(np.floor((fractions[0] + fractions[1]) * n))
    return slice(0, a), slice(a, b), slice(b, n)


def clip_windows(windows, seg: slice) -> list[tuple[int, int]]:
    """Windows intersected with a segment, re-indexed to the segment start."""
    out = []
    for s, e in windows:
        lo, hi = max(s, seg.start), min(e, seg.stop - 1)
        if lo <= hi:
            out.append((lo - seg.start, hi - seg.start))
    return out
