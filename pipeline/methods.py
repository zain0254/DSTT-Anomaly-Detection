"""Method registry for Phase 3 (D-10, D-24, D-26): parameter grids and how
each method turns one error series into alarms over the whole stream."""
from __future__ import annotations

import itertools
import json

import numpy as np

from . import thresholds as T

LAMBDAS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
WS = [50, 100, 200, 400]


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


METHODS = {
    "Base": {"label": "Base (Ahmad 2026)", "grid": [{}]},
    "B1": {"label": "B1 percentile", "grid": _grid(p=[90, 95, 97, 98, 99, 99.5, 99.9])},
    "B2": {"label": "B2 mean + k std", "grid": _grid(k=[1, 1.5, 2, 2.5, 3, 4, 5, 6, 8])},
    "S-robust": {"label": "S-robust median + lambda MAD", "grid": _grid(lam=LAMBDAS)},
    "R2": {"label": "R2 EVT (Yu et al. 2024)", "grid": [{}]},
    "B3": {"label": "B3 / RePAD2 rule (Lee & Lin 2023)", "grid": _grid(w=WS)},
    "B3-k": {"label": "B3-k rolling mean + k std", "grid": _grid(k=LAMBDAS, w=WS)},
    "P1": {"label": "P1 rolling median + lambda MAD (proposed)",
           "grid": _grid(lam=LAMBDAS, w=WS, memory=["plain", "gated"])},
}
STATIC = {"Base", "B1", "B2", "S-robust", "R2"}


def pkey(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


class SeriesContext:
    """One error series (file, seed) with RE_train, floor and caches."""

    def __init__(self, errors: np.ndarray, train_end: int):
        self.e = np.asarray(errors, dtype=float)
        tr = self.e[:train_end]
        self.re_train = tr[np.isfinite(tr)]
        self.floor = T.scale_floor(self.re_train)
        self.f = int(np.flatnonzero(np.isfinite(self.e))[0])
        self._plain = {}
        self._r2 = None

    def r2(self):
        if self._r2 is None:
            self._r2 = T.evt_threshold(self.re_train)
        return self._r2

    def plain_threshold(self, kind: str, w: int, mult: float) -> np.ndarray:
        if (kind, w) not in self._plain:
            self._plain[(kind, w)] = T.rolling_plain_stats(self.e[self.f:], w, kind)
        loc, scale = self._plain[(kind, w)]
        thr = np.full(len(self.e), np.nan)
        thr[self.f + w:] = loc + mult * np.maximum(scale, self.floor)
        return thr

    def apply(self, method: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """(alarms, threshold per row) over the whole stream."""
        n = len(self.e)
        if method in STATIC:
            if method == "Base":
                t = T.base_threshold(self.re_train)
            elif method == "B1":
                t = T.percentile_threshold(self.re_train, params["p"])
            elif method == "B2":
                t = T.mean_k_std_threshold(self.re_train, params["k"])
            elif method == "S-robust":
                t = T.robust_threshold(self.re_train, params["lam"])
            else:
                t = self.r2()[0]
            return T.static_alarms(self.e, t), np.full(n, t)
        if method == "P1" and params["memory"] == "gated":
            return T.rolling_alarms(self.e, params["w"], "median_mad", params["lam"], "gated", self.floor)
        if method == "P1":
            thr = self.plain_threshold("median_mad", params["w"], params["lam"])
        else:
            k = 3.0 if method == "B3" else params["k"]
            thr = self.plain_threshold("mean_std", params["w"], k)
        with np.errstate(invalid="ignore"):
            alarms = np.nan_to_num(self.e, nan=-np.inf) > np.nan_to_num(thr, nan=np.inf)
        return alarms, thr


# D-27 (post-hoc, exploratory): only the edge grids are extended.
WS_P1_EXPL = [10, 20, 30] + WS
WS_B3_EXPL = WS + [600, 800]
EXPLORATORY = dict(METHODS)
EXPLORATORY["B1"] = {**METHODS["B1"], "grid": _grid(p=[80, 85, 90, 95, 97, 98, 99, 99.5, 99.9])}
EXPLORATORY["B3"] = {**METHODS["B3"], "grid": _grid(w=WS_B3_EXPL)}
EXPLORATORY["B3-k"] = {**METHODS["B3-k"], "grid": _grid(k=LAMBDAS, w=WS_B3_EXPL)}
EXPLORATORY["P1"] = {**METHODS["P1"], "grid": _grid(lam=LAMBDAS, w=WS_P1_EXPL, memory=["plain", "gated"])}
GRID_SETS = {"primary": METHODS, "exploratory": EXPLORATORY}


def w_allowed(params: dict, train_end: int, first_score: int = 29) -> bool:
    """A rolling window needs w errors before the validation segment (D-27)."""
    return "w" not in params or train_end - first_score >= params["w"]
