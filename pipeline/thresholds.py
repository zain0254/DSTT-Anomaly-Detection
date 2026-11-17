"""Threshold rules applied to LSTM-AE reconstruction errors e(t).

Static rules (fitted on RE_train, the errors of all training-split points):
  base_threshold       mean + 3 std                    "Base (Ahmad 2026)", untuned
  percentile_threshold p-th percentile                 B1
  mean_k_std_threshold mean + k std                    B2
  robust_threshold     median + lambda * scaled MAD    S-robust
  evt_pot_threshold    GPD tail quantile (POT)         R2 EVT-POT (Alcarria et al. 2026), untuned

Rolling rules (rolling_alarms), threshold at t from the w errors before t:
  kind="mean_std",   mult=3    B3 / RePAD2 rule (Lee & Lin 2023)
  kind="median_mad", mult=lam  P1, memory "plain" or "gated"

All rules alarm when e(t) > threshold(t). Standard deviations use ddof=0.
MAD is scaled by 1.4826. See docs/decisions.md D-08, D-09, D-14, D-15.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MAD_SCALE = 1.4826
FLOOR_FACTOR = 0.1
FLOOR_MIN = 1e-6


def scaled_mad(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x))))


def scale_floor(train_err) -> float:
    """Fixed floor for B3 and P1 (D-09 with its amendment):
    0.1 * scaled MAD of the file's RE_train; if that is 0, 0.1 * std(RE_train);
    if that is also 0, 1e-6."""
    mad = scaled_mad(train_err)
    if mad > 0:
        return FLOOR_FACTOR * mad
    std = float(np.asarray(train_err, dtype=float).std())
    if std > 0:
        return FLOOR_FACTOR * std
    return FLOOR_MIN


# ------------------------------------------------------------ static rules
def base_threshold(train_err) -> float:
    return mean_k_std_threshold(train_err, 3.0)


def percentile_threshold(train_err, p: float) -> float:
    return float(np.percentile(np.asarray(train_err, dtype=float), p))


def mean_k_std_threshold(train_err, k: float) -> float:
    x = np.asarray(train_err, dtype=float)
    return float(x.mean() + k * x.std())


def robust_threshold(train_err, lam: float) -> float:
    x = np.asarray(train_err, dtype=float)
    return float(np.median(x) + lam * scaled_mad(x))


EVT_Q = 0.98
EVT_ALPHA = 0.01
EVT_MIN_EXCESS = 30
EVT_FALLBACK_Q = 0.99
EVT_XI_ZERO = 1e-6


def gpd_tail_threshold(u: float, sigma: float, xi: float, p_u: float, alpha: float) -> float:
    """POT threshold (Alcarria et al. 2026, Eq. 2): the level exceeded with
    probability alpha, given a GPD(xi, sigma) for the excesses over u, which
    occur with probability p_u."""
    if abs(xi) > EVT_XI_ZERO:
        return float(u + (sigma / xi) * ((p_u / alpha) ** xi - 1.0))
    return float(u + sigma * np.log(p_u / alpha))


def evt_pot_threshold(train_err, q: float = EVT_Q, alpha: float = EVT_ALPHA,
                      min_excess: int = EVT_MIN_EXCESS) -> tuple[float, dict]:
    """R2 EVT-POT (D-24). u = q-quantile of RE_train; GPD fitted by maximum
    likelihood (loc fixed at 0) to the excesses over u. Falls back to the
    0.99 quantile of RE_train if there are fewer than `min_excess` excesses or
    the fit fails. Returns (threshold, info); info["fallback"] says which
    path was used."""
    from scipy.stats import genpareto

    x = np.asarray(train_err, dtype=float)
    x = x[np.isfinite(x)]
    u = float(np.quantile(x, q))
    exc = x[x > u] - u
    p_u = len(exc) / len(x)
    info = {"u": u, "n_excess": int(len(exc)), "p_u": p_u, "xi": float("nan"),
            "sigma": float("nan"), "fallback": False, "reason": ""}

    def fallback(reason: str):
        info.update(fallback=True, reason=reason)
        return float(np.quantile(x, EVT_FALLBACK_Q)), info

    if len(exc) < min_excess:
        return fallback(f"{len(exc)} excesses < {min_excess}")
    try:
        xi, _, sigma = genpareto.fit(exc, floc=0)
    except Exception as e:  # scipy optimiser failure
        return fallback(f"fit failed: {e!r}")
    info.update(xi=float(xi), sigma=float(sigma))
    if not (np.isfinite(xi) and np.isfinite(sigma) and sigma > 0):
        return fallback("fit returned invalid parameters")
    t = gpd_tail_threshold(u, sigma, xi, p_u, alpha)
    if not np.isfinite(t):
        return fallback("threshold not finite")
    return t, info


def static_alarms(err, threshold: float) -> np.ndarray:
    """e(t) > threshold; points without a score (NaN) never alarm."""
    e = np.asarray(err, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(e, nan=-np.inf) > threshold


# ----------------------------------------------------------- rolling rules
def _stats(win: np.ndarray, kind: str):
    """(location, scale) along the last axis."""
    if kind == "mean_std":
        return win.mean(axis=-1), win.std(axis=-1)
    if kind == "median_mad":
        loc = np.median(win, axis=-1)
        mad = np.median(np.abs(win - np.expand_dims(loc, -1)), axis=-1)
        return loc, MAD_SCALE * mad
    raise ValueError(f"unknown kind {kind!r}")


def rolling_alarms(err, w: int, kind: str, mult: float, memory: str = "plain",
                   floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling threshold over a whole error stream.

    err: errors of one file, NaN where no score exists (the first T-1 points).
    The stream starts at the first finite error f. Points f .. f+w-1 fill the
    first window and never alarm (threshold NaN). From t = f+w on:
      threshold(t) = loc + mult * max(scale, floor)
    plain: the window is the w errors before t.
    gated: errors that raised an alarm are kept out of the window; when an
           alarm run exceeds w steps, the window is replaced by the run's last
           w errors (D-08).
    Returns (alarms, thresholds), both aligned with err.
    """
    e = np.asarray(err, dtype=float)
    n = len(e)
    alarms = np.zeros(n, dtype=bool)
    thr = np.full(n, np.nan)
    finite = np.flatnonzero(np.isfinite(e))
    if len(finite) == 0:
        return alarms, thr
    f = int(finite[0])
    if not np.all(np.isfinite(e[f:])):
        raise ValueError("errors must be finite after the first score")
    if n - f <= w:
        return alarms, thr

    if memory == "plain":
        win = sliding_window_view(e[f:], w)[:-1]       # row i = w errors before f+w+i
        loc, scale = _stats(win, kind)
        th = loc + mult * np.maximum(scale, floor)
        thr[f + w:] = th
        alarms[f + w:] = e[f + w:] > th
        return alarms, thr

    if memory != "gated":
        raise ValueError(f"unknown memory {memory!r}")
    buf = deque(e[f:f + w], maxlen=w)
    run: list[float] = []
    for t in range(f + w, n):
        loc, scale = _stats(np.fromiter(buf, float, w), kind)
        th = loc + mult * max(scale, floor)
        thr[t] = th
        if e[t] > th:
            alarms[t] = True
            run.append(e[t])
            if len(run) > w:
                buf = deque(run[-w:], maxlen=w)
                run = []
        else:
            run = []
            buf.append(e[t])
    return alarms, thr
