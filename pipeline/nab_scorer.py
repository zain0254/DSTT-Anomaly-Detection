"""Official NAB scoring, ported from numenta/NAB nab/sweeper.py and
nab/runner.py at commit ea702d75 (MIT license, Copyright Numenta Inc.).

The port works on binary alarms and row-index windows instead of anomaly
scores and timestamps; for a detector with scores s(t) and threshold th, NAB's
alarms are s(t) >= th (Sweeper.scoreDataSet picks the sweep row at th).

Per-file raw score:
  sum over windows of max(sweep score of alarms inside the window), or
  -fnWeight if the window has no alarm; plus sum of fp-weighted sweep scores of
  alarms outside windows. Rows in the probationary period are ignored, and a
  window whose rows are all probationary is not counted at all.
Corpus score is the sum of per-file raw scores; normalised score is
  100 * (raw - null) / (perfect - null), perfect = tpWeight * number of windows,
  null = raw score of the detector that never alarms.
Validated against results/final_results.json by scripts/validate_scorer.py.
"""
from __future__ import annotations

import math

import numpy as np

PROFILES = {
    "standard": {"tpWeight": 1.0, "fnWeight": 1.0, "fpWeight": 0.11},
    "reward_low_FP_rate": {"tpWeight": 1.0, "fnWeight": 1.0, "fpWeight": 0.22},
    "reward_low_FN_rate": {"tpWeight": 1.0, "fnWeight": 2.0, "fpWeight": 0.11},
}
PROBATION_PERCENT = 0.15


def scaled_sigmoid(rel_pos: float) -> float:
    if rel_pos > 3.0:
        return -1.0
    return 2.0 / (1.0 + math.exp(5.0 * rel_pos)) - 1.0


MAX_TP = scaled_sigmoid(-1.0)


def probationary_length(n: int, percent: float = PROBATION_PERCENT) -> float:
    return min(math.floor(percent * n), percent * 5000)


def sweep(n: int, windows, cost: dict, probation: float):
    """Per-row weighted score and window id (-1 outside windows, -2 for
    probationary rows), as in Sweeper.calcSweepScore."""
    score = np.empty(n)
    win_id = np.full(n, -1, dtype=int)
    starts = {s: (k, e) for k, (s, e) in enumerate(windows)}
    cur = None            # (k, right index, width)
    prev_right = prev_width = None
    for i in range(n):
        if i in starts:
            k, e = starts[i]
            cur = (k, e, float(e - i + 1))
        if cur is not None:
            k, right, width = cur
            pos = -(right - i + 1) / width
            score[i] = scaled_sigmoid(pos) * cost["tpWeight"] / MAX_TP
            win_id[i] = k
        else:
            if prev_right is None:
                u = -1.0
            else:
                u = scaled_sigmoid(abs(prev_right - i) / float(prev_width - 1))
            score[i] = u * cost["fpWeight"]
        if i < probation:
            win_id[i] = -2
        if cur is not None and i == cur[1]:
            prev_right, prev_width = i, cur[2]
            cur = None
    return score, win_id


def score_file(alarms, windows, profile: str = "standard",
               probation_percent: float = PROBATION_PERCENT) -> dict:
    """NAB raw score of one file. alarms: bool array; windows: sorted,
    non-overlapping inclusive index pairs. probation_percent=0 disables the
    probationary period (used when scoring a test segment of a series whose
    earlier part the model has already seen)."""
    alarms = np.asarray(alarms, dtype=bool)
    n = len(alarms)
    cost = PROFILES[profile]
    probation = probationary_length(n, probation_percent)
    score, win_id = sweep(n, windows, cost, probation)

    active = sorted(set(win_id[win_id >= 0].tolist()))
    best = {k: -cost["fnWeight"] for k in active}
    detected = set()
    fp_sum, n_fp = 0.0, 0
    for i in np.flatnonzero(alarms):
        k = win_id[i]
        if k == -2:
            continue
        if k == -1:
            fp_sum += score[i]
            n_fp += 1
        else:
            best[k] = max(best[k], score[i])
            detected.add(k)
    raw = sum(best.values()) + fp_sum
    return {"raw": raw, "n_windows": len(active), "n_detected": len(detected),
            "n_fp_alarms": n_fp, "null": -cost["fnWeight"] * len(active),
            "perfect": cost["tpWeight"] * len(active)}


def normalise(raw: float, null: float, perfect: float) -> float:
    if perfect == null:
        return float("nan")
    return 100.0 * (raw - null) / (perfect - null)
