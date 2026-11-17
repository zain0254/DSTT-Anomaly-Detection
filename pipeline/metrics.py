"""Evaluation metrics for one alarm sequence on one segment (D-07, D-17).

Point metrics use no point adjustment. NAB uses pipeline.nab_scorer: primary
= every alarm step, secondary = first step of each alarm run; both with
windows clipped to the segment and no probationary period. Neither is
comparable to the NAB leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import nab_scorer


def _div(a: float, b: float) -> float:
    return float(a) / b if b else float("nan")


def point_metrics(alarms, labels) -> dict:
    a = np.asarray(alarms, dtype=bool)
    y = np.asarray(labels, dtype=bool)
    tp = int((a & y).sum())
    fp = int((a & ~y).sum())
    fn = int((~a & y).sum())
    tn = int((~a & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0     # no alarms -> 0 (D-17)
    recall = _div(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "fpr": _div(fp, fp + tn)}


def event_recall(alarms, windows) -> float:
    a = np.asarray(alarms, dtype=bool)
    if not windows:
        return float("nan")
    return sum(bool(a[s:e + 1].any()) for s, e in windows) / len(windows)


def composite_f1(precision: float, ev_recall: float) -> float:
    if np.isnan(ev_recall):
        return float("nan")
    s = precision + ev_recall
    return 2 * precision * ev_recall / s if s > 0 else 0.0


def first_steps(alarms) -> np.ndarray:
    """True only at the first step of each run of alarms."""
    a = np.asarray(alarms, dtype=bool)
    prev = np.concatenate([[False], a[:-1]])
    return a & ~prev


def high_variance_mask(values, train: slice, roll: int = 50, pct: float = 75) -> np.ndarray:
    """Points whose causal `roll`-step rolling std of the raw series exceeds the
    `pct`-th percentile of that rolling std over the training split."""
    rs = pd.Series(np.asarray(values, dtype=float)).rolling(roll, min_periods=roll).std(ddof=0).to_numpy()
    ref = rs[train]
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return np.zeros(len(rs), dtype=bool)
    cut = np.percentile(ref, pct)
    return np.nan_to_num(rs, nan=-np.inf) > cut


def auc_metrics(scores, labels) -> dict:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def nab_metrics(alarms, windows) -> dict:
    """Raw, null, perfect and normalised NAB for both alarm conventions and all
    three profiles. Corpus scores sum raw/null/perfect over files first."""
    out = {}
    variants = {"": np.asarray(alarms, dtype=bool), "_first": first_steps(alarms)}
    for profile in nab_scorer.PROFILES:
        for suffix, a in variants.items():
            r = nab_scorer.score_file(a, windows, profile, probation_percent=0.0)
            key = f"nab_{profile}{suffix}"
            out[f"{key}_raw"] = r["raw"]
            out[f"{key}_null"] = r["null"]
            out[f"{key}_perfect"] = r["perfect"]
            out[key] = nab_scorer.normalise(r["raw"], r["null"], r["perfect"])
    return out


def evaluate_segment(alarms, scores, labels, windows, hv_mask=None) -> dict:
    """All metrics for one segment. alarms, scores, labels and hv_mask are
    already cut to the segment; windows are clipped and re-indexed to it
    (pipeline.data.clip_windows)."""
    a = np.asarray(alarms, dtype=bool)
    y = np.asarray(labels, dtype=bool)
    out = point_metrics(a, y)
    out["n_windows"] = len(windows)
    out["event_recall"] = event_recall(a, windows)
    out["composite_f1"] = composite_f1(out["precision"], out["event_recall"])
    out["alarm_rate"] = float(a.mean()) if len(a) else float("nan")
    if hv_mask is not None:
        m = np.asarray(hv_mask, dtype=bool) & ~y
        out["hv_normal_points"] = int(m.sum())
        out["hv_fpr"] = _div(int(a[m].sum()), int(m.sum()))
    out.update(auc_metrics(scores, y))
    out.update(nab_metrics(a, windows))
    return out


def clean_stream_metrics(alarms) -> dict:
    """Secondary metric (D-22) on an anomaly-free test segment: every alarm is
    false, so FPR = alarms / points. Not used for tuning."""
    a = np.asarray(alarms, dtype=bool)
    n = len(a)
    k = int(a.sum())
    return {"clean_points": n, "clean_false_alarms": k,
            "clean_fpr": _div(k, n), "clean_false_alarms_per_1000": _div(1000 * k, n)}
