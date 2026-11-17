"""Metrics, NAB scorer and data helpers on small hand-made cases."""
import numpy as np
import pandas as pd
import pytest

from pipeline import metrics as M
from pipeline import nab_scorer as N
from pipeline.data import clip_windows, split_bounds, window_indices, windows_to_labels


# ------------------------------------------------------------------ metrics
def test_point_metrics():
    a = [0, 1, 1, 0, 1, 0]
    y = [0, 1, 0, 0, 1, 1]
    m = M.point_metrics(a, y)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 3)


def test_no_alarms_gives_zero_precision():
    m = M.point_metrics([0, 0, 0], [0, 1, 0])
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_event_recall_and_composite():
    a = np.zeros(20, bool)
    a[3] = True                  # hits window (2, 5)
    a[15] = True                 # false alarm
    windows = [(2, 5), (9, 11)]
    assert M.event_recall(a, windows) == 0.5
    y = windows_to_labels(windows, 20)
    p = M.point_metrics(a, y)["precision"]       # 1 of 2 alarms inside
    assert M.composite_f1(p, 0.5) == pytest.approx(0.5)
    assert np.isnan(M.event_recall(a, []))


def test_first_steps():
    assert M.first_steps([0, 1, 1, 0, 1]).tolist() == [False, True, False, False, True]


def test_high_variance_mask():
    v = np.r_[np.zeros(100), np.tile([0, 5], 50)]   # calm train, noisy later
    m = M.high_variance_mask(v, slice(0, 100), roll=10, pct=75)
    assert not m[:100].any() and m[110:].all()


def test_auc_nan_without_positives():
    assert np.isnan(M.auc_metrics([0.1, 0.2], [0, 0])["roc_auc"])
    r = M.auc_metrics([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
    assert r["roc_auc"] == 1.0 and r["pr_auc"] == 1.0


# --------------------------------------------------------------- NAB scorer
def test_nab_no_alarms_is_null():
    r = N.score_file(np.zeros(100, bool), [(40, 49)], probation_percent=0)
    assert r["raw"] == pytest.approx(-1.0) and r["null"] == -1.0
    assert N.normalise(r["raw"], r["null"], r["perfect"]) == 0.0


def test_nab_alarm_at_window_start_is_perfect():
    a = np.zeros(100, bool)
    a[40] = True
    r = N.score_file(a, [(40, 49)], probation_percent=0)
    assert r["raw"] == pytest.approx(1.0)
    assert N.normalise(r["raw"], r["null"], r["perfect"]) == pytest.approx(100.0)


def test_nab_alarm_at_window_end_scores_near_zero():
    a = np.zeros(100, bool)
    a[49] = True
    r = N.score_file(a, [(40, 49)], probation_percent=0)
    # position -(49-49+1)/10 = -0.1
    assert r["raw"] == pytest.approx(N.scaled_sigmoid(-0.1) / N.MAX_TP)


def test_nab_false_positive_before_any_window():
    a = np.zeros(100, bool)
    a[10] = True
    r = N.score_file(a, [(40, 49)], profile="standard", probation_percent=0)
    assert r["raw"] == pytest.approx(-1.0 - 0.11)


def test_nab_false_positive_right_after_window_is_cheaper():
    a = np.zeros(100, bool)
    a[50] = True                 # 1 step after right edge 49, width 10
    r = N.score_file(a, [(40, 49)], probation_percent=0)
    assert r["raw"] == pytest.approx(-1.0 + 0.11 * N.scaled_sigmoid(1 / 9))


def test_nab_probation_drops_early_alarms_and_windows():
    a = np.zeros(100, bool)
    a[5] = True
    r = N.score_file(a, [(2, 8)], probation_percent=0.15)   # window inside first 15 rows
    assert r["n_windows"] == 0 and r["raw"] == 0.0


# ------------------------------------------------------------------- data
def test_split_bounds():
    tr, va, te = split_bounds(10)
    assert (tr, va, te) == (slice(0, 6), slice(6, 8), slice(8, 10))


def test_clip_windows():
    assert clip_windows([(2, 5), (7, 12), (15, 18)], slice(6, 14)) == [(1, 6)]


def test_window_indices_exact_match():
    ts = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=5, freq="h"))
    assert window_indices(ts, [("2020-01-01 01:00:00", "2020-01-01 03:00:00.000000")]) == [(1, 3)]
    with pytest.raises(ValueError):
        window_indices(ts, [("2020-01-01 01:30:00", "2020-01-01 03:00:00")])
