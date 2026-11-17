"""Threshold rules on small arrays with hand-computed answers."""
import numpy as np
import pytest

from pipeline import thresholds as T

S = 1.4826  # MAD scale


# ------------------------------------------------------------------ static
def test_base_is_mean_plus_3_std():
    x = [1, 2, 3, 4, 5]                      # mean 3, std (ddof 0) sqrt(2)
    assert T.base_threshold(x) == pytest.approx(3 + 3 * np.sqrt(2))


def test_b2_mean_k_std():
    assert T.mean_k_std_threshold([1, 2, 3, 4, 5], 1.0) == pytest.approx(3 + np.sqrt(2))


def test_b1_percentile():
    assert T.percentile_threshold([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)
    assert T.percentile_threshold([1, 2, 3, 4, 5], 100) == pytest.approx(5.0)


def test_s_robust_ignores_outlier():
    x = [1, 2, 3, 4, 100]                    # median 3, MAD 1
    assert T.scaled_mad(x) == pytest.approx(S)
    assert T.robust_threshold(x, 2.0) == pytest.approx(3 + 2 * S)


def test_scale_floor():
    assert T.scale_floor([1, 2, 3, 4, 100]) == pytest.approx(0.1 * S)


def test_scale_floor_falls_back_to_std_when_mad_is_zero():
    x = [1, 1, 1, 1, 1, 6]                   # MAD 0, std > 0
    assert T.scaled_mad(x) == 0.0
    assert T.scale_floor(x) == pytest.approx(0.1 * np.std(x))


def test_scale_floor_minimum_on_constant_errors():
    assert T.scale_floor([2.0, 2.0, 2.0]) == pytest.approx(1e-6)


def test_static_alarms_strict_and_nan():
    a = T.static_alarms([np.nan, 2.0, 3.0, 3.5], 3.0)
    assert a.tolist() == [False, False, False, True]


# ------------------------------------------------- R2 EVT (Yu et al. 2024)
def test_evt_quantile_both_branches():
    # gamma = 0: th - sigma * log(q n / n_th)
    assert T.evt_quantile(1.0, 2.0, 0.0, 1e-3, 1000, 20) == pytest.approx(1 - 2 * np.log(0.05))
    # gamma = 0.5: th + sigma/gamma * ((q n / n_th)^-gamma - 1)
    assert T.evt_quantile(1.0, 2.0, 0.5, 1e-3, 1000, 20) == pytest.approx(1 + 4 * (0.05 ** -0.5 - 1))


def test_eq8_score():
    x = np.array([1.0, 1, 1, 1, 6])
    rest = x[:4]
    expected = ((x.mean() - rest.mean()) / x.mean() + (x.var() - rest.var()) / x.var()) / 1
    assert T.eq8_score(x, 5.0) == pytest.approx(expected)
    assert T.eq8_score(x, 10.0) is None          # nothing above: skipped


def test_evt_on_exponential_errors():
    x = np.random.default_rng(0).exponential(1.0, 20_000)
    t, info = T.evt_threshold(x)
    assert info["path"] == "eq8"
    assert t > np.quantile(x, 0.98)                       # above th
    assert 1e-5 <= info["q"] <= 1e-3 and info["q"] in set(T.EVT_Q_GRID.tolist())
    # Exp(1) excesses: GPD with gamma near 0 and sigma near 1
    assert abs(info["gamma"]) < 0.15 and info["sigma"] == pytest.approx(1.0, abs=0.15)
    # E_TH(q) for the chosen q equals the reported threshold
    assert t == pytest.approx(T.evt_quantile(info["th"], info["sigma"], info["gamma"],
                                             info["q"], info["n"], info["n_th"]))


def test_evt_fallback_too_few_excesses():
    x = np.random.default_rng(1).exponential(1.0, 300)    # 6 excesses over the 0.98 quantile
    t, info = T.evt_threshold(x)
    assert info["path"].startswith("fallback") and info["n_th"] < 10
    assert t == pytest.approx(x.max())


def test_evt_fallback_constant_errors():
    t, info = T.evt_threshold(np.full(5000, 0.3))
    assert info["path"].startswith("fallback") and t == pytest.approx(0.3)


# ---------------------------------------------------- rolling, plain memory
def test_b3_plain_mean_std():
    e = [1, 1, 1, 1, 10, 1]
    a, th = T.rolling_alarms(e, w=4, kind="mean_std", mult=3)
    assert np.isnan(th[:4]).all() and not a[:4].any()
    assert th[4] == pytest.approx(1.0)       # window [1,1,1,1]: mean 1, std 0
    win = np.array([1, 1, 1, 10.0])
    assert th[5] == pytest.approx(win.mean() + 3 * win.std())
    assert a.tolist() == [False, False, False, False, True, False]


def test_floor_applies_when_scale_is_small():
    e = [1, 1, 1, 1, 10, 1]
    _, th = T.rolling_alarms(e, w=4, kind="mean_std", mult=3, floor=0.5)
    assert th[4] == pytest.approx(1 + 3 * 0.5)
    win = np.array([1, 1, 1, 10.0])          # std above the floor: floor unused
    assert th[5] == pytest.approx(win.mean() + 3 * win.std())


def test_p1_plain_median_mad():
    e = [1, 2, 3, 4, 5, 100, 3]
    a, th = T.rolling_alarms(e, w=5, kind="median_mad", mult=3, memory="plain")
    assert th[5] == pytest.approx(3 + 3 * S)          # [1..5]: median 3, MAD 1
    assert th[6] == pytest.approx(4 + 3 * S)          # [2,3,4,5,100]: median 4, MAD 1
    assert a.tolist() == [False] * 5 + [True, False]


def test_leading_nan_shifts_warmup():
    e = [np.nan, np.nan, 1, 1, 1, 1, 10]
    a, th = T.rolling_alarms(e, w=4, kind="mean_std", mult=3)
    assert np.isnan(th[:6]).all()
    assert th[6] == pytest.approx(1.0) and a[6]


def test_stream_shorter_than_window():
    a, th = T.rolling_alarms([1.0, 2.0, 3.0], w=3, kind="median_mad", mult=3)
    assert not a.any() and np.isnan(th).all()


# ---------------------------------------------------- rolling, gated memory
def test_gated_keeps_alarmed_error_out():
    e = [1, 2, 3, 4, 5, 100, 3]
    a, th = T.rolling_alarms(e, w=5, kind="median_mad", mult=3, memory="gated")
    assert a[5]
    assert th[6] == pytest.approx(3 + 3 * S)          # window still [1..5], 100 excluded
    assert not a[6]


def test_gated_short_run_never_enters_window():
    # run of 2 alarms (<= w=3) ends; its errors stay out of the window
    e = [1, 2, 1, 10, 10, 1, 2.4]
    a, th = T.rolling_alarms(e, w=3, kind="median_mad", mult=3, memory="gated", floor=0.5)
    assert a.tolist() == [False, False, False, True, True, False, False]
    # window at t=6 is [2, 1, 1]: median 1, MAD 0 -> floor 0.5
    assert th[6] == pytest.approx(1 + 3 * 0.5)


def test_gated_long_run_becomes_new_normal():
    # window [1,2,1]: median 1, MAD 0 -> scale = floor 0.5, threshold 2.5
    e = [1, 2, 1, 10, 10, 10, 10, 10, 10]
    a, th = T.rolling_alarms(e, w=3, kind="median_mad", mult=3, memory="gated", floor=0.5)
    # t=3..6 alarm; at t=6 the run length is 4 > w, so the window becomes
    # [10,10,10] after that decision; from t=7 the threshold is 10 + 1.5
    assert a.tolist() == [False, False, False, True, True, True, True, False, False]
    assert th[6] == pytest.approx(2.5)
    assert th[7] == pytest.approx(11.5)


def test_plain_differs_from_gated_on_same_input():
    e = [1, 2, 1, 10, 10, 10, 10, 10, 10]
    a, _ = T.rolling_alarms(e, w=3, kind="median_mad", mult=3, memory="plain", floor=0.5)
    # t=4: window [2,1,10] median 2, MAD 1 -> 2 + 3*S = 6.45 -> alarm
    # t=5: window [1,10,10] median 10, MAD 0 -> 11.5 -> no alarm
    assert a.tolist() == [False, False, False, True, True, False, False, False, False]


def test_gated_equals_plain_without_alarms():
    rng = np.random.default_rng(1)
    e = rng.normal(10, 1, 300)
    ap, tp = T.rolling_alarms(e, 50, "median_mad", 50, "plain")
    ag, tg = T.rolling_alarms(e, 50, "median_mad", 50, "gated")
    assert not ap.any() and not ag.any()
    np.testing.assert_allclose(tp, tg)


def test_unknown_memory_raises():
    with pytest.raises(ValueError):
        T.rolling_alarms([1.0] * 10, 3, "median_mad", 3, memory="other")
