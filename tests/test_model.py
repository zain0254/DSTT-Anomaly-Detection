"""LSTM-AE shape, score alignment and seed determinism (tiny runs)."""
import numpy as np

from pipeline import model as M


def _series(n=300):
    t = np.arange(n)
    return np.sin(t / 10) + 0.05 * np.random.default_rng(0).normal(size=n)


def test_architecture_and_params():
    net = M.build()
    import torch
    out = net(torch.zeros(4, M.T, 1))
    assert tuple(out.shape) == (4, M.T, 1)
    # encoder 4*64*(1+64+2) + decoder 4*64*(64+64+2) + linear 65
    assert sum(p.numel() for p in net.parameters()) == 4 * 64 * 67 + 4 * 64 * 130 + 65


def test_errors_aligned_to_window_end():
    x = _series()
    e, meta = M.fit_and_score(x, train_end=180, seed=1, threads=1, max_epochs=2)
    assert len(e) == len(x)
    assert np.isnan(e[: M.T - 1]).all() and np.isfinite(e[M.T - 1:]).all()
    assert meta["n_train_windows"] + meta["n_holdout_windows"] == 180 - M.T + 1


def test_same_seed_same_errors():
    x = _series()
    a, _ = M.fit_and_score(x, train_end=180, seed=7, threads=1, max_epochs=2)
    b, _ = M.fit_and_score(x, train_end=180, seed=7, threads=1, max_epochs=2)
    np.testing.assert_array_equal(a, b)
