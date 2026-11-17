"""Shared LSTM autoencoder (base paper architecture, PyTorch CPU) and error
extraction. See docs/decisions.md D-04, D-05, D-11 to D-13, D-23.

  encoder LSTM(64) -> last hidden state repeated T=30 times -> decoder LSTM(64)
  -> per-step Linear(64, 1)

Training: windows of T rows that lie entirely in the training split (all of
them, no label filtering), MinMax scaling fitted on the training split, Adam
(lr 1e-3), MSE loss, batch 32, up to 50 epochs, early stopping on the
reconstruction MSE of the chronologically last 10 % of training windows with
patience 5, best epoch restored.
Score: mean absolute error of each window in scaled units, assigned to the
window's last row; the first T-1 rows have no score (NaN).
"""
from __future__ import annotations

import math
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

T = 30
HIDDEN = 64
LR = 1e-3
BATCH = 32
MAX_EPOCHS = 50
PATIENCE = 5
HOLDOUT = 0.1


class MinMax:
    """Min-max scaler fitted on the training split only."""

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.lo = float(x.min())
        span = float(x.max()) - self.lo
        self.span = span if span > 0 else 1.0
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.lo) / self.span


def windows(x: np.ndarray, t: int = T) -> np.ndarray:
    """Row i is x[i : i+t], the window ending at row i+t-1."""
    return sliding_window_view(np.asarray(x, dtype=np.float32), t)


def build(hidden: int = HIDDEN, t: int = T):
    import torch
    from torch import nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.t = t
            self.encoder = nn.LSTM(1, hidden, batch_first=True)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.output = nn.Linear(hidden, 1)

        def forward(self, x):                                   # (batch, T, 1)
            _, (h, _) = self.encoder(x)
            z = h[-1].unsqueeze(1).repeat(1, self.t, 1)          # RepeatVector(T)
            out, _ = self.decoder(z)
            return self.output(out)                              # TimeDistributed(Dense(1))

    return LSTMAutoencoder()


def set_determinism(seed: int, threads: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fit_and_score(values: np.ndarray, train_end: int, seed: int, threads: int = 4,
                  max_epochs: int = MAX_EPOCHS, log=None) -> tuple[np.ndarray, dict]:
    """Train on rows [0, train_end) and return (errors, meta). errors has one
    entry per row: per-window MAE at the window's last row, NaN for the first
    T-1 rows."""
    import torch

    set_determinism(seed, threads)
    x = MinMax().fit(values[:train_end]).transform(values)
    allw = windows(x)
    n_train_w = train_end - T + 1
    trw = allw[:n_train_w]
    n_hold = max(1, math.ceil(HOLDOUT * len(trw)))
    fit_x = torch.from_numpy(np.ascontiguousarray(trw[:-n_hold])).unsqueeze(-1)
    hold_x = torch.from_numpy(np.ascontiguousarray(trw[-n_hold:])).unsqueeze(-1)

    model = build()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)

    best, best_state, bad, hist = math.inf, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(fit_x), generator=gen)
        tot = 0.0
        for i in range(0, len(fit_x), BATCH):
            xb = fit_x[perm[i:i + BATCH]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            hl = loss_fn(model(hold_x), hold_x).item()
        hist.append([tot / len(fit_x), hl])
        if log:
            log(f"epoch {epoch} train {hist[-1][0]:.6f} holdout {hl:.6f} ({time.perf_counter() - t0:.1f}s)")
        if hl < best:
            best, bad = hl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()

    t1 = time.perf_counter()
    errs = []
    with torch.no_grad():
        for i in range(0, len(allw), 4096):
            xb = torch.from_numpy(np.ascontiguousarray(allw[i:i + 4096])).unsqueeze(-1)
            errs.append((model(xb) - xb).abs().mean(dim=(1, 2)).numpy())
    errors = np.full(len(values), np.nan)
    errors[T - 1:] = np.concatenate(errs)
    meta = {"seed": seed, "threads": threads, "n_rows": len(values), "train_end": train_end,
            "n_train_windows": len(trw) - n_hold, "n_holdout_windows": n_hold,
            "epochs_run": len(hist), "best_epoch": int(np.argmin([h[1] for h in hist])) + 1,
            "best_holdout_mse": best, "train_seconds": round(train_s, 2),
            "score_seconds": round(time.perf_counter() - t1, 2),
            "seconds_per_epoch": round(train_s / len(hist), 3),
            "n_params": sum(p.numel() for p in model.parameters()),
            "torch": torch.__version__, "loss_history": hist}
    return errors, meta
