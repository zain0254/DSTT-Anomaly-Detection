"""Train the shared LSTM-AE for every model file x seed and save its errors.

Resumable: a (file, seed) pair whose .npz and .json both exist is skipped, so
the command can be re-run after a crash or sleep. Each result is written to a
temporary file and renamed, so a half-written result never counts as done.
A failing pair is logged and recorded in failures.json; the run continues.

Outputs in results/phase2_errors/:
  <file id>__s<seed>.npz   errors (float64, one per row, NaN for the first T-1 rows)
  <file id>__s<seed>.json  meta: loss history (train, early-stopping MSE per
                           epoch), best epoch, epochs run, run time, settings
  run_summary.csv          one row per finished pair (rebuilt at the end)
  failures.json            pairs that raised an error in the last run
Log: logs/phase2_train.log

Usage: envs/pipeline/Scripts/python.exe scripts/train.py [--threads 4] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import model as M  # noqa: E402
from pipeline.data import load_series, split_bounds  # noqa: E402

SEEDS = [15, 42, 123]
OUT = ROOT / "results" / "phase2_errors"
LOG = ROOT / "logs" / "phase2_train.log"


def file_id(name: str) -> str:
    return name.replace(".csv", "").replace("/", "__")


def paths(name: str, seed: int) -> tuple[Path, Path]:
    stem = OUT / f"{file_id(name)}__s{seed}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_pair(name: str, seed: int, threads: int) -> dict:
    s = load_series(name)
    train, _, _ = split_bounds(len(s.values))
    t0 = time.perf_counter()
    errors, meta = M.fit_and_score(s.values, train.stop, seed, threads)
    meta.update({"file": name, "wall_seconds": round(time.perf_counter() - t0, 2),
                 "finished": datetime.now().isoformat(timespec="seconds"),
                 "settings": {"T": M.T, "hidden": M.HIDDEN, "lr": M.LR, "batch": M.BATCH,
                              "max_epochs": M.MAX_EPOCHS, "patience": M.PATIENCE,
                              "holdout": M.HOLDOUT}})
    npz, js = paths(name, seed)
    tmp_npz, tmp_js = npz.with_suffix(".tmp.npz"), js.with_suffix(".tmp.json")
    np.savez_compressed(tmp_npz, errors=errors)
    tmp_js.write_text(json.dumps(meta, indent=1))
    os.replace(tmp_npz, npz)
    os.replace(tmp_js, js)
    return meta


def summarise() -> pd.DataFrame:
    rows = []
    for js in sorted(OUT.glob("*__s*.json")):
        if js.name.endswith(".tmp.json"):
            continue
        m = json.loads(js.read_text())
        e = np.load(js.with_suffix(".npz"))["errors"]
        rows.append({"file": m["file"], "seed": m["seed"], "n_rows": m["n_rows"],
                     "n_train_windows": m["n_train_windows"], "epochs_run": m["epochs_run"],
                     "best_epoch": m["best_epoch"], "best_holdout_mse": m["best_holdout_mse"],
                     "train_seconds": m["train_seconds"], "wall_seconds": m["wall_seconds"],
                     "n_nan_errors": int(np.isnan(e).sum()),
                     "all_finite_after_T": bool(np.isfinite(e[M.T - 1:]).all()),
                     "err_min": float(np.nanmin(e)), "err_median": float(np.nanmedian(e)),
                     "err_max": float(np.nanmax(e))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "run_summary.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new fits (smoke test)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    files = json.loads((ROOT / "configs" / "datasets.json").read_text())["model_files"]
    pairs = [(f, s) for f in files for s in SEEDS]
    todo = [(f, s) for f, s in pairs if not all(p.exists() for p in paths(f, s))]
    log(f"start: {len(pairs)} pairs, {len(pairs) - len(todo)} already done, {len(todo)} to run, "
        f"threads={args.threads}")
    failures, done, t_run = [], 0, time.perf_counter()
    for i, (f, s) in enumerate(todo, 1):
        try:
            m = run_pair(f, s, args.threads)
            done += 1
            el = time.perf_counter() - t_run
            log(f"[{i}/{len(todo)}] {f} seed={s} epochs={m['epochs_run']} best={m['best_epoch']} "
                f"{m['wall_seconds']:.1f}s (elapsed {el / 60:.1f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left)")
        except Exception as exc:  # keep going; record the failure
            failures.append({"file": f, "seed": s, "error": repr(exc), "traceback": traceback.format_exc()})
            log(f"[{i}/{len(todo)}] FAILED {f} seed={s}: {exc!r}")
        if args.limit and done >= args.limit:
            break
    (OUT / "failures.json").write_text(json.dumps(failures, indent=1))
    df = summarise()
    log(f"end: {done} new fits, {len(failures)} failures, {len(df)} of {len(pairs)} pairs complete, "
        f"run time {(time.perf_counter() - t_run) / 60:.1f} min")


if __name__ == "__main__":
    main()
