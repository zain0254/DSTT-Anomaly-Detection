"""Time one LSTM-AE fit (one file, one seed) and project the full run over
configs/datasets.json model_files x 3 seeds.

Projection: seconds per (training window x epoch) from this run, times the
training windows of every model file, times an epoch count, times 3 seeds.
Two epoch counts are reported: the epochs this run needed (early stopping) and
the 50-epoch maximum (upper bound). Scoring time is added per file in
proportion to its number of windows.

Writes results/phase2_timing.json. Does not save errors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import model as M  # noqa: E402
from pipeline.data import load_series, split_bounds  # noqa: E402

SEEDS = [15, 42, 123]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="realKnownCause/nyc_taxi.csv")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    s = load_series(args.file)
    train, _, _ = split_bounds(len(s.values))
    t0 = time.perf_counter()
    _, meta = M.fit_and_score(s.values, train.stop, args.seed, args.threads,
                              log=lambda m: print(m, flush=True))
    wall = time.perf_counter() - t0

    per_we = meta["train_seconds"] / (meta["n_train_windows"] * meta["epochs_run"])
    score_per_w = meta["score_seconds"] / (len(s.values) - M.T + 1)
    files = json.loads((ROOT / "configs" / "datasets.json").read_text())["model_files"]
    tot_train_w, tot_w = 0, 0
    for f in files:
        n = len(load_series(f).values)
        tr_end = split_bounds(n)[0].stop
        nw = tr_end - M.T + 1
        tot_train_w += nw - max(1, -(-nw // 10))          # minus 10 % hold-out
        tot_w += n - M.T + 1
    proj = {}
    for label, ep in (("epochs_as_this_run", meta["epochs_run"]), ("epochs_max_50", M.MAX_EPOCHS)):
        sec = len(SEEDS) * (per_we * tot_train_w * ep + score_per_w * tot_w)
        proj[label] = {"epochs": ep, "hours_sequential": round(sec / 3600, 2)}

    out = {"file": args.file, "seed": args.seed, "threads": args.threads, "wall_seconds": round(wall, 1),
           **{k: v for k, v in meta.items() if k != "loss_history"},
           "seconds_per_window_epoch": per_we, "n_model_files": len(files),
           "total_train_windows_all_files": tot_train_w, "total_windows_all_files": tot_w,
           "seeds": SEEDS, "projection": proj, "loss_history": meta["loss_history"]}
    (ROOT / "results" / "phase2_timing.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "loss_history"}, indent=1))


if __name__ == "__main__":
    main()
