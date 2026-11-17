"""Median test error / median training error for the 18 evaluation series
(paper, Discussion). Per file and seed: median of the errors in the test
segment divided by the median of the training-split errors (RE_train). The
per-file ratio is the mean over the 3 seeds.

Output: results/phase3/error_drift.csv with one row per file (ratio per seed
and mean) and a final row file = "ALL (median over 18 files)".
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data import load_series, split_bounds  # noqa: E402

SEEDS = [15, 42, 123]
ERR = ROOT / "results" / "phase2_errors"

files = json.loads((ROOT / "configs" / "datasets.json").read_text())["evaluation"]
rows = []
for f in files:
    tr, _, te = split_bounds(len(load_series(f).values))
    r = {"file": f}
    for s in SEEDS:
        e = np.load(ERR / f"{f.replace('.csv', '').replace('/', '__')}__s{s}.npz")["errors"]
        r[f"ratio_s{s}"] = float(np.nanmedian(e[te]) / np.nanmedian(e[tr]))
    r["ratio_mean_over_seeds"] = float(np.mean([r[f"ratio_s{s}"] for s in SEEDS]))
    rows.append(r)
df = pd.DataFrame(rows)
summary = {"file": "ALL (median over 18 files)",
           **{c: float(df[c].median()) for c in df.columns if c != "file"},
           "n_files_ratio_above_1": int((df.ratio_mean_over_seeds > 1).sum()),
           "min_ratio": float(df.ratio_mean_over_seeds.min()), "max_ratio": float(df.ratio_mean_over_seeds.max())}
df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
df.to_csv(ROOT / "results" / "phase3" / "error_drift.csv", index=False)
print(df.round(3).to_string(index=False))
