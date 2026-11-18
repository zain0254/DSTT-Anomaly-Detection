"""Pre-registered hypothesis tests (REPORT.md section 18: D-26).

Unit = file, metric averaged over the 3 seeds. Paired two-sided Wilcoxon
signed-rank test (scipy, zero differences dropped), mean difference P1 - other
with a percentile bootstrap 95 % CI (10,000 resamples of files, seed 2026 for every comparison),
wins/ties/losses for P1 in the metric's better direction.
Primary comparison P1 vs Base, tested alone. Secondary: P1 vs each other
method, Holm-corrected within each hypothesis measure.
alpha = 0.05 (not stated in D-26; the conventional level is used).

Measures
  H1  composite_f1                     evaluation files, higher is better (n = 18)
  H2  hv_fpr                           evaluation files, lower is better (n = 17; one file undefined)
  H2s clean_fpr, clean_false_alarms_per_1000   clean files, lower is better (n = 18)
  H3  nab_standard (every step, per file)      evaluation files, higher is better (n = 18)

Output: results/phase3/hypothesis_tests.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.methods import METHODS  # noqa: E402

RES = ROOT / "results" / "phase3"
ALPHA = 0.05
MEASURES = [  # hypothesis, role, metric, table, higher_is_better
    ("H1", "primary measure", "composite_f1", "test", True),
    ("H2", "primary measure", "hv_fpr", "test", False),
    ("H2", "secondary measure", "clean_fpr", "clean", False),
    ("H2", "secondary measure", "clean_false_alarms_per_1000", "clean", False),
    ("H3", "primary measure", "nab_standard", "test", True),
]


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def compare(a: pd.Series, b: pd.Series, higher: bool) -> dict:
    rng = np.random.default_rng(2026)          # same resamples for every comparison
    d = (a - b).dropna()
    n = len(d)
    nz = d[d != 0]
    if len(nz) == 0:
        stat, p = float("nan"), 1.0
    else:
        stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    boots = rng.choice(d.to_numpy(), size=(10_000, n), replace=True).mean(axis=1)
    better = d > 0 if higher else d < 0
    worse = d < 0 if higher else d > 0
    return {"n_files": n, "n_nonzero": int(len(nz)), "mean_P1": float(a[d.index].mean()),
            "mean_other": float(b[d.index].mean()), "mean_diff": float(d.mean()),
            "ci95_low": float(np.percentile(boots, 2.5)), "ci95_high": float(np.percentile(boots, 97.5)),
            "wins": int(better.sum()), "ties": int((d == 0).sum()), "losses": int(worse.sum()),
            "W": float(stat), "p": float(p)}


def main() -> None:
    tables = {"test": pd.read_csv(RES / "test_metrics.csv"), "clean": pd.read_csv(RES / "clean_metrics.csv")}
    rows = []
    for hyp, role, metric, tab, higher in MEASURES:
        per_file = tables[tab].groupby(["method", "file"])[metric].mean().unstack("method")
        p1 = per_file["P1"]
        prim = compare(p1, per_file["Base"], higher)
        rows.append({"hypothesis": hyp, "measure_role": role, "metric": metric,
                     "higher_is_better": higher, "comparison": "primary", "other": "Base",
                     **prim, "p_holm": prim["p"]})
        sec = []
        for m in METHODS:
            if m in ("P1", "Base"):
                continue
            sec.append({"hypothesis": hyp, "measure_role": role, "metric": metric,
                        "higher_is_better": higher, "comparison": "secondary", "other": m,
                        **compare(p1, per_file[m], higher)})
        for r, ph in zip(sec, holm([r["p"] for r in sec])):
            r["p_holm"] = ph
        rows += sec
    df = pd.DataFrame(rows)
    sig = df.p_holm < ALPHA
    p1_better = np.where(df.higher_is_better, df.mean_diff > 0, df.mean_diff < 0)
    df["result"] = np.select([sig & p1_better, sig & ~p1_better], ["P1 better", "P1 worse"], "no significant difference")
    df.to_csv(RES / "hypothesis_tests.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[["hypothesis", "metric", "comparison", "other", "n_files", "mean_P1", "mean_other", "mean_diff",
                  "ci95_low", "ci95_high", "wins", "ties", "losses", "p", "p_holm", "result"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
