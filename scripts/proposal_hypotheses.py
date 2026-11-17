"""Proposal hypotheses (docs/decisions.md D-28) on the primary Phase 3 results.
Reads results/phase3/test_metrics.csv and results/phase3/test_traces/; changes
nothing in the primary results.

Outputs (results/phase3/):
  proposal_hypotheses.csv  H1 and H2 against Base and B3
  category_analysis.csv    H3: mean improvement per NAB category
  detection_delay.csv      per method and seed: detected / missed windows, mean relative delay
  detection_delay_windows.csv  one row per test window x seed x method
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data import clip_windows, load_series, split_bounds  # noqa: E402
from pipeline.methods import METHODS  # noqa: E402

RES = ROOT / "results" / "phase3"
ALPHA = 0.05
SEEDS = [15, 42, 123]
BASELINES = ["Base", "B3"]


def per_file(test: pd.DataFrame, metric: str) -> pd.DataFrame:
    return test.groupby(["method", "file"])[metric].mean().unstack("method")


def h1(test: pd.DataFrame) -> list[dict]:
    pf = per_file(test, "f1")
    rows = []
    for b in BASELINES:
        d = pf["P1"] - pf[b]
        wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
        w = wilcoxon(d, alternative="two-sided")
        crit = wins >= 13
        sig = w.pvalue < ALPHA and d.mean() > 0
        rows.append({"hypothesis": "Proposal H1", "metric": "f1", "baseline": b, "n_files": len(d),
                     "mean_P1": pf["P1"].mean(), "mean_baseline": pf[b].mean(), "mean_diff": d.mean(),
                     "wins": wins, "ties": ties, "losses": losses, "criterion": "wins >= 13 of 18",
                     "criterion_met": crit, "test": "Wilcoxon", "statistic": w.statistic, "p": w.pvalue,
                     "verdict": "supported" if crit and sig else "not supported"})
    return rows


def h2(test: pd.DataFrame) -> list[dict]:
    pf = per_file(test, "hv_fpr")
    rows = []
    for b in BASELINES:
        x = pf[["P1", b]].dropna()
        d = x["P1"] - x[b]
        rel = (x[b].mean() - x["P1"].mean()) / x[b].mean()
        crit = rel >= 0.15
        for name, res in (("paired t-test", ttest_rel(x["P1"], x[b])), ("Wilcoxon", wilcoxon(d, alternative="two-sided"))):
            sig = res.pvalue < ALPHA and d.mean() < 0
            rows.append({"hypothesis": "Proposal H2", "metric": "hv_fpr", "baseline": b, "n_files": len(x),
                         "mean_P1": x["P1"].mean(), "mean_baseline": x[b].mean(), "mean_diff": d.mean(),
                         "relative_reduction": rel, "wins": int((d < 0).sum()), "ties": int((d == 0).sum()),
                         "losses": int((d > 0).sum()), "criterion": "relative reduction >= 15 %",
                         "criterion_met": crit, "test": name, "statistic": float(res.statistic),
                         "p": float(res.pvalue), "verdict": "supported" if crit and sig else "not supported"})
    return rows


def categories(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cat = test.groupby("file")["category"].first()
    for metric, higher in (("f1", True), ("composite_f1", True), ("hv_fpr", False)):
        pf = per_file(test, metric)
        for b in BASELINES:
            imp = (pf["P1"] - pf[b]) if higher else (pf[b] - pf["P1"])
            g = pd.DataFrame({"imp": imp, "category": cat}).groupby("category")["imp"]
            for c, s in g:
                rows.append({"metric": metric, "baseline": b, "category": c, "n_files": int(s.notna().sum()),
                             "mean_improvement": s.mean()})
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby(["metric", "baseline"])["mean_improvement"].rank(ascending=False, method="min")
    return df


def delays(eval_files: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in eval_files:
        s = load_series(f)
        _, _, te = split_bounds(len(s.values))
        wins = clip_windows(s.windows, te)
        for seed in SEEDS:
            z = np.load(RES / "test_traces" / f"{f.replace('.csv', '').replace('/', '__')}__s{seed}.npz")
            for m in METHODS:
                a = z[f"{m}__alarms"]
                for k, (lo, hi) in enumerate(wins):
                    hit = np.flatnonzero(a[lo:hi + 1])
                    rows.append({"file": f, "seed": seed, "method": m, "window": k, "start": lo, "end": hi,
                                 "length": hi - lo + 1, "detected": len(hit) > 0,
                                 "relative_delay": hit[0] / (hi - lo + 1) if len(hit) else np.nan})
    w = pd.DataFrame(rows)
    per_seed = w.groupby(["method", "seed"]).agg(n_windows=("detected", "size"), detected=("detected", "sum"),
                                                  mean_relative_delay=("relative_delay", "mean")).reset_index()
    per_seed["missed"] = per_seed.n_windows - per_seed.detected
    return w, per_seed


def main() -> None:
    test = pd.read_csv(RES / "test_metrics.csv")
    ph = pd.DataFrame(h1(test) + h2(test))
    ph.to_csv(RES / "proposal_hypotheses.csv", index=False)
    ca = categories(test)
    ca.to_csv(RES / "category_analysis.csv", index=False)
    ev = json.loads((ROOT / "configs" / "datasets.json").read_text())["evaluation"]
    w, d = delays(ev)
    w.to_csv(RES / "detection_delay_windows.csv", index=False)
    d.to_csv(RES / "detection_delay.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ph.drop(columns=["criterion"]).round(4).to_string(index=False))
        print(ca.round(4).to_string(index=False))
        print(d.groupby("method")[["detected", "missed", "mean_relative_delay"]].agg(["mean", "std"]).round(4)
              .reindex(list(METHODS)).to_string())


if __name__ == "__main__":
    main()
