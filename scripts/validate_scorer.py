"""Validate pipeline/nab_scorer.py against NAB's published results.

For each detector shipped in data/nab/results/ (fetched by fetch_nab.sh) and
each profile: threshold the detector's anomaly scores at its optimised
threshold from config/thresholds.json (alarm = score >= threshold), score every
file with our port (15 % probationary period, as NAB), and compare
  - per-file raw scores with results/<det>/<det>_<profile>_scores.csv
  - the corpus raw total with the "Totals" row of that file
  - the normalised score with results/final_results.json
Normalisation follows nab/runner.py: null baseline = our corpus raw score of
the null detector, perfect = tpWeight * number of labelled windows (116).

Output: results/scorer_validation.csv (one row per detector x profile) and
results/scorer_validation_files.csv (per-file comparison).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import nab_scorer  # noqa: E402
from pipeline.data import NAB_DIR, list_files, load_series, load_windows  # noqa: E402

DETECTORS = ["numenta", "windowedGaussian", "relativeEntropy", "null"]
TOL = 1e-6


def corpus_scores(det: str, profile: str, threshold: float, windows: dict) -> tuple[float, pd.DataFrame]:
    nab_files = pd.read_csv(NAB_DIR / "results" / det / f"{det}_{profile}_scores.csv")
    nab_by_file = dict(zip(nab_files["File"], nab_files["Score"]))
    rows = []
    for name in list_files():
        cat, fname = name.split("/")
        res = pd.read_csv(NAB_DIR / "results" / det / cat / f"{det}_{fname}")
        s = load_series(name, windows=windows[name])
        if len(res) != len(s.values) or not (pd.to_datetime(res["timestamp"]).values == s.timestamps.values).all():
            raise ValueError(f"{det}: timestamps of results file differ from data file for {name}")
        alarms = res["anomaly_score"].to_numpy(dtype=float) >= threshold
        r = nab_scorer.score_file(alarms, s.windows, profile)
        rows.append({"detector": det, "profile": profile, "file": name, "ours": r["raw"],
                     "nab": float(nab_by_file[name])})
    df = pd.DataFrame(rows)
    df["abs_diff"] = (df["ours"] - df["nab"]).abs()
    nab_total = float(nab_files.loc[nab_files["Detector"] == "Totals", "Score"].iloc[0]) \
        if "Totals" in set(nab_files["Detector"]) else float("nan")
    return nab_total, df


def main() -> None:
    windows = load_windows()
    n_windows = sum(len(v) for v in windows.values())
    thresholds = json.loads((NAB_DIR / "config" / "thresholds.json").read_text())
    final = json.loads((NAB_DIR / "results" / "final_results.json").read_text())

    per_file, summary, raw_total = [], [], {}
    for det in DETECTORS:
        for profile in nab_scorer.PROFILES:
            th = thresholds[det][profile]["threshold"]
            nab_total, df = corpus_scores(det, profile, th, windows)
            per_file.append(df)
            raw_total[(det, profile)] = (df["ours"].sum(), nab_total, df["abs_diff"].max(), th)

    for (det, profile), (ours, nab_total, max_file_diff, th) in raw_total.items():
        null = raw_total[("null", profile)][0]
        perfect = nab_scorer.PROFILES[profile]["tpWeight"] * n_windows
        norm = nab_scorer.normalise(ours, null, perfect)
        ref = final[det][profile]
        summary.append({"detector": det, "profile": profile, "threshold": th,
                        "raw_ours": ours, "raw_nab_totals": nab_total,
                        "max_file_abs_diff": max_file_diff,
                        "normalised_ours": norm, "final_results_json": ref,
                        "normalised_abs_diff": abs(norm - ref),
                        "match": bool(abs(norm - ref) < TOL and max_file_diff < TOL)})

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    s = pd.DataFrame(summary)
    s.to_csv(out / "scorer_validation.csv", index=False)
    pd.concat(per_file).to_csv(out / "scorer_validation_files.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s.to_string(index=False))
    print("ALL MATCH" if s["match"].all() else "MISMATCH")
    sys.exit(0 if s["match"].all() else 1)


if __name__ == "__main__":
    main()
