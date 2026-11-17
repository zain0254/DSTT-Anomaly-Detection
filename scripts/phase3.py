"""Phase 3: tune every threshold method on validation, then score the test
segments once (D-26). Reads results/phase2_errors/. Writes results/phase3/:

  tuning_candidates.csv  one row per tuning file x seed x method x parameter set
  tuning_summary.csv     candidates averaged over the 24 files x 3 seeds
  selected_params.csv    one parameter set per method (objective + tie-breaks)
  r2_info.csv            R2 path per model file x seed (Eq. 8 / no-q / fallback)
  test_metrics.csv       one row per evaluation file x seed x method
  clean_metrics.csv      one row per clean file x seed x method (D-22)
  test_summary.csv       per method: mean and sd over seeds of across-file means
  test_traces/<file>__s<seed>.npz  test-segment alarms and thresholds per method

Usage: envs/pipeline/Scripts/python.exe scripts/phase3.py [--workers 4] [--stage tune|test|all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import metrics as MX  # noqa: E402
from pipeline import nab_scorer  # noqa: E402
from pipeline.data import clip_windows, load_series, split_bounds  # noqa: E402
from pipeline.methods import GRID_SETS, METHODS, SeriesContext, pkey, w_allowed  # noqa: E402

SEEDS = [15, 42, 123]
ERR = ROOT / "results" / "phase2_errors"
OUT = ROOT / "results" / "phase3"
OUT_BY_GRID = {"primary": ROOT / "results" / "phase3",
               "exploratory": ROOT / "results" / "phase3_exploratory"}


def errors_of(name: str, seed: int) -> np.ndarray:
    stem = name.replace(".csv", "").replace("/", "__")
    return np.load(ERR / f"{stem}__s{seed}.npz")["errors"]


def val_metrics(alarms: np.ndarray, labels: np.ndarray, windows) -> dict:
    pm = MX.point_metrics(alarms, labels)
    er = MX.event_recall(alarms, windows)
    nab = nab_scorer.score_file(alarms, windows, "standard", probation_percent=0.0)
    return {"composite_f1": MX.composite_f1(pm["precision"], er), "precision": pm["precision"],
            "event_recall": er, "alarm_rate": float(alarms.mean()),
            "nab_standard": nab_scorer.normalise(nab["raw"], nab["null"], nab["perfect"])}


def r2_row(name, seed, ctx) -> dict:
    t, info = ctx.r2()
    return {"file": name, "seed": seed, "threshold": t, **info}


def tune_one(args) -> tuple[list[dict], dict, list[dict]]:
    name, seed, grid_set = args
    s = load_series(name)
    tr, va, _ = split_bounds(len(s.values))
    ctx = SeriesContext(errors_of(name, seed), tr.stop)
    labels = s.labels[va]
    wins = clip_windows(s.windows, va)
    rows, skipped = [], []
    for m, spec in GRID_SETS[grid_set].items():
        for prm in spec["grid"]:
            if not w_allowed(prm, tr.stop):
                skipped.append({"stage": "tune", "file": name, "seed": seed, "method": m,
                                "params": pkey(prm), "errors_before_val": tr.stop - 29})
                continue
            a, _ = ctx.apply(m, prm)
            rows.append({"file": name, "seed": seed, "method": m, "params": pkey(prm),
                         **val_metrics(a[va], labels, wins)})
    return rows, r2_row(name, seed, ctx), skipped


def select(cand: pd.DataFrame, methods=METHODS) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = {(m, pkey(p)): i for m, spec in methods.items() for i, p in enumerate(spec["grid"])}
    summ = (cand.groupby(["method", "params"], as_index=False)
            .agg(composite_f1=("composite_f1", "mean"), nab_standard=("nab_standard", "mean"),
                 alarm_rate=("alarm_rate", "mean"), precision=("precision", "mean"),
                 event_recall=("event_recall", "mean"), n=("composite_f1", "size")))
    summ["grid_order"] = [order[(m, p)] for m, p in zip(summ.method, summ.params)]
    summ = summ.sort_values(["method", "composite_f1", "nab_standard", "alarm_rate", "grid_order"],
                            ascending=[True, False, False, True, True])
    sel = summ.groupby("method", as_index=False).head(1).reset_index(drop=True)
    return summ, sel


def test_one(args) -> tuple[list[dict], list[dict], dict, str, list[dict]]:
    name, seed, selected, is_eval, is_clean, grid_set = args
    s = load_series(name)
    tr, _, te = split_bounds(len(s.values))
    e = errors_of(name, seed)
    ctx = SeriesContext(e, tr.stop)
    hv = MX.high_variance_mask(s.values, tr)
    wins = clip_windows(s.windows, te)
    ev_rows, cl_rows, traces, subs = [], [], {}, []
    for m, prm in selected.items():
        if not w_allowed(prm, tr.stop):      # D-27: largest allowed w of the method's grid
            ws = sorted({g["w"] for g in GRID_SETS[grid_set][m]["grid"]})
            new_w = max(w for w in ws if w_allowed({"w": w}, tr.stop))
            subs.append({"stage": "test", "file": name, "seed": seed, "method": m, "params": pkey(prm),
                         "used_w": new_w, "errors_before_val": tr.stop - 29})
            prm = {**prm, "w": new_w}
        a, th = ctx.apply(m, prm)
        traces[f"{m}__alarms"] = a[te]
        traces[f"{m}__threshold"] = th[te]
        base = {"file": name, "category": s.category, "seed": seed, "method": m, "params": pkey(prm)}
        if is_eval:
            met = MX.evaluate_segment(a[te], e[te], s.labels[te], wins, hv[te])
            met["false_alarms_per_1000"] = 1000 * met["fp"] / len(a[te])
            ev_rows.append({**base, **met})
        if is_clean:
            cl_rows.append({**base, **MX.clean_stream_metrics(a[te])})
    traces["errors"] = e[te]
    traces["labels"] = s.labels[te]
    traces["test_start"] = np.array(te.start)
    return ev_rows, cl_rows, traces, name, subs


def summarise(test: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    metrics = ["composite_f1", "f1", "precision", "recall", "event_recall", "fpr", "hv_fpr",
               "false_alarms_per_1000", "nab_standard", "nab_standard_first", "roc_auc", "pr_auc", "alarm_rate"]
    per_seed = test.groupby(["method", "seed"])[metrics].mean()
    # corpus NAB per seed: sum raw / null / perfect over the 18 files, then normalise
    for prof in nab_scorer.PROFILES:
        for suf in ("", "_first"):
            k = f"nab_{prof}{suf}"
            g = test.groupby(["method", "seed"])[[f"{k}_raw", f"{k}_null", f"{k}_perfect"]].sum()
            per_seed[f"corpus_{k}"] = 100 * (g[f"{k}_raw"] - g[f"{k}_null"]) / (g[f"{k}_perfect"] - g[f"{k}_null"])
    cps = clean.groupby(["method", "seed"])[["clean_fpr", "clean_false_alarms_per_1000"]].mean()
    per_seed = per_seed.join(cps)
    mean = per_seed.groupby("method").mean().add_suffix("_mean")
    sd = per_seed.groupby("method").std(ddof=1).add_suffix("_sd")
    out = mean.join(sd)
    cols = [c for m in per_seed.columns for c in (f"{m}_mean", f"{m}_sd")]
    out = out[cols].reindex(list(METHODS)).reset_index()
    per_seed.reset_index().to_csv(OUT / "test_per_seed.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", choices=["tune", "test", "all"], default="all")
    ap.add_argument("--grids", choices=list(GRID_SETS), default="primary",
                    help="exploratory = D-27 post-hoc grid extension, written to results/phase3_exploratory/")
    args = ap.parse_args()
    global OUT
    OUT = OUT_BY_GRID[args.grids]
    methods = GRID_SETS[args.grids]
    OUT.mkdir(parents=True, exist_ok=True)
    log_rows = []
    ds = json.loads((ROOT / "configs" / "datasets.json").read_text())

    if args.stage in ("tune", "all"):
        t0 = time.perf_counter()
        jobs = [(f, s, args.grids) for f in ds["tuning"] for s in SEEDS]
        rows, r2 = [], []
        with ProcessPoolExecutor(args.workers) as ex:
            for i, (r, info, skipped) in enumerate(ex.map(tune_one, jobs), 1):
                rows += r
                r2.append(info)
                log_rows += skipped
                print(f"tune [{i}/{len(jobs)}] {info['file']} seed={info['seed']} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        cand = pd.DataFrame(rows)
        cand.to_csv(OUT / "tuning_candidates.csv", index=False)
        summ, sel = select(cand, methods)
        summ.to_csv(OUT / "tuning_summary.csv", index=False)
        sel.to_csv(OUT / "selected_params.csv", index=False)
        pd.DataFrame(r2).to_csv(OUT / "r2_info_tuning.csv", index=False)
        print(f"tuning done in {time.perf_counter() - t0:.0f}s")
        print(sel[["method", "params", "composite_f1", "nab_standard", "alarm_rate"]].to_string(index=False))

    if args.stage in ("test", "all"):
        t0 = time.perf_counter()
        sel = pd.read_csv(OUT / "selected_params.csv")
        selected = {m: json.loads(p) for m, p in zip(sel.method, sel.params)}
        files = sorted(set(ds["evaluation"]) | set(ds["clean_test"]))
        jobs = [(f, s, selected, f in ds["evaluation"], f in ds["clean_test"], args.grids)
                for f in files for s in SEEDS]
        ev, cl = [], []
        tr_dir = OUT / "test_traces"
        tr_dir.mkdir(exist_ok=True)
        with ProcessPoolExecutor(args.workers) as ex:
            for i, ((job_name, job_seed, *_), (e_rows, c_rows, traces, name, subs)) in enumerate(zip(jobs, ex.map(test_one, jobs)), 1):
                ev += e_rows
                cl += c_rows
                log_rows += subs
                np.savez_compressed(tr_dir / f"{name.replace('.csv', '').replace('/', '__')}__s{job_seed}.npz", **traces)
        test = pd.DataFrame(ev)
        clean = pd.DataFrame(cl)
        test.to_csv(OUT / "test_metrics.csv", index=False)
        clean.to_csv(OUT / "clean_metrics.csv", index=False)
        # R2 path for every model file x seed (evaluation and clean files too)
        r2 = []
        for f in sorted(set(ds["model_files"])):
            for sd in SEEDS:
                s = load_series(f)
                ctx = SeriesContext(errors_of(f, sd), split_bounds(len(s.values))[0].stop)
                r2.append(r2_row(f, sd, ctx))
        pd.DataFrame(r2).to_csv(OUT / "r2_info.csv", index=False)
        summary = summarise(test, clean)
        summary.to_csv(OUT / "test_summary.csv", index=False)
        print(f"test scoring done in {time.perf_counter() - t0:.0f}s: {len(test)} eval rows, {len(clean)} clean rows")
    if args.grids != "primary":
        pd.DataFrame(log_rows, columns=["stage", "file", "seed", "method", "params", "errors_before_val",
                                        "used_w"]).to_csv(OUT / "window_log.csv", index=False)
        print(f"window log: {len(log_rows)} skipped or substituted candidates")


if __name__ == "__main__":
    main()
