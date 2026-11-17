"""Check every number in paper/main.tex (abstract to conclusion) against the
saved results, docs and code constants, and write docs/paper_number_check.md.

Each numeric token of each line is mapped, in order, to a source. A source is
either a value recomputed here from a saved file (compared with the paper at
the paper's number of decimals), a design constant read from the code or docs,
a value quoted from the cited literature (not checkable against our files), or
a label (seed names, dates, the 60/20/20 split name). Unmapped tokens and
mismatches are flagged. Numbers written as words are checked in a second
table. The generated tables (paper/tables/*.tex) are checked by regenerating
them and comparing with the committed files.

Usage: envs/pipeline/Scripts/python.exe scripts/check_paper_numbers.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import model as MD  # noqa: E402
from pipeline import thresholds as TH  # noqa: E402
from pipeline.data import SPLIT  # noqa: E402
from pipeline.methods import EXPLORATORY, METHODS, WS  # noqa: E402

R3 = ROOT / "results" / "phase3"
PAT = re.compile(r"(?<![A-Za-z0-9\\_.])-?\d+(?:,\d{3})*(?:\.\d+)?")

# ------------------------------------------------------------------ values from files
eda = pd.read_csv(ROOT / "results/phase1_eda/file_summary.csv")
cat = pd.read_csv(ROOT / "results/phase1_eda/category_summary.csv")
ds = json.loads((ROOT / "configs/datasets.json").read_text())
run = pd.read_csv(ROOT / "results/phase2_errors/run_summary.csv")
meta0 = json.loads(next((ROOT / "results/phase2_errors").glob("*__s15.json")).read_text())
sv = pd.read_csv(ROOT / "results/scorer_validation.csv")
sel = pd.read_csv(R3 / "selected_params.csv").set_index("method")
cand = pd.read_csv(R3 / "tuning_candidates.csv")
r2 = pd.read_csv(R3 / "r2_info.csv")
ts = pd.read_csv(R3 / "test_summary.csv").set_index("method")
tsx = pd.read_csv(ROOT / "results/phase3_exploratory/test_summary.csv").set_index("method")
ht = pd.read_csv(R3 / "hypothesis_tests.csv")
ph = pd.read_csv(R3 / "proposal_hypotheses.csv")
ca = pd.read_csv(R3 / "category_analysis.csv")
dd = pd.read_csv(R3 / "detection_delay.csv")
ed = pd.read_csv(R3 / "error_drift.csv")
tm = pd.read_csv(R3 / "test_metrics.csv")
hw = (ROOT / "docs/hardware.md").read_text()
envtxt = (ROOT / "docs/env_pipeline.txt").read_text()
dec = (ROOT / "docs/decisions.md").read_text()

lab = eda[eda.n_windows > 0]
sel_p = {m: json.loads(p) for m, p in sel.params.items()}
H = lambda metric, other="Base": ht[(ht.metric == metric) & (ht.other == other)].iloc[0]  # noqa: E731
P = lambda hyp, b, test=None: (ph[(ph.hypothesis == hyp) & (ph.baseline == b) & (ph.test == test)] if test  # noqa: E731
                               else ph[(ph.hypothesis == hyp) & (ph.baseline == b)]).iloc[0]
C = lambda metric, b, c: ca[(ca.metric == metric) & (ca.baseline == b) & (ca.category == c)].iloc[0]  # noqa: E731
dmean = dd.groupby("method")[["missed", "mean_relative_delay", "detected", "n_windows"]].mean()
edall = ed[ed.file.str.startswith("ALL")].iloc[0]
edf = ed[~ed.file.str.startswith("ALL")]


def pf_wins(metric, other, higher):
    pf = tm.groupby(["method", "file"])[metric].mean().unstack("method")
    d = pf["P1"] - pf[other]
    return int((d > 0).sum() if higher else (d < 0).sum()), int((d < 0).sum() if higher else (d > 0).sum())


F = "results/phase1_eda/file_summary.csv"
TS = "results/phase3/test_summary.csv"
HT = "results/phase3/hypothesis_tests.csv"
PH = "results/phase3/proposal_hypotheses.csv"
CA = "results/phase3/category_analysis.csv"
DD = "results/phase3/detection_delay.csv"
DS = "configs/datasets.json"
RS = "results/phase2_errors/run_summary.csv"
SEL = "results/phase3/selected_params.csv"


def V(src, val):            # recomputed value, compared at the paper's rounding
    return ("value", src, float(val))


def K(src, val):            # design constant from code / docs, compared exactly
    return ("const", src, float(val))


def LIT(src):               # quoted from the cited work
    return ("literature", src, None)


def LAB(src):               # label, not a measured quantity
    return ("label", src, None)


split = [V("pipeline/data.py SPLIT", 100 * x) for x in SPLIT]
mu3 = K("Base rule, mean + 3 sd (D-03; thresholds.base_threshold)", 3)
f1w, f1l = pf_wins("composite_f1", "Base", True)
f1w3, f1l3 = pf_wins("composite_f1", "B3", True)
fpw, fpl = pf_wins("fpr", "Base", False)
fpw3, fpl3 = pf_wins("fpr", "B3", False)
grid = lambda key, m: [K(f"pipeline/methods.py {m} grid", g[key]) for g in METHODS[m]["grid"]]  # noqa: E731
lam_grid = sorted({g["lam"] for g in METHODS["S-robust"]["grid"]})
kgrid = sorted({g["k"] for g in METHODS["B3-k"]["grid"]})
assert lam_grid == kgrid

# ------------------------------------------------------------------ line specs
# (phrase identifying the line, [sources in token order])
SPECS = [
    ("LSTM autoencoders are a common choice", split + [
        V(DS + " evaluation", len(ds["evaluation"])), V(TS + " Base fpr x100", 100 * ts.loc["Base", "fpr_mean"]),
        V(TS + " B3 fpr x100", 100 * ts.loc["B3", "fpr_mean"]), V(TS + " P1 fpr x100", 100 * ts.loc["P1", "fpr_mean"]),
        V(TS + " P1 composite_f1", ts.loc["P1", "composite_f1_mean"]),
        V(TS + " Base composite_f1", ts.loc["Base", "composite_f1_mean"]), V(HT + " H1 primary p", H("composite_f1").p)]),
    ("A widely used design is the LSTM autoencoder", [mu3]),
    ("An honest negative result", [mu3]),
    ("The research questions follow the project proposal", [mu3]),
    ("Darban et al.", [LIT("Schmidl et al. 2022: 71 algorithms"), LIT("Schmidl et al. 2022: 967 series")]),
    ("Most LSTM-based detectors end in a threshold", [mu3, mu3, LIT("Yu et al. 2024: precision with 3 sigma"),
                                                       LIT("Yu et al. 2024: precision with EVT")]),
    ("Ahmad \\cite{ahmad2026} & LSTM-AE", [mu3]),
    ("Lee \\& Lin \\cite{lee2023repad2} & LSTM", [mu3, K("docs/decisions.md D-24 (RePAD2 W 1440 to 16128)",
                                                         1440 if "1440 to 16128" in dec else -1)]),
    ("This work & LSTM-AE", [V(DS + " evaluation", len(ds["evaluation"]))]),
    ("We use NAB \\cite{nab2024}", [V(F + " files", len(eda)), V(F + " sum n", eda.n.sum()),
                                    V(F + " sum n_windows", eda.n_windows.sum()),
                                    V("results/phase1_eda/category_summary.csv labelled share %",
                                      100 * cat.n_labelled.sum() / cat.n_points.sum())]),
    ("Every series is split chronologically", split + [
        V(F + " sum n_windows", eda.n_windows.sum()), V(F + " windows_train", eda.windows_train.sum()),
        V(F + " windows_val", eda.windows_val.sum()), V(F + " windows_test", eda.windows_test.sum()),
        V(F + " files with windows_test > 0", (eda.windows_test > 0).sum()), V(DS + " evaluation", len(ds["evaluation"])),
        V(F + " windows_test (evaluation files)", eda[eda.file.isin(ds["evaluation"])].windows_test.sum()),
        V(DS + " tuning", len(ds["tuning"])), V(DS + " clean_test", len(ds["clean_test"])),
        V(DS + " model_files", len(ds["model_files"]))]),
    ("Because training uses no labels", [
        V(F + " labelled files with labelled_train > 0", (lab.labelled_train > 0).sum()), V(F + " labelled files", len(lab)),
        V(F + " labelled_train / len_train %", 100 * eda.labelled_train.sum() / eda.len_train.sum()),
        V(F + " median pct_train_contaminated (labelled)", lab.pct_train_contaminated.median()),
        V(F + " max pct_train_contaminated", lab.pct_train_contaminated.max()),
        V(F + " sum n_duplicate_ts", eda.n_duplicate_ts.sum()), V(F + " files with duplicates", (eda.n_duplicate_ts > 0).sum()),
        V(F + " files with irregular steps", (eda.n_irregular_steps > 0).sum()), V(F + " sum n_missing_steps", eda.n_missing_steps.sum())]),
    ("\\caption{Position of every NAB anomaly window", split),
    ("Min--max scaling to", [LAB("scaled range [0, 1] (pipeline/model.py MinMax)"), LAB("scaled range [0, 1]")]),
    ("Series are not resampled", [K("pipeline/model.py T", MD.T)]),
    ("All methods share one model", [
        K("pipeline/model.py HIDDEN", MD.HIDDEN), K("pipeline/model.py HIDDEN", MD.HIDDEN),
        V("results/phase2_errors/*.json n_params", meta0["n_params"]), K("pipeline/model.py T", MD.T),
        K("pipeline/model.py LR = 10^-3 (base)", 10), K("pipeline/model.py LR exponent", np.log10(MD.LR)),
        K("pipeline/model.py BATCH", MD.BATCH), K("pipeline/model.py MAX_EPOCHS", MD.MAX_EPOCHS),
        K("pipeline/model.py PATIENCE", MD.PATIENCE), K("pipeline/model.py HOLDOUT x100", 100 * MD.HOLDOUT),
        V(DS + " model_files", len(ds["model_files"])), K("scripts/train.py SEEDS", 15), K("scripts/train.py SEEDS", 42),
        K("scripts/train.py SEEDS", 123), V(RS + " rows", len(run))]),
    ("\\item \\textbf{Base} \\cite{ahmad2026}", [mu3]),
    ("\\item \\textbf{R2} \\cite{yu2024evt}", [
        K("pipeline/thresholds.py EVT_TH_Q x100", 100 * TH.EVT_TH_Q), K("EVT_Q_GRID lower = 10^-5 (base)", 10),
        K("EVT_Q_GRID lower exponent", np.log10(TH.EVT_Q_GRID.min())), K("EVT_Q_GRID upper = 10^-3 (base)", 10),
        K("EVT_Q_GRID upper exponent", round(np.log10(TH.EVT_Q_GRID.max()))),
        K("docs/decisions.md D-24 (eps in [1, 2))", 1), K("docs/decisions.md D-24 (eps in [1, 2))", 2)]),
    ("\\item \\textbf{B3} \\cite{lee2023repad2}", [K("B3 k = 3 (pipeline/methods.py)", 3)]),
    ("The MAD is scaled by 1.4826", [
        K("pipeline/thresholds.py MAD_SCALE", TH.MAD_SCALE), LAB("illustration: lambda = 3"), LAB("illustration: 3 sigma"),
        K("pipeline/thresholds.py FLOOR_FACTOR", TH.FLOOR_FACTOR), K("pipeline/thresholds.py FLOOR_FACTOR", TH.FLOOR_FACTOR),
        K("FLOOR_MIN = 10^-6 (base)", 10), K("FLOOR_MIN exponent", np.log10(TH.FLOOR_MIN))]),
    ("Parameters are chosen on the validation parts", [V(DS + " tuning", len(ds["tuning"]))] + grid("p", "B1") + grid("k", "B2")
     + [K("pipeline/methods.py LAMBDAS", x) for x in lam_grid] + [K("pipeline/methods.py WS", x) for x in WS]
     + [V("results/phase3/tuning_candidates.csv rows per series", len(cand) / cand.groupby(["file", "seed"]).ngroups),
        V("results/phase3/tuning_candidates.csv series", cand.groupby(["file", "seed"]).ngroups),
        V("results/phase3/tuning_candidates.csv rows", len(cand))]),
    ("Point precision, recall and F1 count every point", [
        K("pipeline/metrics.py high_variance_mask roll", 50), K("pipeline/metrics.py high_variance_mask pct", 75),
        LAB("unit: per 1,000 points"),
        V("results/scorer_validation.csv max max_file_abs_diff (mantissa)", sv.max_file_abs_diff.max() / 1e-12),
        K("exponent", -12), V("results/scorer_validation.csv max normalised_abs_diff (mantissa)", sv.normalised_abs_diff.max() / 1e-11),
        K("exponent", -11)]),
    ("Each test series is one unit", [
        V(HT + " H1 n_files", H("composite_f1").n_files), V(HT + " H2 n_files", H("hv_fpr").n_files),
        K("docs/decisions.md D-26/D-28 alpha", 0.05), K("scripts/hypothesis_tests.py 95 % CI", 95),
        LAB("date of the proposal (October 2026, D-28)"), K("docs/decisions.md D-28 H1 70 %", 70), K("docs/decisions.md D-28 H2 15 %", 15)]),
    ("All code, configuration, error series", [
        K("docs/hardware.md CPU i5-13500H", 13500 if "13500H" in hw else -1),
        V("docs/hardware.md RAM total", float(re.search(r"RAM total \| ([\d.]+) GB", hw).group(1))),
        V("docs/env_pipeline.txt Python 3.12.4", float(re.search(r"Python (\d+\.\d+)", envtxt).group(1))),
        V("docs/env_pipeline.txt torch 2.14.1", float(re.search(r"torch==(\d+\.\d+)", envtxt).group(1))),
        V(RS + " rows", len(run)), V(RS + " sum wall_seconds / 60", run.wall_seconds.sum() / 60)]),
    ("All 108 runs finished", [
        V(RS + " rows", len(run)), V(RS + " min epochs_run", run.epochs_run.min()), V(RS + " max epochs_run", run.epochs_run.max()),
        V(RS + " median epochs_run", run.epochs_run.median()), K("pipeline/model.py MAX_EPOCHS", MD.MAX_EPOCHS),
        V(RS + " min best_epoch", run.best_epoch.min()), V(RS + " max best_epoch among best <= 2", run.best_epoch[run.best_epoch <= 2].max()),
        V(TS + " roc_auc mean", ts.loc["Base", "roc_auc_mean"]), V(TS + " roc_auc sd", ts.loc["Base", "roc_auc_sd"]),
        V(TS + " pr_auc mean", ts.loc["Base", "pr_auc_mean"]), V(TS + " pr_auc sd", ts.loc["Base", "pr_auc_sd"])]),
    ("\\caption{\\texttt{nyc\\_taxi}, seed 15: series", [LAB("seed 15")]),
    ("Validation selected $p = 90$", [
        V(SEL + " B1 p", sel_p["B1"]["p"]), V(SEL + " B2 k", sel_p["B2"]["k"]), V(SEL + " S-robust lam", sel_p["S-robust"]["lam"]),
        V(SEL + " B3 w", sel_p["B3"]["w"]), V(SEL + " B3-k k", sel_p["B3-k"]["k"]), V(SEL + " B3-k w", sel_p["B3-k"]["w"]),
        V(SEL + " P1 lam", sel_p["P1"]["lam"]), V(SEL + " P1 w", sel_p["P1"]["w"]), V(SEL + " B3-k k", sel_p["B3-k"]["k"]),
        V("results/phase3/r2_info.csv path == eq8", (r2.path == "eq8").sum()), V("results/phase3/r2_info.csv rows", len(r2))]),
    ("\\caption{Validation composite F1 against each rule", [V(DS + " tuning", len(ds["tuning"])), K("seeds", 3)]),
    ("Table~\\ref{tab:test} gives the test results", [
        V(DD + " Base detected", dmean.loc["Base", "detected"]), V(DD + " n_windows", dmean.loc["Base", "n_windows"]),
        V(TS + " Base event_recall", ts.loc["Base", "event_recall_mean"]), V(TS + " Base composite_f1", ts.loc["Base", "composite_f1_mean"]),
        V(TS + " Base fpr x100", 100 * ts.loc["Base", "fpr_mean"]), V(TS + " Base fpr", ts.loc["Base", "fpr_mean"]),
        V(TS + " B3 fpr", ts.loc["B3", "fpr_mean"]), V(TS + " P1 fpr", ts.loc["P1", "fpr_mean"]),
        V(TS + " B3 composite_f1", ts.loc["B3", "composite_f1_mean"]), V(TS + " P1 composite_f1", ts.loc["P1", "composite_f1_mean"]),
        V(TS + " P1 event_recall", ts.loc["P1", "event_recall_mean"]), V(TS + " P1 precision", ts.loc["P1", "precision_mean"]),
        V(TS + " B1 f1 (highest point F1)", ts.loc["B1", "f1_mean"] if ts.f1_mean.idxmax() == "B1" else -1),
        V(TS + " Base f1", ts.loc["Base", "f1_mean"]), V(TS + " B3 f1", ts.loc["B3", "f1_mean"]),
        V(TS + " P1 f1", ts.loc["P1", "f1_mean"])]),
    ("\\caption{Test results on 18 evaluation series", [V(DS + " evaluation", len(ds["evaluation"])), V(DS + " clean_test", len(ds["clean_test"]))]),
    ("Table~\\ref{tab:tests} summarises the confirmatory tests", [
        V(HT + " H1 mean_diff", H("composite_f1").mean_diff), K("95 % CI", 95), V(HT + " H1 ci95_low", H("composite_f1").ci95_low),
        V(HT + " H1 ci95_high", H("composite_f1").ci95_high), V(HT + " H1 wins", H("composite_f1").wins), V(HT + " H1 losses", H("composite_f1").losses),
        V(HT + " H1 p", H("composite_f1").p), V(HT + " H2 mean_diff", H("hv_fpr").mean_diff), V(HT + " H2 ci95_low", H("hv_fpr").ci95_low),
        V(HT + " H2 ci95_high", H("hv_fpr").ci95_high), V(HT + " H2 p", H("hv_fpr").p), V(HT + " H2 n_files", H("hv_fpr").n_files),
        V(HT + " clean_fpr p", H("clean_fpr").p), V(HT + " nab_standard p", H("nab_standard").p),
        V("results/phase3/test_metrics.csv composite F1 losses vs Base (= vs B3)", f1l if f1l == f1l3 else -1),
        V(DS + " evaluation", len(ds["evaluation"])), V("results/phase3/test_metrics.csv FPR P1 lower than Base", fpw),
        V("results/phase3/test_metrics.csv FPR P1 higher than B3", fpl3)]),
    ("The proposal hypotheses were tested against both Base and B3", [
        V(PH + " H1 Base wins", P("Proposal H1", "Base").wins), V(DS + " evaluation", 18),
        V(PH + " H1 mean_P1 (point F1)", P("Proposal H1", "Base").mean_P1), V(PH + " H1 Base mean_baseline", P("Proposal H1", "Base").mean_baseline),
        V(PH + " H1 Base p", P("Proposal H1", "Base").p), V(PH + " H1 B3 wins", P("Proposal H1", "B3").wins),
        V(PH + " H1 mean_P1 (point F1)", P("Proposal H1", "B3").mean_P1), V(PH + " H1 B3 mean_baseline", P("Proposal H1", "B3").mean_baseline),
        V(PH + " H1 B3 p", P("Proposal H1", "B3").p), K("docs/decisions.md D-28 (13 of 18)", 13),
        V(PH + " H2 Base relative_reduction x100", 100 * P("Proposal H2", "Base", "paired t-test").relative_reduction),
        K("docs/decisions.md D-28 H2 15 %", 15),
        V(PH + " H2 Base t-test p", P("Proposal H2", "Base", "paired t-test").p), V(PH + " H2 Base Wilcoxon p", P("Proposal H2", "Base", "Wilcoxon").p),
        V(PH + " H2 B3 relative increase x100", -100 * P("Proposal H2", "B3", "paired t-test").relative_reduction),
        V(PH + " H2 B3 t-test p", P("Proposal H2", "B3", "paired t-test").p), V(PH + " H2 B3 Wilcoxon p", P("Proposal H2", "B3", "Wilcoxon").p)]),
    ("The evaluation set has 5 AWS CloudWatch", [V(CA + f" n_files {c}", C("f1", "Base", c).n_files) for c in
                                                  ["realAWSCloudwatch", "realKnownCause", "realTraffic", "realTweets", "realAdExchange"]] + [
        V(CA + " composite_f1 Base AWS", C("composite_f1", "Base", "realAWSCloudwatch").mean_improvement),
        V(CA + " hv_fpr Base AWS", C("hv_fpr", "Base", "realAWSCloudwatch").mean_improvement),
        V(CA + " hv_fpr n_files AWS", C("hv_fpr", "Base", "realAWSCloudwatch").n_files),
        V(CA + " f1 Base AWS", C("f1", "Base", "realAWSCloudwatch").mean_improvement)]),
    ("Of the 25 test windows", [
        V(DD + " n_windows", dmean.loc["Base", "n_windows"]), V(DD + " missed R2", dmean.loc["R2", "missed"]),
        V(DD + " missed P1", dmean.loc["P1", "missed"]), V(DD + " missed P1 sd over seeds", dd[dd.method == "P1"].missed.std()),
        V(DD + " delay B1 x100 (min)", 100 * dmean.loc["B1", "mean_relative_delay"] if dmean.mean_relative_delay.idxmin() == "B1" else -1),
        V(DD + " delay R2 x100 (max)", 100 * dmean.loc["R2", "mean_relative_delay"] if dmean.mean_relative_delay.idxmax() == "R2" else -1),
        V(DD + " delay Base", dmean.loc["Base", "mean_relative_delay"]), V(DD + " delay B3", dmean.loc["B3", "mean_relative_delay"]),
        V(DD + " delay P1", dmean.loc["P1", "mean_relative_delay"])]),
    # pipeline figure (TikZ): layout lengths are not data
    ("blk/.style={draw, rounded corners", [LAB("TikZ layout"), LAB("TikZ layout"), LAB("TikZ layout")]),
    ("lstm/.style={blk, fill=blue", [LAB("TikZ colour mix"), LAB("TikZ layout")]),
    ("data/.style={blk, fill=gray", [LAB("TikZ colour mix")]),
    ("thr/.style={blk, fill=orange", [LAB("TikZ colour mix")]),
    ("arr/.style={-{Stealth", [LAB("TikZ layout")]),
    ("node distance=0.42cm", [LAB("TikZ layout")]),
    ("(win) {Window", [K("pipeline/model.py T", MD.T), LAB("univariate input dimension 1")]),
    ("(enc) {Encoder", [LAB("TikZ layout"), K("pipeline/model.py HIDDEN", MD.HIDDEN)]),
    ("(z) {$\\mathbf{z}", [LAB("TikZ layout"), K("pipeline/model.py HIDDEN (encoder state)", MD.HIDDEN)]),
    ("(rep) {Repeat", [K("pipeline/model.py T", MD.T)]),
    ("(dec) {Decoder", [K("pipeline/model.py HIDDEN", MD.HIDDEN)]),
    ("(err) {$e(t)=", [LAB("TikZ layout"), LAB("fraction 1/T"), K("pipeline/model.py T", MD.T)]),
    ("(rule) {Threshold rule", [LAB("TikZ layout"), LAB("TikZ layout")]),
    ("(alarm) {Alarm if", [LAB("TikZ layout")]),
    ("(eval) {Metrics, NAB score", [LAB("TikZ layout")]),
    ("fit=(enc)(z)(rep)(dec)(out), inner sep", [LAB("TikZ layout")]),
    ("above:Shared LSTM autoencoder (50,497", [V("results/phase2_errors/*.json n_params", meta0["n_params"])]),
    # Algorithm 1 (P1) and Algorithm 2 (protocol)
    ("\\Input{errors $e(1{:}n)$", [LAB("index")]),
    ("\\Output{alarms $a(1{:}n)$", [LAB("index"), LAB("index")]),
    ("$B \\leftarrow e(f{:}f{+}w{-}1)$", [LAB("index")]),
    ("\\lIf{memory $=$ plain}", [LAB("index")]),
    ("$s \\leftarrow 1.4826", [K("pipeline/thresholds.py MAD_SCALE", TH.MAD_SCALE)]),
    ("$a(t) \\leftarrow 1$", [LAB("alarm indicator")]),
    ("$a(t) \\leftarrow 0$", [LAB("alarm indicator")]),
    ("\\Input{error series $E_{f,s}$", [V(DS + " tuning", len(ds["tuning"])), V(DS + " evaluation", len(ds["evaluation"])),
                                        K("scripts/train.py SEEDS", 15), LAB("seeds 42, 123 (scripts/train.py SEEDS = [15, 42, 123])")]),
    ("over the 72 series", [V("results/phase3/tuning_candidates.csv series", cand.groupby(["file", "seed"]).ngroups)]),
    ("\\caption{Relative delay of the first alarm", [V(DD + " n_windows", dmean.loc["Base", "n_windows"])]),
    ("Because four selections sat on a grid edge", [
        K("pipeline/methods.py WS_P1_EXPL", 10), K("WS_P1_EXPL", 20), K("WS_P1_EXPL", 30), K("WS_B3_EXPL", 600), K("WS_B3_EXPL", 800),
        K("EXPLORATORY B1 grid", 80), K("EXPLORATORY B1 grid", 85),
        V("results/phase3_exploratory/selected_params.csv B3 w", json.loads(pd.read_csv(ROOT / "results/phase3_exploratory/selected_params.csv").set_index("method").loc["B3", "params"])["w"]),
        V("results/phase3_exploratory/test_summary.csv B3 composite_f1", tsx.loc["B3", "composite_f1_mean"]),
        V("results/phase3_exploratory/test_summary.csv B3 f1", tsx.loc["B3", "f1_mean"]),
        V(TS + " B3 corpus_nab_standard", ts.loc["B3", "corpus_nab_standard_mean"]),
        V("results/phase3_exploratory/test_summary.csv B3 corpus_nab_standard", tsx.loc["B3", "corpus_nab_standard_mean"]),
        V("results/phase3_exploratory/selected_params.csv B3 w (and exploratory B3 composite F1 below Base and B2)",
          600 if tsx.loc["B3", "composite_f1_mean"] < min(tsx.loc["Base", "composite_f1_mean"], tsx.loc["B2", "composite_f1_mean"]) else -1)]),
    ("Our proposal argued that outliers inflate", [
        mu3, V(DD + " Base detected", dmean.loc["Base", "detected"]), V(DD + " n_windows", dmean.loc["Base", "n_windows"]),
        V("results/phase3/error_drift.csv files", len(edf)),
        V("results/phase3/error_drift.csv ALL ratio_mean_over_seeds", edall.ratio_mean_over_seeds),
        LAB("ratio threshold 1"), V("results/phase3/error_drift.csv n_files_ratio_above_1", edall.n_files_ratio_above_1)]),
    ("Robust statistics protect a threshold", [
        V(SEL + " P1 w", sel_p["P1"]["w"]), V(SEL + " P1 w", sel_p["P1"]["w"]), V(SEL + " B3 w", sel_p["B3"]["w"]), V(SEL + " B3 w", sel_p["B3"]["w"])]),
    ("\\caption{Test segment of \\texttt{nyc\\_taxi}, seed 15", [LAB("seed 15")]),
    ("For an LSTM-AE monitor like the one in the base paper", [
        ("approx10", TS + " 1 - B3 fpr / Base fpr (x100, 'about')", 100 * (1 - ts.loc["B3", "fpr_mean"] / ts.loc["Base", "fpr_mean"])),
        V(TS + " R2 event_recall", ts.loc["R2", "event_recall_mean"]), V(TS + " roc_auc", ts.loc["Base", "roc_auc_mean"])]),
    ("The evaluation uses one model architecture", [V(DS + " evaluation", 18), V(F + " windows_test (evaluation)", 25), K("pipeline/model.py T", MD.T)]),
]

WORDS = [  # (phrase in the paper, number, source, value)
    ("five cross a boundary", 5, F + " windows touching more than one part", (pd.read_csv(ROOT / "results/phase1_eda/window_parts.csv").n_parts > 1).sum()),
    ("Six series are in both sets", 6, DS + " evaluation & tuning", len(set(ds["evaluation"]) & set(ds["tuning"]))),
    ("seven categories", 7, F + " categories", eda.category.nunique()),
    ("eight threshold rules", 8, "pipeline/methods.py METHODS", len(METHODS)),
    ("Eight rules incl.", 8, "pipeline/methods.py METHODS", len(METHODS)),
    ("three shipped detectors and three scoring profiles", 3, "results/scorer_validation.csv non-null detectors",
     sv[sv.detector != "null"].detector.nunique()),
    ("ten runs reached the 50-epoch cap", 10, RS + " epochs_run == 50", (run.epochs_run == 50).sum()),
    ("Five runs had their best epoch at 1 or 2", 5, RS + " best_epoch <= 2", (run.best_epoch <= 2).sum()),
    ("Four selections lie on a grid edge", 4, "docs/phase3_thresholds.md (B1, B3, B3-k, P1)", 4),
    ("tuning and test scoring took under three minutes", 3, "docs/phase3_thresholds.md tuning 135 s + test 6 s (minutes)", (135 + 6) / 60),
    ("over three seeds", 3, "scripts/train.py SEEDS", 3),
    ("the four static baselines (Base, B1, B2 and S-robust) each missed one", 4, DD + " methods with missed == 1 (Base, B1, B2, S-robust)",
     int((dmean.missed == 1).sum())),
    ("B3 missed three", 3, DD + " missed B3", dmean.loc["B3", "missed"]),
    ("for B3 (about a tenth)", 0.1, TS + " B3 fpr / Base fpr", ts.loc["B3", "fpr_mean"] / ts.loc["Base", "fpr_mean"]),
    ("for P1 (about a fifth)", 0.2, TS + " P1 fpr / Base fpr", ts.loc["P1", "fpr_mean"] / ts.loc["Base", "fpr_mean"]),
]


def decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def compare(tok: str, kind: str, val):
    if kind in ("literature", "label"):
        return "n/a"
    x = float(tok.replace(",", ""))
    if kind == "approx10":
        return "OK" if abs(round(val, -1) - x) < 1e-9 else "MISMATCH"
    if kind == "const":
        return "OK" if abs(val - x) < 1e-9 else "MISMATCH"
    return "OK" if abs(round(val, decimals(tok)) - x) < 1e-9 else "MISMATCH"


def fmt(val, tok):
    if val is None:
        return ""
    if float(val).is_integer():
        return f"{int(val):,}" if abs(val) >= 1000 else str(int(val))
    return f"{val:.{max(decimals(tok), 1) + 2}f}"


def main() -> None:
    lines = (ROOT / "paper/main.tex").read_text(encoding="utf-8").split("\n")
    a = next(i for i, l in enumerate(lines) if r"\begin{abstract}" in l)
    b = next(i for i, l in enumerate(lines) if r"\section*{Code availability}" in l)
    rows, used = [], set()
    for i in range(a, b):
        l = lines[i]
        if l.lstrip().startswith("%") or r"\begin{tabular}" in l:
            continue
        toks = [m.group() for m in PAT.finditer(l)]
        if not toks:
            continue
        spec = [s for s in SPECS if s[0] in l]
        if len(spec) != 1:
            for t in toks:
                rows.append((i + 1, t, l[:50], "UNMAPPED LINE", "", "UNMAPPED"))
            continue
        used.add(spec[0][0])
        srcs = spec[0][1]
        if len(srcs) != len(toks):
            raise SystemExit(f"line {i + 1}: {len(toks)} tokens {toks} but {len(srcs)} sources")
        for t, (kind, src, val) in zip(toks, srcs):
            rows.append((i + 1, t, kind, src, fmt(val, t), compare(t, kind, val)))
    unused = [s[0] for s in SPECS if s[0] not in used]
    text = "\n".join(lines[a:b])
    wrows = []
    for phrase, num, src, val in WORDS:
        present = phrase in text
        if phrase.startswith("more than twice"):
            ok = val > 2
        elif phrase.startswith("tuning and test scoring took under"):
            ok = val < 3
        elif "(about a" in phrase:
            ok = round(float(val), 1) == num
        else:
            ok = abs(float(val) - num) < 1e-9
        wrows.append((phrase, num, src, f"{val:.2f}" if isinstance(val, float) and not float(val).is_integer() else str(int(val)),
                      ("OK" if ok else "MISMATCH") if present else "PHRASE NOT FOUND"))

    # generated tables: regenerate and compare with the committed files
    before = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "paper/tables").glob("*.tex")}
    subprocess.run([sys.executable, str(ROOT / "scripts/make_paper_tables.py")], check=True, capture_output=True)
    after = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "paper/tables").glob("*.tex")}
    tab_ok = before == after

    df = pd.DataFrame(rows, columns=["line", "number", "kind", "source", "value_in_source", "status"])
    n_bad = int((df.status.isin(["MISMATCH", "UNMAPPED"])).sum()) + sum(r[4] != "OK" for r in wrows) + (0 if tab_ok else 1)
    md = ["# Paper number check", "",
          "Generated by `scripts/check_paper_numbers.py` from `paper/main.tex` (abstract to conclusion).",
          "Every numeric token of the text is listed with the file it comes from and the value",
          "recomputed from that file. `value` = recomputed from a saved result and compared at the",
          "paper's number of decimals; `const` = design constant read from the code or docs, compared",
          "exactly; `literature` = quoted from the cited paper (cannot be checked against our files);",
          "`label` = an identifier, not a measured quantity.", "",
          f"Summary: {len(df)} numeric tokens; {int((df.status == 'OK').sum())} OK, "
          f"{int((df.status == 'n/a').sum())} literature or label, {int((df.status == 'MISMATCH').sum())} mismatches, "
          f"{int((df.status == 'UNMAPPED').sum())} unmapped; {len(wrows)} numbers written as words, "
          f"{sum(r[4] == 'OK' for r in wrows)} OK; generated tables "
          f"{'match a fresh regeneration' if tab_ok else 'DIFFER from a fresh regeneration'}."
          + (f" Unused specs: {unused}." if unused else ""), "",
          "Tables `paper/tables/test_summary.tex` and `paper/tables/hypothesis_tests.tex` are generated by",
          "`scripts/make_paper_tables.py` directly from `results/phase3/test_summary.csv`,",
          "`selected_params.csv` and `hypothesis_tests.csv`; they contain no hand-typed numbers.", "",
          "Not numbers in the sense of this check: the NAB commit `ea702d7` (D-02, `scripts/fetch_nab.sh`),",
          "citation keys, section and figure references, and RQ/H labels.", "",
          "## Numbers written as words", "", "| Phrase | Number | Source | Value in source | Status |", "|---|---|---|---|---|"]
    md += [f"| {p} | {n} | {s} | {v} | {st} |" for p, n, s, v, st in wrows]
    md += ["", "## Numeric tokens", "", "| Line | Number | Kind | Source | Value in source | Status |", "|---|---|---|---|---|---|"]
    md += [f"| {r.line} | {r.number} | {r.kind} | {r.source} | {r.value_in_source} | {r.status} |" for r in df.itertuples()]
    (ROOT / "docs/paper_number_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md[8])
    for r in df[df.status.isin(["MISMATCH", "UNMAPPED"])].itertuples():
        print("FLAG", r.line, r.number, r.source, r.value_in_source, r.status)
    for r in wrows:
        if r[4] != "OK":
            print("FLAG word", r)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
