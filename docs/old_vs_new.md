# Old project vs new design

Source of the old project: `C:\Users\numan\OneDrive\Desktop\dstt_project_8015`,
commit `02b007f` (read only). Line numbers refer to files at that commit
(`git show HEAD:<file>`). The working tree also has uncommitted post-hoc
changes (section 3, item 9); the reported results below were produced by the
committed code.

The items marked "unclear" in section 5 were decided on 2026-11-18; see
docs/decisions.md (D-04 to D-12).

## 1. Old protocol as implemented

**Data.** NAB commit `ea702d7`, 58 files, 116 windows. Rows sorted, duplicate
timestamps dropped, missing values forward/back-filled
(`src/atd/data.py:97-102`). Window edges are mapped with `searchsorted`, not
exact timestamp match (`data.py:110-111`). Min-max scaler fitted on the
training segment only (`data.py:153-154`).

**Two protocols** (`PROTOCOL.md:24-31`):

| | Full-corpus (reported as primary) | Split |
|---|---|---|
| Files | all 58 | 18 evaluation + 30 tuning, 41 distinct (`configs/datasets.json`) |
| Train / scored | first 15 % / last 85 % | 60 / 20 / 20 |
| Threshold tuning | leave-one-file-out (LOFO) over the scored segments of the other 57 files | validation segments of the 30 tuning files, objective averaged over files and seeds, one parameter set per method |
| Window size | W* reused from split protocol | W in {30, 50, 100}, W* = best mean validation ROC-AUC; W* = 30 (`results/split/window_selection.csv`) |

**LOFO details** (`src/atd/pipeline.py:203-232`). For every file f and method,
the candidate with the highest mean composite F1 over the other 57 files is
picked (ties: corpus NAB of the other files, then lower alarm rate). The
"other files" objective is computed on their scored 85 % segments, which are
the same segments later reported as test results (`pipeline.py:186-200`,
`pipeline.py:246-253`).

**Training data: normal-only, using training labels.** Training windows that
contain any labelled anomaly point inside the training segment are dropped
(`data.py:172-175`), and the "training errors" used to calibrate B1/B2/B4 and
to seed rolling rules exclude the same windows (`data.py:182-189`). At most
2,500 evenly spaced training windows (`data.py:176-178`). Last 10 % of training
windows is the early-stopping hold-out (`src/atd/model.py:69-71`).

**Model** (PyTorch, `model.py:39-54`): encoder LSTM(64) → Linear(64→32) + tanh
latent → repeated W times → decoder LSTM(32→64) → Linear(64→1). Adam 1e-3;
split: batch 64, ≤40 epochs, patience 5; full: batch 32, ≤60 epochs,
patience 6 (`configs/experiment.json`); best hold-out epoch restored
(`model.py:94-101`). Seeds {15, 42, 123}.

**Anomaly score:** mean *squared* error of the window ending at t, assigned to
t (`model.py:108`, `scripts/train_score.py:78-79`); first W-1 points NaN.

**Metrics** (`src/atd/metrics.py:37-110`): point P/R/F1/FPR/accuracy (no point
adjust), event recall/precision/F1, detection delay, composite F1, FPR on
high-variance normal points (causal 50-step rolling std above the 75th
percentile of the training segment, `data.py:192-199`), calm FPR,
clean-stream FPR, ROC-AUC and PR-AUC (average precision) of the raw score.

**NAB scoring** (`src/atd/nab_score.py`, used in `metrics.py:100-109`):
- The scoring function is a port of `nab/sweeper.py`. I checked it against the
  official `Sweeper` (NAB `ea702d7`, probation 0) on 3,000 random files: max
  absolute difference 4.4e-16. The function itself is correct.
- No probationary period; windows clipped to the scored segment
  (`nab_score.py:17-18`, `data.py:72-78`).
- Reported NAB uses **alert onsets** (first step of each alarm run) instead of
  every alarm step (`metrics.py:100-104`). Every-step scores are kept as
  `*_pointwise`.
- Normalisation `100 (raw - null) / (perfect - null)` with null = -fn·k and
  perfect = tp·k per segment; corpus score sums raw/null/perfect over files
  (same formula as NAB `runner.py normalize`).

## 2. Threshold rules in the old code

All in `src/atd/thresholds.py` (HEAD).

B1, fixed percentile of training errors:
```
27  def fixed_percentile(train_err: np.ndarray, p: float) -> float:
29      return float(np.percentile(train_err, p))
```
B2 and B4, fixed mean + k·std (population std, ddof=0); B4 is the same
function with k = 3:
```
32  def fixed_mean_ksigma(train_err: np.ndarray, k: float) -> float:
34      return float(np.mean(train_err) + k * np.std(train_err))
...
134     if method.startswith("B2") or method.startswith("B4"):
135         th = fixed_mean_ksigma(train_err, params["k"])
```
`train_err` is `normal_train_errors(...)` (`pipeline.py:60`), i.e. the
label-filtered training errors, not all training errors.

B3 and P1 plain memory, vectorised rolling statistics over the w values before
t, seeded with the last w (label-filtered) training errors:
```
57      seed = _seed_window(seed, w)
58      full = np.concatenate([seed, stream])
59      win = sliding_window_view(full, w)[: len(stream)]  # row i = w values before stream[i]
60      if kind == "median_mad":
61          loc = np.median(win, axis=1)
62          scale = np.median(np.abs(win - loc[:, None]), axis=1)
63      elif kind == "mean_std":
64          loc = win.mean(axis=1)
65          scale = win.std(axis=1)
68      return loc, _floor(loc, scale)
```
MAD is the raw MAD (no 1.4826 factor). A scale floor is applied to every
rolling rule:
```
22  EPS_ABS = 1e-12
23  EPS_REL = 1e-3   # scale floor relative to the location (avoids MAD = 0 on flat streams)
38  def _floor(loc, scale):
39      return np.maximum(scale, EPS_ABS + EPS_REL * np.abs(loc))
```
Gated memory (P1 gated, A2, A4):
```
94          if e > th:
95              alarm[t] = 1
96              run += 1
97              if run <= w:
98                  continue          # flagged: memory unchanged
99          else:
100             run = 0
101         store = e                 # normal value, or persistent change accepted as new normal
```
So after more than w consecutive alarms, each further alarmed error enters the
buffer one at a time; the first w alarmed errors of the run are never added.

Base paper labelling: `PROTOCOL.md:51` ("B3 | rolling mean + 3σ ... (base
paper, k = 3 fixed)"), `PROTOCOL.md:52` ("B4 | global mean + 3σ of training
errors (base paper, untuned)"), `configs/experiment.json:255-257`
(`"base_paper_methods": ["B3_rolling_mean_3sigma", "B4_fixed_mean_3sigma"]`).

## 3. Bugs, leakage and weaknesses found

1. **No scaler leakage.** Scaler is fitted on `values[:a]` (`data.py:154`).
2. **Split protocol: no test labels in tuning.** Tuning uses validation
   segments only (`pipeline.py:87-89`); W* uses validation ROC-AUC.
3. **Full-corpus protocol: test segments are also tuning data.** LOFO keeps a
   file's own labels out of its parameters, but the parameters are chosen on
   the other files' scored segments, which are the reported test data. 57 of
   58 test segments inform every selection, and the most frequent LOFO choice
   is identical across nearly all files (e.g. B1 p = 99.5 for 57/58 files,
   `results/full/selected_params.csv`). There is no untouched test data at the
   corpus level. This was the primary protocol.
4. **Training labels used for calibration.** Normal-only training removes
   labelled windows from the training set and from the errors used for B1, B2,
   B4 and the rolling-rule seeds (`data.py:172-175`, `data.py:182-189`). Fixed
   thresholds therefore benefit from labels that a deployed detector does not
   have. "B4 base paper, untuned" is mean + 3σ of label-filtered errors, not of
   all training errors.
5. **Base paper misattributed.** B3 (rolling mean + 3σ) is labelled as the base
   paper method. Per the corrected design, the base paper is the global
   mean + 3σ of training errors.
6. **Model differs from the base paper.** Extra Linear(64→32) + tanh
   bottleneck (`model.py:45-51`), MSE instead of MAE score (`model.py:108`),
   W searched over {30, 50, 100}, training capped at 2,500 windows.
7. **NAB scores are not NAB scores in the usual sense.** The function is
   correct (item in section 1), but alert onsets replace every-step alarms, no
   probationary period, windows clipped at segment edges. The onset choice
   changes results heavily: corpus standard NAB for P1 (full protocol) is
   30.287 with onsets and -1317.161 every-step
   (`results/full/corpus_nab.csv`). Also, the full protocol used the first
   15 % as "probation" while NAB uses min(floor(0.15 n), 750) rows, so for
   files longer than 5,000 rows the old scored segment drops windows NAB would
   score; 107 of 116 windows were scored (`results/full/test_metrics.csv`,
   `n_events` summed per seed).
8. **P1 never used gated memory.** In both protocols plain memory was always
   selected, so P1 results equal A1 exactly (`results/*/summary_by_method.csv`).
   In the split protocol B3 also equals A3 (A3 picked k = 3).
9. **Post-hoc changes are test-informed.** The uncommitted working tree adds a
   MAD floor, log transform, alarm-run merging, longer windows and a robust
   scaler (`scripts/posthoc.py`, `configs/posthoc/`), designed after the test
   results were known. They cannot be reported as confirmatory.
10. Minor: duplicate key in `DOMAINS` (`data.py:31`, harmless);
    `normal_train_errors` silently falls back to all errors when fewer than 32
    clean ones remain (`data.py:189`); rolling seeds are built from
    non-contiguous errors (filtered windows removed); `_seed_window` returns
    zeros for an empty seed (`thresholds.py:45-46`); precision is set to 0 when
    a method never alarms, so composite F1 = 0 for silent detectors
    (`metrics.py:52`).

## 4. Old headline results (copied from files)

Full-corpus protocol, per-file means over 52 files with scored windows
(`results/full/summary_by_method.csv`) and corpus NAB standard, onsets
(`results/full/corpus_nab.csv`), W* = 30:

| Method | Composite F1 | Point F1 | Event recall | FPR | HV FPR | Corpus NAB std | NAB std every-step |
|---|---|---|---|---|---|---|---|
| B1 fixed percentile | 0.4730 | 0.2259 | 0.7174 | 0.0781 | 0.1723 | 38.977 | -530.375 |
| B2 fixed mean + kσ | 0.4676 | 0.2223 | 0.6923 | 0.0692 | 0.1343 | 43.528 | -525.008 |
| B3 rolling mean + 3σ | 0.4821 | 0.2066 | 0.8266 | 0.0311 | 0.1412 | 36.555 | -431.773 |
| B4 global mean + 3σ | 0.4576 | 0.2366 | 0.7826 | 0.0872 | 0.1997 | 32.668 | -722.091 |
| P1 median + λ·MAD | 0.4327 | 0.2214 | 0.8147 | 0.0985 | 0.2553 | 30.287 | -1317.161 |
| A3 mean + kσ plain | 0.5144 | 0.1519 | 0.7519 | 0.0146 | 0.0707 | 47.109 | -193.941 |

ROC-AUC 0.6029, PR-AUC 0.285 (same for all methods).

Split protocol, per-file means over 18 evaluation files
(`results/split/summary_by_method.csv`, `results/split/corpus_nab.csv`):

| Method | Composite F1 | Point F1 | Event recall | FPR | HV FPR | Corpus NAB std | NAB std every-step |
|---|---|---|---|---|---|---|---|
| B1 | 0.7067 | 0.3698 | 0.9722 | 0.1960 | 0.2380 | 63.997 | -455.585 |
| B2 | 0.6989 | 0.3926 | 0.9722 | 0.2069 | 0.2720 | 60.709 | -480.481 |
| B3 | 0.7407 | 0.2550 | 0.9105 | 0.0240 | 0.0899 | 66.625 | -72.582 |
| B4 | 0.7435 | 0.3727 | 0.9722 | 0.1943 | 0.2449 | 73.271 | -399.290 |
| P1 | 0.5470 | 0.1794 | 0.7932 | 0.0372 | 0.0901 | 54.663 | -128.278 |

ROC-AUC 0.6156, PR-AUC 0.5095. H1 "Not supported" in both protocols
(`results/*/hypotheses.json`); e.g. split, P1 vs B1 composite F1 mean
difference -0.1597, Holm p = 0.157, 18 files.

## 5. Old vs new design

| Aspect | Old | New | Type |
|---|---|---|---|
| Primary protocol | Full corpus, 15/85, LOFO on scored segments | Chronological 60/20/20 per file, validation-only tuning, test scored once | bug fix (removes test-data tuning, item 3) |
| Base paper method | B3 rolling mean + 3σ and B4 | "Base (Ahmad 2026)": B2 with k = 3 fixed, untuned | bug fix (attribution, item 5) |
| B3 role | Base paper | Non-robust rolling counterpart of P1 | bug fix (item 5) |
| Training data / calibration errors | Normal-only, training labels used | Not yet specified | unclear: decide before training (item 4) |
| Framework | PyTorch | PyTorch | same |
| Architecture | LSTM(64) → Linear(32)+tanh → repeat → LSTM(64) → Linear | LSTM(64) → repeat 30 → LSTM(64) → linear per step | design choice (follow base paper) |
| Window | W ∈ {30, 50, 100}, W* by validation ROC-AUC | T = 30 fixed | design choice |
| Score | Window MSE at window end | Window MAE; mapping to time point proposed: window end | design choice |
| Training budget | ≤ 2,500 windows; ≤ 40/60 epochs, patience 5/6; batch 64/32 | 50 epochs + early stopping; batch and window cap unspecified | unclear |
| Early-stopping data | Last 10 % of training windows | Not specified | unclear |
| Tuning scope (split) | One parameter set per method, pooled over tuning files | "chosen on validation only"; pooled vs per file not specified | unclear |
| MAD | Raw MAD | Not yet fixed (raw or ×1.4826; only rescales the λ grid) | design choice |
| Scale floor | 1e-3·|loc| floor on all rolling rules | Not yet fixed | unclear |
| Gated memory after > w alarms | Later alarmed errors enter one by one | "accepted as the new normal"; exact mechanics not specified | unclear |
| Rolling seed | Last w label-filtered training errors | Continuous error stream before t | design choice |
| NAB scorer | Correct sweeper port, alert onsets, no probation, windows clipped, not validated against final_results.json | Official scorer, every alarm step, validated against final_results.json | bug fix (scoring convention) |
| NAB on test segment | Clipped windows, no probation | Not yet decided | unclear |
| Window-edge mapping | searchsorted | Exact timestamp match (as NAB) | design choice |
| Ablations A1-A4, B4 | Present | Not in design | design choice |
| Seeds | {15, 42, 123} | 3 seeds, values not set | unclear |
| Statistics | Wilcoxon + Holm, bootstrap CI | Not specified | unclear |
| Post-hoc variants | Uncommitted, test-informed | Not carried over | bug fix (item 9) |
