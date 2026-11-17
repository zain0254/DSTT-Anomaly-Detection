# Design decisions

Each entry: what was decided, by whom, and why. "User" means the project
owner decided; "Implementation" means a detail fixed while writing code that
the owner can overrule.

## D-01 Repository location and license (2026-11-18, user)
Both repos are cloned side by side in `C:\Users\numan\dstt\`, outside
OneDrive. OneDrive locks files during git/pip operations and would sync the
venv. License: MIT for both repos.

## D-02 NAB version (2026-11-18, implementation)
numenta/NAB pinned at `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140` (master on
2026-11-18). `scripts/fetch_nab.sh` fetches `data/`, `labels/`, `config/`,
`nab/` and the results of four detectors (numenta, windowedGaussian,
relativeEntropy, null) used only to validate the scorer port. NAB is MIT
licensed at this commit, so the ported scorer can live in this repo with
attribution.

## D-03 Base paper method (2026-11-18, user correction)
The base paper (Ahmad 2026) thresholds at mean(RE_train) + 3 * std(RE_train).
It is labelled "Base (Ahmad 2026)" and k is fixed at 3, never tuned. The
rolling mean + 3 sigma rule (B3) is not the base paper; it is the non-robust
rolling counterpart of P1. The old project labelled B3 as the base paper
(docs/old_vs_new.md, section 3 item 5).

## D-04 Training data (2026-11-18, user)
The LSTM-AE is trained on all training-split data and RE_train is computed on
all training-split points, with no label filtering. Labels are used only for
tuning (validation segment) and scoring (test segment). Reason: a deployed
detector has no labels for its training data, and the base paper's rule is
defined on RE_train. The old project removed labelled windows from training
and calibration (old_vs_new.md, item 4).

## D-05 Early stopping (2026-11-18, user)
Early stopping monitors the reconstruction loss on the last 10 % of the
training split (chronologically last training windows). The validation
segment is used only for threshold tuning.

## D-06 Tuning and evaluation (2026-11-18, user)
One parameter set per method, chosen by pooling over the tuning files'
validation segments. The chosen set is applied once to the evaluation files'
test segments. Files for tuning and evaluation are proposed after EDA and
confirmed by the owner.

## D-07 NAB scoring on the test segment (2026-11-18, user)
Primary: official NAB scoring of every alarm step, windows clipped to the test
segment, no probationary period, standard / reward_low_FP / reward_low_FN
profiles. Secondary (labelled as such): only the first step of each alarm run
is reported to the scorer. Neither number is comparable to the NAB
leaderboard, which optimises one threshold per detector on all labels of the
full files and uses a 15 % probationary period.
The scorer port itself is validated on full files with NAB's default
probationary period against `results/final_results.json`
(`results/scorer_validation.csv`).

## D-08 Gated memory (2026-11-18, user)
P1 gated: an error that raises an alarm does not enter the rolling window.
When an alarm run exceeds w steps (run length w + 1), the window is replaced
by the run's last w errors and the run counter is reset; the new level is
accepted as normal at once. The alarm decision at that step stands (the
replacement happens after the decision). Errors of runs that end before
exceeding w never enter the window.

## D-09 Robust scale and floor (2026-11-18, user)
MAD is scaled by 1.4826 (consistent estimator of sigma under normality), for
P1 and S-robust. B3 and P1 use the same fixed scale floor:
floor = 0.1 * 1.4826 * MAD(RE_train of that file), not tuned. The rolling
scale used is max(rolling scale, floor).
Implementation note: if a file's training errors have MAD = 0, the floor is 0
and a constant rolling window would alarm on any increase. EDA will report
whether any selected file is affected; no epsilon is added unless the owner
decides so.

Amendment (2026-11-18, user, decided before any model results): if a
file's training errors have scaled MAD = 0, floor = 0.1 * std(RE_train); if
that is also 0, floor = 1e-6. Same rule for B3 and P1. This replaces the
earlier note that a zero MAD gives a zero floor. Implemented in
`pipeline/thresholds.py: scale_floor`, tested in `tests/test_thresholds.py`.

## D-10 Methods (2026-11-18, user)
| ID | Rule | Tuned on validation |
|---|---|---|
| Base (Ahmad 2026) | mean + 3 std of RE_train | nothing |
| B1 | p-th percentile of RE_train | p |
| B2 | mean + k std of RE_train | k |
| S-robust | median + lambda * scaled MAD of RE_train | lambda |
| B3 / RePAD2 rule (Lee & Lin 2023) | rolling mean + 3 std of the previous w errors, floored (D-09) | w (k = 3 fixed, implementation reading of "rolling mean + 3 sigma") |
| R2 EVT (Yu et al. 2024, Alg. 2) | GPD tail threshold from RE_train, q by Eq. (8) on RE_train (D-24) | nothing on labels |
| B3-k | rolling mean + k std of the previous w errors, floored (D-09) | k, w (D-26) |
| P1 | rolling median + lambda * scaled MAD of the previous w errors, floored, memory plain or gated | lambda, w, memory |
The old ablations A1-A4 and the old B4 are dropped.

## D-11 Model (2026-11-18, user)
Follows the base paper: LSTM(64) encoder, last hidden state repeated 30
times, LSTM(64) decoder, per-step linear output, window T = 30 fixed (no
search), MinMax scaling fitted on the training split only, Adam, MSE loss,
up to 50 epochs with early stopping (D-05). Anomaly score = mean absolute
error of the window. The old project's extra Linear(64->32) + tanh bottleneck
and MSE score are not used.
Framework: the base paper uses Keras/TensorFlow. We use PyTorch because
TensorFlow has no native Windows GPU support after 2.10 (user decision,
2026-11-18). The architecture is unchanged.

## D-12 PyTorch build (2026-11-18, user)
CPU build of PyTorch (`torch==2.14.1` from PyPI, which is CPU-only on
Windows). The model has about 50k parameters, so the GPU speed-up is small;
CPU runs are deterministic for a fixed seed; the download is about 200 MB
instead of about 2.5 GB. One file / one seed is timed before any training run
and a full estimate is given to the owner first.

## D-13 Score-to-time mapping (2026-11-18, implementation)
The MAE of the window x[t-29 .. t] is the score of time t (window end), so
scores are causal and exist for t >= 29. The first 29 points of each file
have no score and never alarm.

## D-14 Rolling rules over a continuous stream (2026-11-18, implementation)
Rolling rules run over the whole error stream of a file from its first score
onward. The threshold at t uses the w errors before t (train errors seed the
validation segment, validation errors seed the test segment). The first w
scored points have no threshold and never alarm. Gated memory state is carried
through the train and validation segments in the same way. Standard
deviations use ddof = 0.

## D-15 Alarm condition (2026-11-18, implementation)
All rules alarm when e(t) > threshold(t) (strict).

## D-16 Seeds (2026-11-18, implementation)
Seeds {15, 42, 123}, the same values as the old project.

## D-17 Metric conventions (2026-11-18, implementation)
- Point labels: every point inside a NAB window (clipped to the segment) is
  positive. No point adjustment.
- Precision of a detector with no alarms is 0 (so its F1 and composite F1 are
  0, not undefined).
- Event recall: share of windows in the segment with at least one alarm.
- Composite F1: harmonic mean of point precision and event recall
  (Garg et al. 2022).
- FPR = FP / (FP + TN) over normal points.
- High-variance FPR: FPR over normal points whose causal 50-step rolling std
  of the raw series is above the 75th percentile of the rolling std over the
  file's training split (same definition as the old project).
- ROC-AUC (`roc_auc_score`) and PR-AUC as average precision
  (`average_precision_score`) on raw errors; NaN if the segment has no
  positive or no negative point. These do not depend on the threshold rule.

## D-18 Window edges (2026-11-18, implementation)
NAB window edges are mapped to rows by exact timestamp match, as NAB's
sweeper requires; a missing edge is an error.

## D-19 Duplicate timestamps (2026-11-18, implementation)
NAB files contain 49 duplicate timestamps in 8 files (docs/phase1_eda.md).
Rows are kept as they are, because NAB scores every row. A window edge that
falls on a duplicated timestamp maps to its first occurrence, as
`list.index` does in NAB's sweeper (`pipeline/data.window_indices`).

## D-20 No resampling (2026-11-18, implementation)
25 files have irregular sampling (27,625 missing steps in total, mostly
realTraffic and rogue_agent_key_hold). Series are not resampled or
interpolated: NAB labels and scores rows, and filling gaps would add points
that NAB does not score. The LSTM window of 30 rows therefore covers a
variable time span in these files.

## D-21 File selection (2026-11-18, user, confirmed after EDA)
Generated by `scripts/make_selection.py` into `configs/datasets.json` from
`results/phase1_eda/file_summary.csv`:
- Evaluation: the 18 files with at least one window touching the test part
  (25 test windows), scored once on their test segments.
- Tuning: the 24 real-category files with at least one window touching the
  validation part, excluding grok_asg_anomaly (its window 2 spans val and
  test) and all artificial files; 25 validation windows; used only through
  their validation segments, pooled (D-06).
- 6 files are in both sets; their validation and test segments are disjoint.
- Model files: the union, 36 files.
Reasons are in docs/phase1_eda.md ("Proposed file selection").

## D-22 Clean-stream false-alarm rate (2026-11-18, user, declared before any results)
Secondary metric: FPR and false alarms per 1,000 points on the anomaly-free
test segments of the 18 tuning-only files (`clean_test` in
configs/datasets.json). Every alarm there is false. Not used for tuning.
Implemented in `pipeline/metrics.clean_stream_metrics`.

## D-23 Training details not fixed by the base paper (2026-11-18, implementation)
Batch size 32 (Keras default), Adam learning rate 1e-3 (Keras default),
early-stopping patience 5 epochs with the best epoch restored, shuffled
mini-batches per epoch with a seeded generator,
`torch.use_deterministic_algorithms(True)` and 4 CPU threads per fit. The
hold-out is the last 10 % of training windows by position (as Keras
`validation_split` does); the first hold-out windows share up to 29 rows with
the last fitting windows. Owner can overrule any of these before the full
run.

## D-24 Reference threshold rules (2026-11-18, user, decided before any threshold results)
Added while the LSTM-AE training run was in progress; no threshold has been
computed or scored yet.
- R1 RePAD2 rule (Lee & Lin, IoTBDS 2023) is the same rule as B3 (rolling
  mean + 3 std of the previous w errors). No duplicate method; B3 is labelled
  "B3 / RePAD2 rule (Lee & Lin 2023)". Differences from the original: their
  window lengths (1440 to 16128) exceed the length of many NAB files, so w is
  tuned on validation like the other rolling rules; they threshold the AARE
  of an LSTM predictor, we apply the rule to LSTM-AE reconstruction MAE. The
  D-09 floor applies as for B3.
- R2 "EVT (Yu et al. 2024, Alg. 2)", IEEE I2MTC 2024,
  doi:10.1109/I2MTC60896.2024.10561049 (peer reviewed; replaces an arXiv
  preprint considered earlier the same day, before any threshold results).
  Static threshold from RE_train, not tuned on labels:
  th = 0.98 quantile of RE_train; excesses y = e - th for e > th, N_TH = |y|,
  n = |RE_train|; GPD fitted to y by maximum likelihood
  (`scipy.stats.genpareto.fit(y, floc=0)`) giving shape gamma and scale sigma;
  E_TH(q) = th + (sigma / gamma) * ((q n / N_TH)^(-gamma) - 1), or
  th - sigma * log(q n / N_TH) if |gamma| < 1e-6. q is chosen on a 50-point
  log grid in [1e-5, 1e-3] by maximising their Eq. (8) on training errors only:
  (dmu / mu + dvar / var) / |e_a|, e_a = {e > E_TH(q)}, mu and var the mean
  and variance of RE_train, dmu and dvar their drop when e_a is removed. A q
  with |e_a| = 0 is skipped; if every q is skipped, the q with the lowest
  E_TH is used and logged. Fallback if N_TH < 10 or the fit fails:
  E_TH = max(RE_train), logged per file.
  Property of the grid: the expected number of training errors above E_TH(q)
  is about q n, so for these files (n roughly 650 to 13,600 training
  windows) many grid values have no exceedance and are skipped.
  Not implemented: their dynamic update (Eq. 9). As printed,
  eps = 2 - exp(-(...)^2) lies in [1, 2), so the update can only raise the
  threshold, and the update frequency is not given; only the fully specified
  static part is reproduced. They score windows by squared L2 error; we apply
  the rule to per-window MAE (D-11).
  Implemented in `pipeline/thresholds.evt_threshold`; unit tests: closed-form
  E_TH for both gamma branches, Eq. (8) on a hand-made array, exponential
  errors (E_TH above the 0.98 quantile, q inside the grid), two fallback
  paths.

## D-25 Short early-stopped fits are kept (2026-11-18, user)
5 of 108 fits stopped with best epoch <= 2 (docs/phase2_model.md). They are
not retrained: the decided protocol (D-05, D-23) is applied to every fit
without exceptions, so no fit is treated differently after its outcome is
seen.

## D-26 Phase 3 protocol and pre-registered tests (2026-11-18, user, before any threshold result)
Tuning
- Objective: mean composite F1 over the 24 tuning files x 3 seeds, computed
  on validation segments (windows clipped to the segment). Ties: higher mean
  every-step NAB standard (per file and seed, normalised on the segment,
  then averaged), then lower mean alarm rate, then the earlier grid entry.
- One parameter set per method, shared by all seeds.
- Grids: B1 p in {90, 95, 97, 98, 99, 99.5, 99.9}; B2 k in {1, 1.5, 2, 2.5,
  3, 4, 5, 6, 8}; S-robust and P1 lambda in {2, 3, 4, 5, 6, 8, 10, 12, 15,
  20}; B3, B3-k and P1 w in {50, 100, 200, 400}; P1 memory in {plain,
  gated}. Base and R2 have no parameters.
- Added method B3-k: rolling mean + k std of the previous w errors with the
  same floor as B3, k in {2, 3, 4, 5, 6, 8, 10, 12, 15, 20}, w as above.
  B3 (k = 3 fixed, RePAD2 rule) is kept. B3-k vs P1 and B2 vs S-robust are the
  equal-budget Gaussian-vs-robust comparisons (same parameters tuned, same
  grid for the multiplier).
Test
- The selected parameters are applied once to the test segments of the 18
  evaluation files x 3 seeds. Reported per method as mean +/- sd over seeds
  of the across-file mean (and corpus NAB per seed). Clean-stream rate (D-22)
  on the test segments of the 18 clean files.
- Saved per file and seed: chosen thresholds and alarm arrays on the test
  segments, the full validation tuning table, and the R2 path per file and
  seed (Eq. 8, fallback, or "no q with exceedances").
Pre-registered hypothesis tests (run after the test summary is reviewed)
- H1: composite F1. H2: high-variance FPR; secondary H2 measures: clean-stream
  FPR and false alarms per 1,000 points. H3: every-step NAB standard.
- Primary comparison: P1 vs Base (Ahmad 2026). Secondary: P1 vs every other
  method, Holm-corrected within each hypothesis.
- Unit of analysis: file (metric averaged over the 3 seeds), n = 18
  (clean-stream measures: n = 18 clean files). Paired two-sided Wilcoxon
  signed-rank test, mean difference with bootstrap 95 % CI, wins/ties/losses.

## D-27 Exploratory grid extension (2026-11-18, user, POST-HOC)
Decided after the primary test results and the pre-registered tests were
known, because four selections sat on grid edges (B1 p = 90, B3 and B3-k
w = 400, P1 w = 50). Exploratory only: it does not replace or modify the
primary results (results/phase3/) or the D-26 verdicts.
- Only the edge grids are extended: P1 w adds {10, 20, 30}; B3 and B3-k w add
  {600, 800}; B1 p adds {80, 85}. All other grids, the objective and the
  tie-breaks are as in D-26.
- A rolling window w is used for a file only if the file has at least w
  errors before its validation segment (train_end - 29 >= w). Otherwise that
  candidate is skipped for that file and logged; its tuning mean is taken
  over the remaining series. At test, if the selected w is not allowed for a
  file, the largest allowed w of that method's grid is used and logged.
- Re-selected on validation, test scored once, outputs in
  results/phase3_exploratory/; reported in a separate, labelled section of
  docs/phase3_thresholds.md.

## D-28 Proposal hypotheses (2026-11-18, user; pre-registered in the October 2026 proposal defense)
Tested exactly as written in the proposal, in addition to D-26, on the
primary Phase 3 results (no primary result is changed or rerun). alpha = 0.05,
as stated in the proposal (p < 0.05). Unit = file, metric averaged over the
3 seeds. Each hypothesis is tested against Base (static mu + 3 sigma) and
against B3 (rolling mu + 3 sigma, which the proposal labelled as the base
paper).
- Proposal H1: P1 has a higher point F1 than the baseline on at least 70 %
  of files (proposal: 7 of 10; here at least 13 of 18), plus a two-sided
  Wilcoxon signed-rank test. Implementation reading: supported only if both
  hold (at least 13 wins and p < 0.05 with P1 higher).
- Proposal H2: P1 reduces the mean high-variance FPR by at least 15 %
  relative to the baseline ((mean_other - mean_P1) / mean_other, n = 17),
  tested with a paired t-test (as proposed) and a Wilcoxon signed-rank test,
  both two-sided and reported side by side. The verdict is given per test
  (criterion met and p < 0.05 with P1 lower); neither test is preferred.
- Proposal H3 (RQ4): per NAB category of the 18 evaluation files, mean
  improvement of P1 over Base and over B3 in point F1, composite F1 and
  high-variance FPR (improvement = P1 - other for F1 measures, other - P1 for
  FPR). H3 = realAWSCloudwatch shows the largest improvement. Descriptive
  only, with n files per category; no significance claim.
- Detection delay (proposal RO6): for each detected test window (windows
  clipped to the test segment, as in all test metrics), (first alarm index -
  window start) / window length; mean per method over detected windows,
  pooled over files, reported as mean +/- sd over seeds. Missed windows are
  counted separately.
Label correction: D-26 called the NAB test "H3". In the paper NAB belongs to
RQ2/RQ3 and H3 is the proposal's category hypothesis above. The D-26 text is
kept as registered; the doc labels that test "D-26 NAB test (labelled H3 in
D-26)".
