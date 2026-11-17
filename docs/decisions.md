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
| B3 | rolling mean + 3 std of the previous w errors, floored (D-09) | w (k = 3 fixed, implementation reading of "rolling mean + 3 sigma") |
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
  (Garg et al. 2021).
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
