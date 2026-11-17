# Edit log

## 2026-11-18
- Initial commit in both repos: README stub, .gitignore, MIT license
  (README.md, .gitignore, LICENSE). License confirmed by owner.
- Compared the old project with the new design (docs/old_vs_new.md). Reason:
  owner asked for an audit before building the new pipeline.
- Design correction from owner: base paper method is B2 with k = 3 fixed,
  labelled "Base (Ahmad 2026)"; B3 is the non-robust rolling counterpart of
  P1; model follows the base paper (LSTM(64), repeat 30, LSTM(64), per-step
  linear, T = 30, MinMax on train, Adam, MSE, 50 epochs with early stopping,
  per-window MAE score). Recorded in docs/decisions.md D-03, D-11. Reason:
  the old project misattributed the base paper rule and changed the model.
- Framework changed from Keras/TensorFlow to PyTorch (D-11). Reason:
  TensorFlow has no native Windows GPU support after 2.10. Then CPU build of
  PyTorch chosen (D-12): small model, deterministic CPU runs, smaller
  download. requirements.txt: tensorflow replaced by torch==2.14.1.
- Owner decisions for the open items of old_vs_new.md: all training data
  without label filtering (D-04), early stopping on last 10 % of the training
  split (D-05), pooled tuning on validation (D-06), official every-step NAB
  primary and first-step NAB secondary (D-07), gated memory replaces the
  window after a run longer than w (D-08), scaled MAD and a 0.1 * training
  scaled MAD floor for B3 and P1 (D-09), method set (D-10).
- Implementation details recorded as D-13 to D-18 (score at window end,
  continuous rolling stream, strict alarm condition, seeds, metric
  conventions, exact window-edge matching).
- Phase 0 setup on branch feature/phase0-setup:
  - scripts/fetch_nab.sh: sparse, blob-filtered checkout of NAB at
    ea702d75 into data/nab (data, labels, config, nab, results of numenta,
    windowedGaussian, relativeEntropy, null). Reason: pinned data, small
    download.
  - requirements.txt, scripts/setup_env.sh, docs/env_pipeline.txt: Python
    3.12 venv in envs/pipeline with exact pins; pip freeze saved.
  - scripts/hardware_report.ps1, docs/hardware.md: hardware report.
  - pipeline/data.py: NAB loader, exact window-edge matching (D-18),
    60/20/20 split, window clipping.
  - pipeline/nab_scorer.py, scripts/validate_scorer.py: port of NAB's
    sweeper and normalisation. Validation on 3 detectors x 3 profiles matches
    NAB per file (max abs diff 5.0e-12) and final_results.json (max abs diff
    1.9e-11); results/scorer_validation.csv, results/scorer_validation_files.csv.
  - pipeline/metrics.py: point P/R/F1, event recall, composite F1, FPR,
    high-variance FPR, ROC-AUC, PR-AUC, NAB every-step (primary) and
    first-step (secondary) (D-07, D-17).
  - pipeline/thresholds.py: Base, B1, B2, S-robust, B3, P1 plain/gated
    (D-08 to D-10, D-14, D-15).
  - tests/: 32 unit tests on hand-made arrays, all passing.
  - gh CLI not installed; owner said it is not needed. Pushing uses Git
    Credential Manager.
- D-09 amended (owner): zero-MAD floor falls back to 0.1 * std(RE_train),
  then 1e-6 (pipeline/thresholds.py, tests/test_thresholds.py, 2 new tests,
  34 passing). Reason: a zero floor would make rolling rules alarm on any
  increase over a constant window; decided before any model results.
- Phase 1a EDA on branch feature/phase1-eda:
  - pipeline/data.py: duplicate timestamps at window edges map to their first
    occurrence, matching NAB (D-19); new unit test (35 passing); scorer
    validation re-run, unchanged.
  - pipeline/plotting.py: shared figure style (validated categorical palette,
    PNG 300 dpi + PDF).
  - scripts/eda.py: per-file summary, window positions per split part,
    training contamination, six figures (figures/phase1_eda/), CSVs in
    results/phase1_eda/.
  - scripts/sync_figures.sh: mirrors figures/ into the images repo, commits
    and pushes.
  - docs/phase1_eda.md: figure descriptions from saved numbers and a proposed
    (not final) evaluation/tuning file selection. D-20 (no resampling)
    recorded. Reason: owner asked for EDA before training and for window
    counts per part and training contamination to justify the selection.
- Selection confirmed by owner (D-21): scripts/make_selection.py writes
  configs/datasets.json (18 evaluation, 24 tuning, 18 clean-test, 36 model
  files).
- Clean-stream false-alarm rate added as a secondary metric before any
  results (D-22): pipeline/metrics.clean_stream_metrics and a unit test.
- pipeline/model.py: base-paper LSTM-AE in PyTorch (50,497 parameters),
  MinMax on train, early stopping on last 10 % of training windows,
  per-window MAE at window end (D-11 to D-13, D-23); tests/test_model.py
  (shape, alignment, seed determinism). 39 tests passing.
- scripts/time_one.py: timed one fit (nyc_taxi, seed 15, 4 threads): 44
  epochs, 38.6 s training; projection for 36 files x 3 seeds 0.77 h
  (same epochs) to 0.88 h (50 epochs) sequential; results/phase2_timing.json.
