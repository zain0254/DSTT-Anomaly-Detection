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
