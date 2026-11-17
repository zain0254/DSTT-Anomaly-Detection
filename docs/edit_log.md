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
- D-24 (owner, before any threshold results): B3 relabelled
  "B3 / RePAD2 rule (Lee & Lin 2023)" in docs, README and
  pipeline/thresholds.py; R2 EVT-POT added (pipeline/thresholds.py:
  evt_pot_threshold, gpd_tail_threshold) with 4 unit tests. Done while the
  training run was in progress; the run was not restarted.
- R2 replaced (owner, still before any threshold results): EVT-POT from the
  Alcarria et al. arXiv preprint dropped; R2 is now "EVT (Yu et al. 2024,
  Alg. 2)", static part only (pipeline/thresholds.py: evt_threshold,
  evt_quantile, eq8_score; 5 unit tests replace the 4 EVT-POT tests). D-24
  rewritten; docs/references.md added. Reason: owner prefers a peer-reviewed
  source. The training run was not interrupted.
- Training run (scripts/train.py): 108/108 fits, 0 failures, 47.5 min summed
  wall time; outputs in results/phase2_errors/ (errors, loss history, run
  time per fit), log in logs/phase2_train.log. 5 fits with best epoch <= 2
  kept as trained (D-25). docs/phase2_model.md.
- scripts/plot_phase2.py: loss curves and one reconstruction-error figure
  (figures/phase2_model/).
- docs/references.md: full citations for Ahmad (2026), Lee & Lin (2023),
  Garg et al. (2022); NAB cited as the GitHub repository at the pinned
  commit instead of Lavin & Ahmad (2015). Reason: the paper cites 2021+
  sources only. decisions.md D-17 now cites Garg et al. 2022.
- D-26 recorded before any threshold result: tuning objective, grids, B3-k
  added, one parameter set across seeds, pre-registered H1-H3 tests. Reason:
  owner approved the Phase 3 plan.
- Phase 3 on branch feature/phase3-thresholds: pipeline/methods.py (method
  registry, shared rolling statistics), gated memory uses a numpy ring
  buffer (same behaviour, faster; tests unchanged plus a consistency test),
  scripts/phase3.py (tuning on validation, one test scoring), results in
  results/phase3/, scripts/plot_phase3.py (tuning figure),
  docs/phase3_thresholds.md. Hypothesis tests not run yet.
- Pre-registered tests run (scripts/hypothesis_tests.py,
  results/phase3/hypothesis_tests.csv) and added to docs/phase3_thresholds.md
  with verdicts: H1, H2, H3 not supported (primary P1 vs Base). alpha = 0.05
  stated as the conventional level because D-26 does not give one.
- D-27 exploratory run (post-hoc): pipeline/methods.py gains the
  EXPLORATORY grid set and w_allowed; scripts/phase3.py gains --grids
  (primary unchanged, exploratory writes results/phase3_exploratory/) and
  logs skipped windows (window_log.csv). Only B3/B3-k changed selection
  (w = 600). Reported in a separate section of docs/phase3_thresholds.md.
