# Threshold rules for LSTM-autoencoder anomaly detection on NAB

One LSTM autoencoder (base paper architecture, PyTorch) is trained per series;
the compared methods are threshold rules applied to its reconstruction errors
e(t):

- Base (Ahmad 2026): mean + 3 std of training errors, untuned
- B1: p-th percentile of training errors
- B2: mean + k std of training errors, k tuned
- S-robust: median + lambda * scaled MAD of training errors
- B3 / RePAD2 rule (Lee & Lin 2023): rolling mean + 3 std of the previous w errors
  (non-robust counterpart of P1)
- R2 EVT (Yu et al. 2024, Alg. 2): GPD tail threshold of training errors, untuned
- P1: rolling median + lambda * scaled MAD of the previous w errors (proposed),
  plain or gated memory

Data: Numenta Anomaly Benchmark (NAB) at commit ea702d75, chronological
60/20/20 split per file, all parameters tuned on validation only, test scored
once, 3 seeds. Design decisions: docs/decisions.md. Change history:
docs/edit_log.md. Comparison with the earlier project: docs/old_vs_new.md.

## Setup (Windows, Git Bash)

    bash scripts/fetch_nab.sh          # NAB data at the pinned commit -> data/nab
    bash scripts/setup_env.sh          # venv in envs/pipeline
    envs/pipeline/Scripts/python.exe -m pytest
    envs/pipeline/Scripts/python.exe scripts/validate_scorer.py

## Layout

    pipeline/    shared package: data, splits, model, threshold rules, metrics, plots
    references/  external reference methods, one folder and venv each
    scripts/     data fetch, environment setup, validation, figure sync
    results/     CSV/JSON outputs
    figures/     figures by phase (PNG 300 dpi + PDF)
    docs/        decisions, edit log, hardware, environment

Figures are mirrored to the DSTT-Anomaly-Detection-images repository.
