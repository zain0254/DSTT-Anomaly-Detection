# Phase 3: threshold tuning and test scoring

Produced by `scripts/phase3.py` (tuning 135 s, test 6 s, 4 worker processes)
under the protocol pre-registered in docs/decisions.md D-26. Numbers below are
copied from `results/phase3/*.csv`. The hypothesis tests (D-26) have not been
run yet.

## Tuning (validation, 24 tuning files x 3 seeds)

`tuning_candidates.csv` holds all 10,944 rows (152 candidates x 72 series);
`tuning_summary.csv` averages them; `selected_params.csv` holds the choice per
method. Figure: `figures/phase3_thresholds/fig01_tuning_objective`.

Selections on the edge of their grid: B1 p = 90 (lowest p), B3 and B3-k
w = 400 (largest w), P1 w = 50 (smallest w). B3-k selected k = 3, so it is
identical to B3 on the test set. P1 selected plain memory.

R2 (Yu et al. 2024) path over all 108 model series (`r2_info.csv`):
104 Eq. (8), 4 "no q with exceedances" (lowest E_TH used:
Twitter_volume_AMZN seeds 15, 42, 123 and speed_t4013 seed 123),
0 fallback.

## Test (18 evaluation files x 3 seeds; clean-stream: 18 clean files x 3 seeds)

Each cell: mean ± sd over the 3 seeds of the across-file mean (corpus NAB:
raw, null and perfect summed over the 18 files per seed, then normalised).
High-variance FPR is undefined for 1 of the 18 evaluation files,
ec2_cpu_utilization_ac20cd (no normal test point above the training variance
cut, D-17, for any seed); its mean covers the other 17 files. ROC-AUC 0.633 ± 0.003 and PR-AUC 0.497 ± 0.011 are the same
for every method (threshold-free).

| Method | Selected | Composite F1 | Point F1 | Precision | Event recall | FPR | HV FPR | Corpus NAB std (every step) | Corpus NAB std (first step) | Clean FPR | Clean FA / 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base (Ahmad 2026) | `{}` | 0.733 ± 0.011 | 0.355 ± 0.013 | 0.645 ± 0.016 | 0.972 ± 0.000 | 0.189 ± 0.000 | 0.231 ± 0.001 | -370.2 ± 2.6 | 72.0 ± 2.5 | 0.061 ± 0.002 | 61.5 ± 1.9 |
| B1 percentile | `{"p": 90}` | 0.587 ± 0.009 | 0.407 ± 0.013 | 0.454 ± 0.008 | 0.972 ± 0.000 | 0.319 ± 0.006 | 0.488 ± 0.008 | -1026.9 ± 54.8 | 13.3 ± 6.8 | 0.187 ± 0.001 | 187.3 ± 0.7 |
| B2 mean + k std | `{"k": 2.5}` | 0.718 ± 0.014 | 0.371 ± 0.017 | 0.623 ± 0.017 | 0.972 ± 0.000 | 0.199 ± 0.000 | 0.250 ± 0.000 | -413.3 ± 8.7 | 65.5 ± 2.9 | 0.081 ± 0.002 | 81.1 ± 2.2 |
| S-robust median + lambda MAD | `{"lam": 5}` | 0.655 ± 0.018 | 0.332 ± 0.015 | 0.561 ± 0.018 | 0.944 ± 0.000 | 0.215 ± 0.001 | 0.268 ± 0.002 | -540.2 ± 8.4 | 44.6 ± 2.9 | 0.164 ± 0.003 | 164.1 ± 3.3 |
| R2 EVT (Yu et al. 2024) | `{}` | 0.639 ± 0.039 | 0.245 ± 0.007 | 0.636 ± 0.039 | 0.806 ± 0.032 | 0.164 ± 0.001 | 0.171 ± 0.002 | -284.8 ± 6.3 | 61.7 ± 2.1 | 0.017 ± 0.003 | 17.1 ± 2.7 |
| B3 / RePAD2 rule (Lee & Lin 2023) | `{"w": 400}` | 0.743 ± 0.008 | 0.207 ± 0.010 | 0.718 ± 0.011 | 0.917 ± 0.000 | 0.018 ± 0.000 | 0.070 ± 0.001 | -28.7 ± 3.6 | 65.7 ± 1.7 | 0.025 ± 0.001 | 24.5 ± 1.3 |
| B3-k rolling mean + k std | `{"k": 3, "w": 400}` | 0.743 ± 0.008 | 0.207 ± 0.010 | 0.718 ± 0.011 | 0.917 ± 0.000 | 0.018 ± 0.000 | 0.070 ± 0.001 | -28.7 ± 3.6 | 65.7 ± 1.7 | 0.025 ± 0.001 | 24.5 ± 1.3 |
| P1 rolling median + lambda MAD (proposed) | `{"lam": 6, "memory": "plain", "w": 50}` | 0.544 ± 0.027 | 0.161 ± 0.002 | 0.490 ± 0.039 | 0.769 ± 0.033 | 0.035 ± 0.002 | 0.080 ± 0.003 | -113.6 ± 8.4 | 50.4 ± 4.7 | 0.068 ± 0.001 | 68.2 ± 1.4 |

Per-seed values: `test_per_seed.csv`; per file and seed: `test_metrics.csv`,
`clean_metrics.csv`; test-segment alarms and thresholds: `test_traces/`.
