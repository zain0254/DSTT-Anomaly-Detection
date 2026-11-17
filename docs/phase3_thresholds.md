# Phase 3: threshold tuning and test scoring

Produced by `scripts/phase3.py` (tuning 135 s, test 6 s, 4 worker processes)
under the protocol pre-registered in docs/decisions.md D-26. Numbers below are
copied from `results/phase3/*.csv`.

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

## Figures (primary results)

`scripts/plot_phase3_results.py`, `figures/phase3_thresholds/`:
- fig01_tuning_objective: validation composite F1 against each method's
  parameters (from `scripts/plot_phase3.py`).
- fig02_test_metrics: composite F1, point F1, event recall, FPR,
  high-variance FPR and clean-stream false alarms per 1,000 points per
  method; bars are the mean over seeds, error bars the sd over seeds.
- fig03_paired_per_file: per evaluation file (mean over seeds), P1 against
  Base and against B3 on composite F1 and FPR. Composite F1: P1 better on 4
  and worse on 13 of 18 files against both. FPR: P1 lower on 10 and higher on
  6 against Base, lower on 3 and higher on 11 against B3.
- fig04_example_trace: nyc_taxi, seed 15 (the file and seed used in the
  Phase 2 figure, chosen before looking at threshold results), test segment
  with the error series, the Base, B3 and P1 thresholds, their alarms and
  the NAB windows.

## Pre-registered hypothesis tests (D-26)

`scripts/hypothesis_tests.py`, output `results/phase3/hypothesis_tests.csv`.
Unit = file (mean over 3 seeds). Paired two-sided Wilcoxon signed-rank test
(zero differences dropped), mean difference P1 - other with a percentile
bootstrap 95 % CI (10,000 resamples, seed 2026), wins/ties/losses for P1 in
the metric's better direction. Primary: P1 vs Base, tested alone. Secondary:
P1 vs the six other methods, Holm-corrected within each measure. alpha = 0.05
(D-26 does not state alpha; the conventional level is used). B3-k is
identical to B3 on test (both selected k = 3, w = 400), so the secondary
family contains the same comparison twice, which makes the Holm correction
slightly more conservative than necessary.

Verdicts (primary comparison P1 vs Base):
- **H1 (composite F1): not supported.** P1 is lower than Base: mean
  difference -0.190 [95 % CI -0.351, -0.008],
  4 wins / 1 tie / 13 losses, p = 0.035. Base is
  significantly better.
- **H2 (high-variance FPR, n = 17): not supported.** P1 has a lower
  mean high-variance FPR (difference -0.151, bootstrap CI
  -0.328 to -0.012), but the pre-registered Wilcoxon test
  is not significant (p = 0.109; 9 wins / 3 ties /
  5 losses). The bootstrap CI of the mean excludes 0 while the rank
  test does not reach significance; the pre-registered decision rule is the
  Wilcoxon test. Secondary H2 measures on the clean streams: no significant
  difference from Base (clean FPR difference 0.0068, p = 0.163;
  false alarms per 1,000 points difference 6.75, p = 0.163).
- **H3 (every-step NAB standard, per file): not supported.** No significant
  difference (mean difference 359.4, CI -149.5 to
  994.5, p = 0.832).

Secondary comparisons with a Holm-adjusted p < 0.05: P1 worse than B3 (and
B3-k) on composite F1, clean FPR, clean false alarms per 1,000 points and
NAB; P1 worse than R2 on clean FPR and clean false alarms per 1,000 points;
P1 better than B1 on high-variance FPR and NAB. All others: no significant
difference.

| H | Metric | Comparison | vs | n | P1 | Other | Diff P1 - other [95 % CI] | W/T/L | p | p (Holm) | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | composite_f1 | primary | Base | 18 | 0.5437 | 0.7333 | -0.1896 [-0.3507, -0.007597] | 4/1/13 | 0.0352 | 0.0352 | P1 worse |
| H1 | composite_f1 | secondary | B1 | 18 | 0.5437 | 0.5868 | -0.04312 [-0.2088, 0.1287] | 8/0/10 | 0.6095 | 0.6095 | no significant difference |
| H1 | composite_f1 | secondary | B2 | 18 | 0.5437 | 0.7176 | -0.1739 [-0.3377, 0.006334] | 4/1/13 | 0.0442 | 0.1769 | no significant difference |
| H1 | composite_f1 | secondary | S-robust | 18 | 0.5437 | 0.6548 | -0.1111 [-0.2783, 0.07191] | 5/2/11 | 0.1788 | 0.5364 | no significant difference |
| H1 | composite_f1 | secondary | R2 | 18 | 0.5437 | 0.6393 | -0.09557 [-0.282, 0.0992] | 5/2/11 | 0.2146 | 0.5364 | no significant difference |
| H1 | composite_f1 | secondary | B3 | 18 | 0.5437 | 0.7429 | -0.1991 [-0.3403, -0.07096] | 4/1/13 | 0.0075 | 0.0449 | P1 worse |
| H1 | composite_f1 | secondary | B3-k | 18 | 0.5437 | 0.7429 | -0.1991 [-0.3403, -0.07096] | 4/1/13 | 0.0075 | 0.0449 | P1 worse |
| H2 | hv_fpr | primary | Base | 17 | 0.08009 | 0.2309 | -0.1508 [-0.3278, -0.01179] | 9/3/5 | 0.1093 | 0.1093 | no significant difference |
| H2 | hv_fpr | secondary | B1 | 17 | 0.08009 | 0.4883 | -0.4082 [-0.5455, -0.2813] | 17/0/0 | 0.0000 | 0.0001 | P1 better |
| H2 | hv_fpr | secondary | B2 | 17 | 0.08009 | 0.2504 | -0.1703 [-0.3461, -0.03158] | 10/3/4 | 0.0640 | 0.2559 | no significant difference |
| H2 | hv_fpr | secondary | S-robust | 17 | 0.08009 | 0.2681 | -0.188 [-0.3539, -0.05586] | 10/4/3 | 0.0131 | 0.0654 | no significant difference |
| H2 | hv_fpr | secondary | R2 | 17 | 0.08009 | 0.1705 | -0.09044 [-0.2816, 0.06285] | 5/4/8 | 0.9165 | 1.0000 | no significant difference |
| H2 | hv_fpr | secondary | B3 | 17 | 0.08009 | 0.06992 | 0.01017 [-0.03479, 0.05705] | 6/5/6 | 0.6949 | 1.0000 | no significant difference |
| H2 | hv_fpr | secondary | B3-k | 17 | 0.08009 | 0.06992 | 0.01017 [-0.03479, 0.05705] | 6/5/6 | 0.6949 | 1.0000 | no significant difference |
| H2 | clean_fpr | primary | Base | 18 | 0.06824 | 0.06149 | 0.006755 [-0.0622, 0.0593] | 5/1/12 | 0.1626 | 0.1626 | no significant difference |
| H2 | clean_fpr | secondary | B1 | 18 | 0.06824 | 0.1873 | -0.1191 [-0.2168, -0.03757] | 12/1/5 | 0.0217 | 0.0650 | no significant difference |
| H2 | clean_fpr | secondary | B2 | 18 | 0.06824 | 0.08106 | -0.01281 [-0.08451, 0.0446] | 6/1/11 | 0.3088 | 0.3860 | no significant difference |
| H2 | clean_fpr | secondary | S-robust | 18 | 0.06824 | 0.1641 | -0.09589 [-0.1716, -0.02638] | 8/1/9 | 0.1930 | 0.3860 | no significant difference |
| H2 | clean_fpr | secondary | R2 | 18 | 0.06824 | 0.0171 | 0.05115 [0.01933, 0.08213] | 1/2/15 | 0.0052 | 0.0209 | P1 worse |
| H2 | clean_fpr | secondary | B3 | 18 | 0.06824 | 0.02451 | 0.04374 [0.02153, 0.06931] | 3/1/14 | 0.0016 | 0.0099 | P1 worse |
| H2 | clean_fpr | secondary | B3-k | 18 | 0.06824 | 0.02451 | 0.04374 [0.02153, 0.06931] | 3/1/14 | 0.0016 | 0.0099 | P1 worse |
| H2 | clean_false_alarms_per_1000 | primary | Base | 18 | 68.24 | 61.49 | 6.755 [-62.2, 59.3] | 5/1/12 | 0.1626 | 0.1626 | no significant difference |
| H2 | clean_false_alarms_per_1000 | secondary | B1 | 18 | 68.24 | 187.3 | -119.1 [-216.8, -37.57] | 12/1/5 | 0.0217 | 0.0650 | no significant difference |
| H2 | clean_false_alarms_per_1000 | secondary | B2 | 18 | 68.24 | 81.06 | -12.81 [-84.51, 44.6] | 6/1/11 | 0.3088 | 0.3860 | no significant difference |
| H2 | clean_false_alarms_per_1000 | secondary | S-robust | 18 | 68.24 | 164.1 | -95.89 [-171.6, -26.38] | 8/1/9 | 0.1930 | 0.3860 | no significant difference |
| H2 | clean_false_alarms_per_1000 | secondary | R2 | 18 | 68.24 | 17.1 | 51.15 [19.33, 82.13] | 1/2/15 | 0.0052 | 0.0209 | P1 worse |
| H2 | clean_false_alarms_per_1000 | secondary | B3 | 18 | 68.24 | 24.51 | 43.74 [21.53, 69.31] | 3/1/14 | 0.0016 | 0.0099 | P1 worse |
| H2 | clean_false_alarms_per_1000 | secondary | B3-k | 18 | 68.24 | 24.51 | 43.74 [21.53, 69.31] | 3/1/14 | 0.0016 | 0.0099 | P1 worse |
| H3 | nab_standard | primary | Base | 18 | -173.1 | -532.5 | 359.4 [-149.5, 994.5] | 10/0/8 | 0.8317 | 0.8317 | no significant difference |
| H3 | nab_standard | secondary | B1 | 18 | -173.1 | -1352 | 1179 [377.7, 2177] | 14/0/4 | 0.0077 | 0.0395 | P1 better |
| H3 | nab_standard | secondary | B2 | 18 | -173.1 | -583.5 | 410.4 [-98.03, 1040] | 11/0/7 | 0.4423 | 1.0000 | no significant difference |
| H3 | nab_standard | secondary | S-robust | 18 | -173.1 | -773.2 | 600.1 [36.1, 1270] | 9/0/9 | 0.3465 | 1.0000 | no significant difference |
| H3 | nab_standard | secondary | R2 | 18 | -173.1 | -419.1 | 246 [-267.7, 890.7] | 7/0/11 | 0.5798 | 1.0000 | no significant difference |
| H3 | nab_standard | secondary | B3 | 18 | -173.1 | -64.34 | -108.7 [-197.3, -34.97] | 3/0/15 | 0.0066 | 0.0395 | P1 worse |
| H3 | nab_standard | secondary | B3-k | 18 | -173.1 | -64.34 | -108.7 [-197.3, -34.97] | 3/0/15 | 0.0066 | 0.0395 | P1 worse |

## Exploratory: extended edge grids (D-27, post-hoc)

**This section is post-hoc.** It was decided after the primary results and
the pre-registered tests above were known, because four selections sat on
grid edges. It does not change the primary results or the verdicts. Run with
`scripts/phase3.py --grids exploratory`; outputs in
`results/phase3_exploratory/`.

Grid additions: P1 w in {10, 20, 30}; B3 and B3-k w in {600, 800}; B1 p in
{80, 85}. Window allowance (`window_log.csv`): w = 800 was skipped for
speed_7578 (647 errors before validation) for B3 (3 seeds) and B3-k (10 k
values x 3 seeds), 33 candidates in total; no substitution was needed at
test. The 10,944 candidates shared with the primary grid reproduce the
primary validation metrics exactly (maximum absolute difference 0).

Re-selection on validation (`selected_params.csv`):
- B1: p = 90 again (p = 80 and 85 score lower on validation).
- P1: lambda = 6, plain, w = 50 again (w = 10, 20, 30 score lower).
- B3 and B3-k: w = 600 (validation composite F1 0.596 vs 0.591 at w = 400);
  B3-k again selects k = 3, so it stays identical to B3.
- All other methods are unchanged, and their test results are identical to
  the primary ones.

Test results of the only changed method (mean ± sd over seeds):

| B3 / B3-k | Selected w | Composite F1 | Point F1 | Event recall | FPR | HV FPR | Corpus NAB std (every step) | Corpus NAB std (first step) | Clean FA / 1000 |
|---|---|---|---|---|---|---|---|---|---|
| primary (D-26) | 400 | 0.743 ± 0.008 | 0.207 ± 0.010 | 0.917 ± 0.000 | 0.018 ± 0.000 | 0.070 ± 0.001 | -28.7 ± 3.6 | 65.7 ± 1.7 | 24.5 ± 1.3 |
| exploratory (D-27) | 600 | 0.713 ± 0.008 | 0.240 ± 0.003 | 0.877 ± 0.011 | 0.018 ± 0.000 | 0.064 ± 0.001 | -16.8 ± 3.5 | 64.5 ± 2.4 | 24.0 ± 2.6 |

With w = 600, B3 has a lower composite F1 and event recall on test than with
the pre-registered w = 400, and a higher point F1 and every-step NAB. No
hypothesis test was run on the exploratory results.
