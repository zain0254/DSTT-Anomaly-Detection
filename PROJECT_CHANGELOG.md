# Project changelog

## 2026-11-18 — Manuscript revision (`paper/main.tex`)

Structural and stylistic revision of the paper. No experiment was rerun and no result file changed;
every number in the revised text is traced to `results/` or a code constant by
`scripts/check_paper_numbers.py` (382 numeric tokens, 0 mismatches, 0 unmapped; 23 numbers written as
words, all OK; generated tables identical to a fresh regeneration). Preamble, packages, macros,
citation keys, equations, tables, figures, algorithms and existing labels are unchanged; the build has
no LaTeX errors and no undefined references.

### Ethics and reproducibility
- Renamed *Ethics and Responsible Data Science* to *Ethics and Reproducibility* and rewrote it around
  public artifacts: NAB as the only data source (no PII, no user content, no human-subject data);
  licenses (NAB data and scorer MIT; PyTorch, NumPy, pandas, SciPy, scikit-learn BSD-style; Python and
  Matplotlib PSF-based; repository MIT); reproducibility (NAB pinned to `ea702d7`, exact package pins,
  seeds 15/42/123 with PyTorch deterministic algorithms, train-only scaling, stored grids, selections,
  108 error series and per-seed metrics, number-check script).
- Removed the sentence on AI-tool assistance, as requested.
- Code URL changed to `https://github.com/zain0254/DSTT-Anomaly-Detection` (paper body and Code
  availability). Both this URL and the old `...-Detection-` URL resolve to the same repository.

### Claims grounded in artifacts
- Abstract, Introduction opening, and Conclusion rebuilt from scratch around measured values: 30-step
  windows, 64-unit LSTM encoder/decoder (50,497 parameters), selected P1 setting (w = 50, λ = 6, plain
  memory), B3 window (w = 400), FPR 18.9% → 1.8% (B3) / 3.5% (P1), composite F1 0.733 vs 0.544
  (p = 0.035), 6.7 of 25 windows missed by P1.
- Added Eq. (1), the explicit P1 threshold (rolling median + λ · max(1.4826 · MAD, floor)).
- Rewrote the threshold-rule list so every item has the same form (rule: formula, what is tuned); moved
  the R2 and floor details into the following paragraph.
- New *NAB cost profiles* paragraph: corpus scores for B3, P1 and Base under standard, reward-low-FP and
  reward-low-FN profiles (from `test_summary.csv`), explained by the scorer's fpWeight 0.11 → 0.22.
- H2 explanation: two streams where Base flags every high-variance normal point
  (`rds_cpu_utilization_cc0c53`, `grok_asg_anomaly`) contribute 78% of the summed reduction.
- Exploratory B3 (w = 600): opposite moves of point F1 and composite F1 explained by point recall
  0.131 → 0.163 and event recall 0.917 → 0.877.

### Failure modes (new Section VII-C and Table IV)
- New subsection *Failure modes on individual streams* with Table IV (test/train error ratio, event
  recall of P1 and B1, composite-F1 deltas against B1 and Base), all computed from
  `results/phase3/test_traces/*.npz`, `test_metrics.csv` and `error_drift.csv`:
  - `ambient_temperature_system_failure`: slow drift inside a 363-step window; P1 raises 0 alarms in
    every seed, Base 102.7 on average.
  - `rds_cpu_utilization_cc0c53`: level shift before the window (test/train error ratio 31.9); static
    rules alarm on every point, P1 adapts and misses the window; same mechanism behind P1's wins on
    `grok_asg_anomaly` and `ec2_cpu_utilization_ac20cd`.
  - `speed_7578`: two windows 7 steps apart; 29 of 50 buffered errors come from the first window, past
    the median's 50% breakdown point, so the threshold climbs to 2.9× the largest error of the second.
  - `occupancy_t4013`: ~32% error rise does not clear six scaled MADs; P1 detects 1 of 6 window instances.
- States that cold start cannot occur: rolling buffers fill at the start of the training stream and the
  grid admits only window lengths that fit before validation (`pipeline/methods.py`, `w_allowed`).
- The repository has no IQR baseline, so failure cases are compared with the existing baselines
  (Base, B1 percentile, B2 mean + kσ, B3).

### Register and structure
- Removed filler transitions and throat-clearing openers; replaced nominal chains with active verbs.
- Results prose no longer walks through table cells; it explains why the deltas occur (static bar vs
  drifting test errors, buffer saturation, recall cost of λ = 6, point-F1 crediting).
- Related Work: replaced "Author X proposed…" sequences with integral and grouped comparative citations.
- Contribution list (4 items), research questions RQ1–RQ4 and proposal hypotheses H1–H3 rewritten as
  grammatically parallel lists; RQs and hypotheses moved from prose into labelled lists.
- Merged *Reproducibility and resources* into a short computing-resources paragraph; repository details
  now live in *Ethics and Reproducibility*.
- New labels: `eq:p1`, `sec:metrics`, `sec:drift`, `sec:failures`, `tab:failure`.

### Tooling
- `scripts/check_paper_numbers.py`: specs updated to the revised text; new helpers recompute the
  failure-mode values from the saved test traces (`win_stats`) and per-file test metrics; obsolete specs
  removed. `paper/number_check.md` regenerated.

### Notes
- Length grew from 8 to 9 pages (mainly the new failure-mode subsection and Table IV).
- `submission/DSTT_paper_latex.zip` still contains the previous `main.tex`; regenerate it before
  submitting.
- `IEEEtran.bst` is not part of the repository; local verification builds used `ieeetr.bst` as a
  stand-in. Overleaf and full TeX Live installs provide `IEEEtran.bst`.
