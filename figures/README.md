# Figures

Every figure of the project, in one place (this folder replaces the former
separate images repository). Each is stored as PNG (300 dpi) for viewing and
PDF for print, except the paper previews, which are PNG only. Nothing here is drawn by hand: every figure is
written by a script from saved results, and the script is named under each
figure.

| Folder | What it holds | Written by |
|---|---|---|
| [paper/](paper) | PNG previews of the five figures in the paper, at print size | `scripts/make_paper_figures.py --png-dir figures/paper` (the PDFs used by LaTeX are in `paper/figures/`) |
| [phase1_eda/](phase1_eda) | Data exploration | `scripts/eda.py` |
| [phase2_model/](phase2_model) | Model training | `scripts/plot_phase2.py` |
| [phase3_thresholds/](phase3_thresholds) | Tuning and test results | `scripts/plot_phase3.py`, `plot_phase3_results.py`, `plot_proposal.py` |

Contents: [Figures in the paper](#figures-in-the-paper) ·
[Phase 1: data](#phase-1-data) · [Phase 2: model](#phase-2-model) ·
[Phase 3: thresholds and results](#phase-3-thresholds-and-results)

---

## Figures in the paper

| Paper | File | Section of the paper |
|---|---|---|
| Fig. 1 | [paper/fig1_windows_vs_split.png](paper/fig1_windows_vs_split.png) | III. Data, file selection |
| Fig. 2 | [paper/fig2_pipeline.png](paper/fig2_pipeline.png) (TikZ, drawn in `paper/main.tex`) | IV. Method |
| Fig. 3 | [paper/fig3_paired_per_file.png](paper/fig3_paired_per_file.png) | VI. Results, hypothesis tests |
| Fig. 4 | [paper/fig4_category_improvement.png](paper/fig4_category_improvement.png) | VI. Results, categories |
| Fig. 5 | [paper/fig5_masking_example.png](paper/fig5_masking_example.png) | VII. Discussion, why robustness did not help |

Files in `paper/` are named by their figure number in the paper. Files in
the phase folders keep the numbering of their phase.

### Fig. 1: where the anomaly windows fall relative to the split
Only 18 series have a window in the test part (coloured names); these form
the evaluation set.

![Fig. 1](paper/fig1_windows_vs_split.png)

### Fig. 2: pipeline
One shared LSTM autoencoder produces the error e(t); the eight methods differ
only in the threshold rule.

![Fig. 2](paper/fig2_pipeline.png)

### Fig. 3: P1 against Base and B3, per series
Each point is one evaluation series (mean over 3 seeds). On composite F1, P1
is worse than both on 13 of 18 series.

![Fig. 3](paper/fig3_paired_per_file.png)

### Fig. 4: improvement of P1 per NAB category
Descriptive only; categories hold 1 to 5 series.

![Fig. 4](paper/fig4_category_improvement.png)

### Fig. 5: the masking effect on nyc_taxi
(a) the whole test segment, (b) the last anomaly window enlarged, (c) the
alarms in (b). P1's threshold climbs with the anomaly, so only 1 error inside
the window exceeds it, against 30 for B3 and 73 for Base.

![Fig. 5](paper/fig5_masking_example.png)

---

## Phase 1: data

Explained in [REPORT.md, sections 3 to 5](../REPORT.md#3-the-data).

**fig01_example_per_category**: one series per NAB category, anomaly windows
shaded, split boundaries dashed.

![](phase1_eda/fig01_example_per_category.png)

**fig02_labelled_share**: share of points inside anomaly windows, overall
and per category (about 10 % in every labelled category).

![](phase1_eda/fig02_labelled_share.png)

**fig03_value_distributions**: value distribution of every series, grouped by
category. Spike-dominated series have most values squeezed near zero.

![](phase1_eda/fig03_value_distributions.png)

**fig04_lengths_and_sampling**: series lengths and sampling intervals per
category.

![](phase1_eda/fig04_lengths_and_sampling.png)

**fig05_windows_vs_split**: exploration version of paper Fig. 1.

![](phase1_eda/fig05_windows_vs_split.png)

**fig06_windows_per_part_and_contamination**: windows touching train,
validation and test per series, and the share of training points that lie
inside anomaly windows (the model learns these as normal, by design).

![](phase1_eda/fig06_windows_per_part_and_contamination.png)

---

## Phase 2: model

Explained in [REPORT.md, sections 6 and 7](../REPORT.md#6-the-model).

**fig01_loss_curves**: training and early-stopping loss per epoch for four
series (seed 15).

![](phase2_model/fig01_loss_curves.png)

**fig02_reconstruction_error**: `nyc_taxi`, seed 15: the series and its
reconstruction error over the whole file, with anomaly windows and split
boundaries.

![](phase2_model/fig02_reconstruction_error.png)

---

## Phase 3: thresholds and results

Explained in [REPORT.md, sections 8 to 13](../REPORT.md#8-the-eight-threshold-rules).

**fig01_tuning_objective**: validation composite F1 against each rule's
parameters; the selected values are circled.

![](phase3_thresholds/fig01_tuning_objective.png)

**fig02_test_metrics**: test results of all eight rules on six measures
(mean and sd over seeds).

![](phase3_thresholds/fig02_test_metrics.png)

**fig03_paired_per_file**: exploration version of paper Fig. 3.

![](phase3_thresholds/fig03_paired_per_file.png)

**fig04_example_trace**: full-width version of the `nyc_taxi` test segment
with the Base, B3 and P1 thresholds and alarms (paper Fig. 5 is the
single-column version with the enlarged window).

![](phase3_thresholds/fig04_example_trace.png)

**fig05_category_improvement**: exploration version of paper Fig. 4.

![](phase3_thresholds/fig05_category_improvement.png)

**fig06_detection_delay**: how late the first alarm comes inside each
detected window, and how many windows each rule missed.

![](phase3_thresholds/fig06_detection_delay.png)
