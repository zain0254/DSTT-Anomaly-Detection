# Questions and answers

Short answers to the questions a reviewer is most likely to ask, each with
the file that backs it up. For the full story in plain words, read
[REPORT.md](REPORT.md).

**About the question and the result**

- [What is the project in one sentence?](#what-is-the-project-in-one-sentence)
- [Did the proposed method work?](#did-the-proposed-method-work)
- [If P1 lost, what is the contribution?](#if-p1-lost-what-is-the-contribution)
- [Why did the base rule catch almost every anomaly?](#why-did-the-base-rule-catch-almost-every-anomaly)
- [Why did P1 lose to the simple rolling mean rule?](#why-did-p1-lose-to-the-simple-rolling-mean-rule)
- [Why does point F1 rank the rules differently from composite F1?](#why-does-point-f1-rank-the-rules-differently-from-composite-f1)

**About the data and the split**

- [Why NAB?](#why-nab)
- [Why only 18 series in the evaluation?](#why-only-18-series-in-the-evaluation)
- [Why train the model without using the labels?](#why-train-the-model-without-using-the-labels)
- [Is there any leakage from the test data?](#is-there-any-leakage-from-the-test-data)
- [Some series are in both the tuning and the evaluation set. Is that a problem?](#some-series-are-in-both-the-tuning-and-the-evaluation-set-is-that-a-problem)
- [Why no resampling or filling of gaps?](#why-no-resampling-or-filling-of-gaps)

**About the model and the methods**

- [Is the model the same as in the base paper?](#is-the-model-the-same-as-in-the-base-paper)
- [Why PyTorch on CPU and not Keras on the GPU?](#why-pytorch-on-cpu-and-not-keras-on-the-gpu)
- [Why is B3-k identical to B3?](#why-is-b3-k-identical-to-b3)
- [Why were the Yu et al. and RePAD2 methods not run as in the original papers?](#why-were-the-yu-et-al-and-repad2-methods-not-run-as-in-the-original-papers)
- [Some models stopped after one or two epochs. Why keep them?](#some-models-stopped-after-one-or-two-epochs-why-keep-them)

**About evaluation and statistics**

- [Why composite F1 as the main measure?](#why-composite-f1-as-the-main-measure)
- [Why no point adjustment?](#why-no-point-adjustment)
- [How do you know the NAB scores are correct?](#how-do-you-know-the-nab-scores-are-correct)
- [Why are the NAB scores negative?](#why-are-the-nab-scores-negative)
- [Why the Wilcoxon test with n = 18 and not a t-test over all runs?](#why-the-wilcoxon-test-with-n--18-and-not-a-t-test-over-all-runs)
- [Were the hypotheses fixed before the results?](#were-the-hypotheses-fixed-before-the-results)
- [Several selections were at the edge of their grid. Is the tuning unfair?](#several-selections-were-at-the-edge-of-their-grid-is-the-tuning-unfair)

**About the repository and reproducibility**

- [How do I reproduce the results?](#how-do-i-reproduce-the-results)
- [How do I know the numbers in the paper are correct?](#how-do-i-know-the-numbers-in-the-paper-are-correct)
- [What changed from the earlier version of the project?](#what-changed-from-the-earlier-version-of-the-project)
- [Are the references accessible?](#are-the-references-accessible)
- [Was AI used?](#was-ai-used)

---

## About the question and the result

### What is the project in one sentence?
We test whether replacing the base paper's fixed mean + 3 sd threshold with
a rolling median + MAD threshold improves LSTM autoencoder anomaly detection
on NAB, under a protocol where no test information can leak into training or
tuning. Overview: [REPORT.md, section 1](REPORT.md#1-the-question).

### Did the proposed method work?
No. P1 had a composite F1 of 0.544 against 0.733 for the base rule, and the
pre-registered test found it significantly worse (p = 0.035). None of the
pre-registered hypotheses was supported. Source:
[results/phase3/hypothesis_tests.csv](../results/phase3/hypothesis_tests.csv),
[results/phase3/proposal_hypotheses.csv](../results/phase3/proposal_hypotheses.csv).

### If P1 lost, what is the contribution?
1. A clear, tested answer: on this benchmark, making the threshold
   **adaptive** matters; making it **robust** does not.
2. A leakage-free benchmark of eight threshold rules on one shared model,
   including two published rules (RePAD2, Yu et al. EVT).
3. An explanation with evidence: the base rule fails through drift of the
   error level (test errors larger than training errors in 17 of 18 series),
   and the short-window robust rule suffers from the masking it was meant to
   prevent.
4. A fully reproducible repository in which every number in the paper is
   checked against the result files by a script.

### Why did the base rule catch almost every anomaly?
Because its threshold is too low on later data. Errors on the test part were
larger than on the training part (median ratio 1.20), so a threshold fitted
on training errors fires often. That catches 24 of 25 test windows but also
flags 18.9 % of normal points. Source:
[results/phase3/error_drift.csv](../results/phase3/error_drift.csv),
[results/phase3/detection_delay.csv](../results/phase3/detection_delay.csv).

### Why did P1 lose to the simple rolling mean rule?
P1 chose a short window (w = 50) on validation. In a short window an anomaly
soon makes up a large share of the values, so the median and MAD rise with
it and the anomaly hides under the threshold. B3 chose w = 400, which a
short anomaly barely moves. The paper's Fig. 5 shows it on `nyc_taxi`:
inside the last window 1 error is above P1's threshold, 30 above B3's.
See [REPORT.md, section 12](REPORT.md#12-why-the-proposed-rule-lost).

### Why does point F1 rank the rules differently from composite F1?
Point F1 rewards flagging many points inside the long NAB windows, so rules
that alarm a lot (B1, B2, Base) score well on it even with many false
alarms. Composite F1 counts a window once (event recall) and still penalises
false alarms through precision, which is closer to what an operator needs.
Both are reported in the paper's Table II.

## About the data and the split

### Why NAB?
It is public, labelled, widely used for streaming anomaly detection, has an
official scorer, and contains cloud server metrics (AWS CloudWatch) close to
the base paper's AIOps setting. Its known weaknesses are discussed in the
paper's Limitations (Wu & Keogh 2023).

### Why only 18 series in the evaluation?
Only 18 of the 58 series have an anomaly window in their last 20 % (the test
part). The others cannot be scored on the test part at all. All 25 test
windows are in these 18 series. Source:
[results/phase1_eda/file_summary.csv](../results/phase1_eda/file_summary.csv),
figure [fig05_windows_vs_split](../figures/paper/fig05_windows_vs_split.png).

### Why train the model without using the labels?
A deployed detector never knows where the anomalies in its history are, and
the base paper fits its threshold on all training errors. Removing labelled
windows (as the earlier version did) gives the static rules information that
does not exist in practice. Decision D-04 in [decisions.md](decisions.md).

### Is there any leakage from the test data?
No. The scaler is fitted on the training part, the model learns only from
the training part, every threshold parameter is chosen on validation parts,
and the test parts are scored once with frozen parameters. The split is in
time order. Decisions D-04 to D-06 and D-26 in [decisions.md](decisions.md).

### Some series are in both the tuning and the evaluation set. Is that a problem?
Six series are in both. For them, tuning uses the validation part and
evaluation uses the test part, which do not overlap in time, and every
threshold rule is causal (it only looks backwards). The one series whose
anomaly crosses from validation into test (`grok_asg_anomaly`) was removed
from tuning for this reason. Decision D-21.

### Why no resampling or filling of gaps?
NAB labels and scores rows. Filling time gaps would add points that NAB does
not score and move window edges. Decision D-20.

## About the model and the methods

### Is the model the same as in the base paper?
Yes: LSTM(64) encoder, last hidden state repeated over the 30-step window,
LSTM(64) decoder, a linear output per step, MAE score, MinMax scaling,
Adam and MSE loss. Details the paper does not give (batch size, learning
rate, patience) use the Keras defaults. Decisions D-11 and D-23.

### Why PyTorch on CPU and not Keras on the GPU?
TensorFlow has no native Windows GPU support after version 2.10. The model
has about 50 thousand parameters, so a GPU gives little speed-up, and CPU
runs are deterministic for a fixed seed. All 108 fits took 47.5 minutes.
Decisions D-11, D-12; [hardware.md](hardware.md).

### Why is B3-k identical to B3?
B3-k was added so that the rolling mean rule gets the same tuning budget as
P1 (both tune a multiplier and a window). On validation it chose k = 3,
which is exactly B3's fixed value, so the two rules give the same alarms.
This also shows that the RePAD2 default was already the best multiplier.
Source: [results/phase3/selected_params.csv](../results/phase3/selected_params.csv).

### Why were the Yu et al. and RePAD2 methods not run as in the original papers?
RePAD2's windows (1,440 to 16,128 steps) are longer than many NAB series, so
its window was tuned like the other rolling rules. Yu et al.'s dynamic update
(their Eq. 9) is not fully specified in the paper, so only the fully
specified static part was reproduced. Both rules are applied to our model's
errors so that only the threshold differs. Decision D-24.

### Some models stopped after one or two epochs. Why keep them?
5 of 108 fits had their best epoch at 1 or 2. The training protocol was
fixed in advance and applied to every fit; retraining only the ones that
looked bad would treat fits differently after seeing their outcome.
Decision D-25, [phase2_model.md](phase2_model.md).

## About evaluation and statistics

### Why composite F1 as the main measure?
It combines point precision with event recall (Garg et al. 2022): a rule is
rewarded for catching each anomaly at least once and penalised for false
alarms, without the inflation of point adjustment. It was declared as the
main measure before any result existed (D-26).

### Why no point adjustment?
Point adjustment marks a whole window as detected if one point in it is
flagged. Kim et al. (2022) show that this lets even random scores look
strong. We report plain point F1 instead.

### How do you know the NAB scores are correct?
Our scorer is a port of NAB's official code. Run on the full files with
NAB's settings, it reproduces NAB's published results for three detectors
and three scoring profiles to within about 5e-12 per file. Source:
[results/scorer_validation.csv](../results/scorer_validation.csv),
[scripts/validate_scorer.py](../scripts/validate_scorer.py).

### Why are the NAB scores negative?
The primary NAB score passes every alarm step to the scorer, and every false
positive step costs points, so rules with many alarms go negative. The
first-step variant (only the first step of each alarm run) is also reported
and is positive for all rules. Neither is comparable to the NAB leaderboard,
which tunes on all labels and uses a probationary period (D-07).

### Why the Wilcoxon test with n = 18 and not a t-test over all runs?
The 3 seeds of one series are not independent observations of the method,
so we average them and treat each series as one unit (n = 18). The Wilcoxon
signed-rank test does not assume normal differences, which suits 18 values.
For the proposal's H2 we also ran the paired t-test because the proposal
named it; both results are reported. Decisions D-26, D-28.

### Were the hypotheses fixed before the results?
Yes. D-26 (tests, measures, comparisons) was committed to git before any
threshold was computed, and D-28 repeats the proposal's hypotheses exactly
as defended in October 2026. The git history shows the order. The only
post-hoc analysis (D-27) is labelled exploratory and kept in its own folder.

### Several selections were at the edge of their grid. Is the tuning unfair?
We checked this in the exploratory run (D-27): extending the grids did not
change P1's choice (w = 10, 20 and 30 scored lower on validation). B3 moved
to w = 600, which gave a lower test composite F1 than w = 400. The primary
conclusions do not change. Source:
[phase3_thresholds.md](phase3_thresholds.md), "Exploratory".

## About the repository and reproducibility

### How do I reproduce the results?
Follow the "Reproduce" table in the [README](../README.md#reproduce). The
whole pipeline runs on a laptop CPU in about one hour, most of it training.
The saved error series let you rerun every later step in minutes without
training.

### How do I know the numbers in the paper are correct?
[scripts/check_paper_numbers.py](../scripts/check_paper_numbers.py) reads
every number in the paper and recomputes it from the result files. The
output, [paper_number_check.md](paper_number_check.md), lists each number
with its source: 0 mismatches. The paper's tables are generated from the
CSV files by [scripts/make_paper_tables.py](../scripts/make_paper_tables.py).

### What changed from the earlier version of the project?
The earlier version tuned on data that was also reported as test data,
removed labelled windows from training, attributed the rolling rule to the
base paper, and used a slightly different model. All of this is listed with
file and line references in [old_vs_new.md](old_vs_new.md).

### Are the references accessible?
Yes. All 12 references are from 2021 onwards and each has a DOI or a direct
link in [paper/refs.bib](../paper/refs.bib) and
[references.md](references.md). Ten are free to read; two (Yu et al. 2024
and Jeon et al. 2023) are behind publisher paywalls but reachable by DOI.

### Was AI used?
Yes, as stated in the paper's ethics section: AI tools helped with coding,
literature search and editing. All experiments, numbers and conclusions were
checked by the authors, and the number check script ties every reported
value to a result file.
