# Phase 1a: exploratory data analysis of NAB

Produced by `scripts/eda.py` (runtime about 6 s). Every number below is read
from `results/phase1_eda/*.csv`; figures are in `figures/phase1_eda/` (PNG
300 dpi + PDF). Labels are used only through window positions; no model output
exists yet. Split boundaries follow `pipeline/data.split_bounds`:
train = rows [0, floor(0.6 n)), val = [floor(0.6 n), floor(0.8 n)), test = the rest.

## Corpus overview (`category_summary.csv`)

| Category | Files | Points | Windows | Labelled points | % labelled | Windows touching train / val / test |
|---|---|---|---|---|---|---|
| artificialNoAnomaly | 5 | 20,160 | 0 | 0 | 0.0 | 0 / 0 / 0 |
| artificialWithAnomaly | 6 | 24,192 | 6 | 2,418 | 10.0 | 1 / 5 / 0 |
| realAWSCloudwatch | 17 | 67,740 | 30 | 6,312 | 9.3 | 17 / 9 / 6 |
| realAdExchange | 6 | 9,610 | 14 | 960 | 10.0 | 9 / 5 / 1 |
| realKnownCause | 7 | 69,561 | 19 | 6,594 | 9.5 | 8 / 3 / 8 |
| realTraffic | 7 | 15,664 | 14 | 1,560 | 10.0 | 5 / 3 / 7 |
| realTweets | 10 | 158,631 | 33 | 15,651 | 9.9 | 25 / 6 / 3 |
| **All** | **58** | **365,558** | **116** | | **9.16** | **65 / 31 / 25** |

The part counts add up to 121, not 116, because 5 windows touch two parts
(`window_parts.csv`): 4 cross the train/val boundary
(ec2_disk_write_bytes_1ef3de, exchange-2_cpm_results, TravelTime_387,
Twitter_volume_GOOG) and 1 crosses the val/test boundary (grok_asg_anomaly,
window 2).

## Data quality (`file_summary.csv`)

- No NaN values in any file.
- 49 duplicate timestamps in 8 files: ec2_disk_write_bytes_1ef3de (11),
  ec2_network_in_5abac7 (11), ec2_request_latency_system_failure (11),
  machine_temperature_system_failure (12), exchange-2_cpc_results (1),
  exchange-2_cpm_results (1), occupancy_t4013 (1), speed_t4013 (1). NAB keeps
  these rows and scores them; we do the same, and window edges on a duplicate
  map to its first occurrence as in NAB (decision D-19).
- 25 files have irregular steps; together they miss 27,625 steps relative to
  their median sampling interval. The largest gaps are in TravelTime_387
  (7,472 missing steps for 2,500 rows), TravelTime_451 (5,231 / 2,162),
  rogue_agent_key_hold (3,456 / 1,882) and the other realTraffic files
  (2,180 to 2,376 each). We do not resample: NAB scores rows, and the LSTM
  window of 30 rows therefore spans a variable time in these files (D-20).
- art_flatline has zero training variance; art_daily_perfect_square_wave and
  art_increase_spike_density have at most 2 distinct training values. These
  are the files where the zero-MAD floor fallback (D-09 amendment) may
  apply. None of them has a validation or test window.

## Figures

**fig01_example_per_category.** One series per category (the file with the
most windows in the category), NAB windows shaded and the 60 % / 80 %
boundaries dashed. It shows the range of shapes the shared model has to
reconstruct: synthetic daily square waves, sparse spikes on a flat baseline
(AWS, AdExchange, Tweets), a strong daily/weekly cycle (nyc_taxi, 5 windows,
3 of them in test) and gappy traffic speed (speed_7578, 1,127 rows, 1,498
missing steps).

**fig02_labelled_share.** Share of points inside NAB windows: 9.16 % over the
corpus, 0.0 % for artificialNoAnomaly and 9.3 % to 10.0 % for every other
category. NAB sizes windows to about 10 % of each labelled file, so the
labelled share carries little information per category; the number and
position of windows matter more.

**fig03_value_distributions.** One box per file, values min-max scaled over
the whole file (for display only; the model scaler is fitted on the training
split). Spike-dominated files (most AWS files, the four exchange-3/exchange-4 files and all Tweets files)
have their interquartile range squeezed near 0 because a few spikes set the
maximum. This matters for the threshold rules: on these files most
reconstruction errors will be small and a few very large, which is the
setting where mean/std statistics and median/MAD statistics differ most.

**fig04_lengths_and_sampling.** Points per file and median sampling step per
category. Lengths range from 1,127 (speed_7578) to 22,695
(machine_temperature_system_failure); median 4,032. Tweets files have about
15,800 to 15,900 points each. Median steps: 5 min for most categories, 60 min
for AdExchange, 5/30/60 min within KnownCause and 5/10 min within Traffic.

**fig05_windows_vs_split.** For each of the 52 files with at least one
window, the window positions relative to the 60/20/20 split. Files with no
window in the test part are labelled "[no test window]" in grey: 34 of the
52. Only 18 files have a window touching the test part.

**fig06_windows_per_part_and_contamination.** Left: windows touching each part
per file. Right: training contamination, the share of training points inside
NAB windows. Because the model and RE_train use all training data (D-04),
these points are learned as normal.

## Training contamination

- 38 of the 52 labelled files have labelled points in the training part.
  Over all files, 18,609 of 219,318 training points (8.48 %) are inside
  windows. Per labelled file the median is 8.33 % and the maximum 16.91 %
  (iio_us-east-1_i-a2eb1cd9_NetworkIn).
- Effect on the study: anomalies in training lower the reconstruction error
  the model produces for similar anomalies later, and they inflate the mean
  and std of RE_train used by Base and B2 more than the median and MAD used
  by S-robust. This is part of what the comparison measures, so contaminated
  files are not removed; contamination is reported per file so results can be
  read against it.

## Proposed file selection (not final; owner to confirm)

Rules use window positions only, as in the old split protocol.

**Evaluation set E: the 18 files with at least one window touching the test
part.** 25 test windows, 7,162 labelled test points out of 28,202 test points
(25.4 %). Categories: AWS 5, KnownCause 5, Traffic 4, Tweets 3, AdExchange 1;
no artificial file has a test window. Training contamination in E ranges from
0.00 % to 11.87 % (Twitter_volume_CVS); 7 of the 18 files have none.

**Tuning set T: files with at least one window touching the validation
part,** used only through their validation segments (pooled, D-06). Three
variants:

| Variant | Files | Val windows | Labelled val points | Notes |
|---|---|---|---|---|
| T-all | 30 | 31 | 7,724 | includes 5 artificialWithAnomaly files |
| T-real | 25 | 26 | 5,709 | real categories only |
| T-real minus grok_asg_anomaly | 24 | 25 | 5,689 | removes the one window split across val and test |

**Recommendation: E (18 files) for evaluation and T-real minus
grok_asg_anomaly (24 files) for tuning.** Reasons:
1. E is the complete set of files that can be scored on the test part; any
   smaller set would drop test windows (only 25 exist).
2. E contains only real files, so tuning on real files keeps the tuning
   distribution matched to evaluation. The 5 artificial tuning files have one
   synthetic window each (flat middle, jumps, a no-jump shift, load-balancer
   spikes) and would make up 5 of 30 tuning files.
3. grok_asg_anomaly window 2 starts in validation and ends in test. Tuning on
   its validation part means part of the same anomaly event informs the
   parameters that are then scored on its test part. Removing grok from T
   (but keeping it in E) avoids this at the cost of 1 validation window.
4. Overlap between E and T is otherwise allowed: 7 files are in both
   (grok_asg_anomaly, rds_cpu_utilization_cc0c53, exchange-4_cpm_results,
   machine_temperature_system_failure, nyc_taxi, speed_7578,
   Twitter_volume_KO; 6 after removing grok). Their validation and test
   segments are disjoint in time and the threshold rules are causal.
5. Contamination is not used as a selection criterion: removing files by
   training contamination would select on labels in a way that favours some
   threshold rules (see above).

Files in neither set (17): the 5 artificialNoAnomaly files,
art_increase_spike_density, ec2_cpu_utilization_77c1ca / 825cc2 / c6585a,
ec2_network_in_257a54, iio_us-east-1_i-a2eb1cd9_NetworkIn,
exchange-2_cpc_results, rogue_agent_key_updown, TravelTime_451 and
Twitter_volume_AAPL / CRM / UPS. They have no window in validation or test.

Limitation: 25 test windows in 18 files is small; per-file comparisons will
have n = 18 and confidence intervals will be wide.

## Per-file summary (`file_summary.csv`)

Columns: n = rows; step = median sampling step (s); irr = irregular steps;
miss = missing steps; dup = duplicate timestamps; win = windows; lab % =
labelled share; tr/va/te = windows touching each part; train lab = labelled
training points (contamination, % of training rows).

| File | Category | n | step | irr | miss | dup | win | lab % | tr | va | te | train lab (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| art_daily_no_noise | artificialNoAnomaly | 4032 | 300 | 0 | 0 | 0 | 0 | 0.00 | 0 | 0 | 0 | 0 (0.00) |
| art_daily_perfect_square_wave | artificialNoAnomaly | 4032 | 300 | 0 | 0 | 0 | 0 | 0.00 | 0 | 0 | 0 | 0 (0.00) |
| art_daily_small_noise | artificialNoAnomaly | 4032 | 300 | 0 | 0 | 0 | 0 | 0.00 | 0 | 0 | 0 | 0 (0.00) |
| art_flatline | artificialNoAnomaly | 4032 | 300 | 0 | 0 | 0 | 0 | 0.00 | 0 | 0 | 0 | 0 (0.00) |
| art_noisy | artificialNoAnomaly | 4032 | 300 | 0 | 0 | 0 | 0 | 0.00 | 0 | 0 | 0 | 0 (0.00) |
| art_daily_flatmiddle | artificialWithAnomaly | 4032 | 300 | 0 | 0 | 0 | 1 | 10.00 | 0 | 1 | 0 | 0 (0.00) |
| art_daily_jumpsdown | artificialWithAnomaly | 4032 | 300 | 0 | 0 | 0 | 1 | 10.00 | 0 | 1 | 0 | 0 (0.00) |
| art_daily_jumpsup | artificialWithAnomaly | 4032 | 300 | 0 | 0 | 0 | 1 | 10.00 | 0 | 1 | 0 | 0 (0.00) |
| art_daily_nojump | artificialWithAnomaly | 4032 | 300 | 0 | 0 | 0 | 1 | 10.00 | 0 | 1 | 0 | 0 (0.00) |
| art_increase_spike_density | artificialWithAnomaly | 4032 | 300 | 0 | 0 | 0 | 1 | 10.00 | 1 | 0 | 0 | 403 (16.66) |
| art_load_balancer_spikes | artificialWithAnomaly | 4032 | 300 | 0 | 0 | 0 | 1 | 10.00 | 0 | 1 | 0 | 0 (0.00) |
| ec2_cpu_utilization_24ae8d | realAWSCloudwatch | 4032 | 300 | 0 | 0 | 0 | 2 | 9.97 | 0 | 0 | 2 | 0 (0.00) |
| ec2_cpu_utilization_53ea38 | realAWSCloudwatch | 4032 | 300 | 0 | 0 | 0 | 2 | 9.97 | 1 | 1 | 0 | 201 (8.31) |
| ec2_cpu_utilization_5f5533 | realAWSCloudwatch | 4032 | 300 | 0 | 0 | 0 | 2 | 9.97 | 1 | 1 | 0 | 201 (8.31) |
| ec2_cpu_utilization_77c1ca | realAWSCloudwatch | 4032 | 300 | 0 | 0 | 0 | 1 | 10.00 | 1 | 0 | 0 | 403 (16.66) |
| ec2_cpu_utilization_825cc2 | realAWSCloudwatch | 4032 | 300 | 2 | 2 | 0 | 1 | 8.51 | 1 | 0 | 0 | 343 (14.18) |
| ec2_cpu_utilization_ac20cd | realAWSCloudwatch | 4032 | 300 | 2 | 5 | 0 | 1 | 10.00 | 0 | 0 | 1 | 0 (0.00) |
| ec2_cpu_utilization_c6585a | realAWSCloudwatch | 4032 | 300 | 0 | 0 | 0 | 0 | 0.00 | 0 | 0 | 0 | 0 (0.00) |
| ec2_cpu_utilization_fe7f93 | realAWSCloudwatch | 4032 | 300 | 0 | 0 | 0 | 3 | 10.04 | 2 | 1 | 0 | 270 (11.16) |
| ec2_disk_write_bytes_1ef3de | realAWSCloudwatch | 4730 | 300 | 13 | 11 | 11 | 1 | 10.00 | 1 | 1 | 0 | 439 (15.47) |
| ec2_disk_write_bytes_c0d644 | realAWSCloudwatch | 4032 | 300 | 0 | 0 | 0 | 3 | 10.04 | 2 | 1 | 0 | 270 (11.16) |
| ec2_network_in_257a54 | realAWSCloudwatch | 4032 | 300 | 2 | 2 | 0 | 1 | 10.00 | 1 | 0 | 0 | 403 (16.66) |
| ec2_network_in_5abac7 | realAWSCloudwatch | 4730 | 300 | 13 | 12 | 11 | 2 | 10.02 | 1 | 1 | 0 | 237 (8.35) |
| elb_request_count_8c0756 | realAWSCloudwatch | 4032 | 300 | 8 | 8 | 0 | 2 | 9.97 | 1 | 0 | 1 | 201 (8.31) |
| grok_asg_anomaly | realAWSCloudwatch | 4621 | 300 | 0 | 0 | 0 | 3 | 10.06 | 2 | 1 | 1 | 310 (11.18) |
| iio_us-east-1_i-a2eb1cd9_NetworkIn | realAWSCloudwatch | 1243 | 300 | 0 | 0 | 0 | 2 | 10.14 | 2 | 0 | 0 | 126 (16.91) |
| rds_cpu_utilization_cc0c53 | realAWSCloudwatch | 4032 | 300 | 1 | 1 | 0 | 2 | 9.97 | 0 | 1 | 1 | 0 (0.00) |
| rds_cpu_utilization_e47b3b | realAWSCloudwatch | 4032 | 300 | 0 | 0 | 0 | 2 | 9.97 | 1 | 1 | 0 | 201 (8.31) |
| exchange-2_cpc_results | realAdExchange | 1624 | 3600 | 6 | 25 | 1 | 1 | 10.04 | 1 | 0 | 0 | 163 (16.74) |
| exchange-2_cpm_results | realAdExchange | 1624 | 3600 | 6 | 25 | 1 | 2 | 9.98 | 2 | 1 | 0 | 121 (12.42) |
| exchange-3_cpc_results | realAdExchange | 1538 | 3600 | 64 | 109 | 0 | 3 | 9.95 | 2 | 1 | 0 | 102 (11.06) |
| exchange-3_cpm_results | realAdExchange | 1538 | 3600 | 64 | 109 | 0 | 1 | 9.95 | 0 | 1 | 0 | 0 (0.00) |
| exchange-4_cpc_results | realAdExchange | 1643 | 3600 | 4 | 4 | 0 | 3 | 10.04 | 2 | 1 | 0 | 110 (11.17) |
| exchange-4_cpm_results | realAdExchange | 1643 | 3600 | 4 | 4 | 0 | 4 | 9.98 | 2 | 1 | 1 | 82 (8.32) |
| ambient_temperature_system_failure | realKnownCause | 7267 | 3600 | 10 | 621 | 0 | 2 | 9.99 | 1 | 0 | 1 | 363 (8.33) |
| cpu_utilization_asg_misconfiguration | realKnownCause | 18050 | 300 | 0 | 0 | 0 | 1 | 8.30 | 0 | 0 | 1 | 0 (0.00) |
| ec2_request_latency_system_failure | realKnownCause | 4032 | 300 | 14 | 13 | 11 | 3 | 8.58 | 1 | 0 | 2 | 135 (5.58) |
| machine_temperature_system_failure | realKnownCause | 22695 | 300 | 1 | 0 | 12 | 4 | 9.99 | 2 | 1 | 1 | 1134 (8.33) |
| nyc_taxi | realKnownCause | 10320 | 1800 | 0 | 0 | 0 | 5 | 10.03 | 1 | 1 | 3 | 207 (3.34) |
| rogue_agent_key_hold | realKnownCause | 1882 | 300 | 91 | 3456 | 0 | 2 | 10.10 | 1 | 1 | 0 | 95 (8.41) |
| rogue_agent_key_updown | realKnownCause | 5315 | 300 | 10 | 23 | 0 | 2 | 9.97 | 2 | 0 | 0 | 530 (16.62) |
| TravelTime_387 | realTraffic | 2500 | 600 | 909 | 7472 | 0 | 3 | 9.96 | 3 | 1 | 0 | 181 (12.07) |
| TravelTime_451 | realTraffic | 2162 | 600 | 1372 | 5231 | 0 | 1 | 10.04 | 1 | 0 | 0 | 217 (16.73) |
| occupancy_6005 | realTraffic | 2380 | 300 | 625 | 2263 | 0 | 1 | 10.04 | 0 | 1 | 0 | 0 (0.00) |
| occupancy_t4013 | realTraffic | 2500 | 300 | 590 | 2175 | 1 | 2 | 10.00 | 0 | 0 | 2 | 0 (0.00) |
| speed_6005 | realTraffic | 2500 | 300 | 677 | 2376 | 0 | 1 | 9.56 | 0 | 0 | 1 | 0 (0.00) |
| speed_7578 | realTraffic | 1127 | 300 | 391 | 1498 | 0 | 4 | 10.29 | 1 | 1 | 2 | 29 (4.29) |
| speed_t4013 | realTraffic | 2495 | 300 | 591 | 2180 | 1 | 2 | 10.02 | 0 | 0 | 2 | 0 (0.00) |
| Twitter_volume_AAPL | realTweets | 15902 | 300 | 0 | 0 | 0 | 4 | 9.99 | 4 | 0 | 0 | 1588 (16.64) |
| Twitter_volume_AMZN | realTweets | 15831 | 300 | 0 | 0 | 0 | 4 | 9.98 | 2 | 2 | 0 | 790 (8.32) |
| Twitter_volume_CRM | realTweets | 15902 | 300 | 0 | 0 | 0 | 3 | 10.02 | 3 | 0 | 0 | 1593 (16.70) |
| Twitter_volume_CVS | realTweets | 15853 | 300 | 0 | 0 | 0 | 3 | 9.63 | 2 | 0 | 1 | 1129 (11.87) |
| Twitter_volume_FB | realTweets | 15833 | 300 | 0 | 0 | 0 | 2 | 9.99 | 1 | 1 | 0 | 791 (8.33) |
| Twitter_volume_GOOG | realTweets | 15842 | 300 | 0 | 0 | 0 | 3 | 9.04 | 3 | 1 | 0 | 1141 (12.00) |
| Twitter_volume_IBM | realTweets | 15893 | 300 | 0 | 0 | 0 | 2 | 10.00 | 1 | 0 | 1 | 795 (8.34) |
| Twitter_volume_KO | realTweets | 15851 | 300 | 0 | 0 | 0 | 3 | 10.01 | 1 | 1 | 1 | 529 (5.56) |
| Twitter_volume_PFE | realTweets | 15858 | 300 | 0 | 0 | 0 | 4 | 10.01 | 3 | 1 | 0 | 1191 (12.52) |
| Twitter_volume_UPS | realTweets | 15866 | 300 | 0 | 0 | 0 | 5 | 9.99 | 5 | 0 | 0 | 1585 (16.65) |
