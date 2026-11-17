# Phase 2: LSTM-AE training run

Run by `scripts/train.py` on 2026-11-18 (sequential, 4 CPU threads,
settings D-11, D-23). Numbers from `results/phase2_errors/run_summary.csv`,
`failures.json` and `logs/phase2_train.log`.

- Fits: 108 of 108 (36 model files x seeds 15, 42, 123); 0 failures.
- Time: sum of per-fit wall times 2,847.3 s (47.5 min); the main run took
  47.3 min for 107 fits after a 1-fit smoke test.
- Epochs run: 6 to 50, median 20; 10 fits reached the 50-epoch cap.
- Best epoch: 1 to 50, median 15.
- Errors: every file has exactly 29 NaN entries (rows before the first full
  window) and all other errors are finite; range 0.000208 to 3.043 (scaled
  MAE).
- Fits with best epoch <= 2: 5. They are kept as trained; the decided
  protocol (D-05, D-23) applies to all fits and none is retrained (D-25).

  | File | Seed | Epochs run | Best epoch |
  |---|---|---|---|
  | realAWSCloudwatch/ec2_cpu_utilization_fe7f93 | 15 | 7 | 2 |
  | realAWSCloudwatch/ec2_network_in_5abac7 | 15 | 7 | 2 |
  | realAWSCloudwatch/ec2_network_in_5abac7 | 123 | 7 | 2 |
  | realAdExchange/exchange-3_cpc_results | 42 | 6 | 1 |
  | realTweets/Twitter_volume_CVS | 123 | 7 | 2 |

Figures (`figures/phase2_model/`, from `scripts/plot_phase2.py`):
- fig01_loss_curves: training and early-stopping MSE per epoch, seed 15, for
  nyc_taxi (best epoch 39 of 44), machine_temperature_system_failure (5 of
  10), Twitter_volume_KO (28 of 33) and speed_7578 (13 of 18).
- fig02_reconstruction_error: nyc_taxi, seed 15, series and per-window MAE
  with NAB windows shaded and the split boundaries dashed.
No threshold has been computed.
