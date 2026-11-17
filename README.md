# Threshold rules for LSTM-autoencoder anomaly detection on NAB

One LSTM autoencoder is trained per series; the compared methods are threshold
rules applied to its reconstruction errors:

- B1: fixed p-th percentile of training errors
- B2: fixed mean + k*sigma of training errors
- B3: rolling mean + 3*sigma of the previous w errors (base paper)
- P1: rolling median + lambda*MAD of the previous w errors (proposed), with
  plain or gated memory

Data: Numenta Anomaly Benchmark (NAB), chronological 60/20/20 split per file,
all parameters tuned on validation only, test scored once, 3 seeds.

Figures are published separately in DSTT-Anomaly-Detection-images.

Work in progress.
