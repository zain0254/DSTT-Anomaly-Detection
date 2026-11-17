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
