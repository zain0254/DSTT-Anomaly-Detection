# Run times

Wall-clock times printed by the scripts on the machine in
[hardware.md](hardware.md). Training time per fit is also saved in
`results/phase2_errors/*.json` (`wall_seconds`).

| Step | Script | Time |
|---|---|---|
| LSTM-AE training, 36 series x 3 seeds | `scripts/train.py` | 47.5 min (sum of `wall_seconds`) |
| Threshold tuning on validation, 4 worker processes | `scripts/phase3.py` | 135 s |
| Single test scoring | `scripts/phase3.py` | 6 s |
| Exploratory grid extension | `scripts/phase3.py --grids exploratory` | 4.0 min |
