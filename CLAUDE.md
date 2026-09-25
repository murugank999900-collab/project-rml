# Project RML rules

- Primary metric is `peak_accuracy_highest_snr` on the held-out test split
  (RML2016.10a: +18 dB, target >= 0.90; RML2018.01a: +30 dB, target >= 0.94).
  Overall accuracy is secondary and never the selection metric.
- Select models on the validation split only (`val_peak_accuracy_highest_snr`). Never use test
  results for training, early stopping, hyperparameter, architecture or checkpoint selection.
- `scripts/train.py` and `src/rml/training/` must never read the test split
  (enforced by `tests/test_isolation.py`). Only `scripts/evaluate_final.py --final` evaluates test.
- Train on the complete official training split (`data.train_snrs: all`).
- Never modify a split file in `splits/` once created. `export_split` refuses to overwrite.
- Never commit datasets, checkpoints or other large binaries. Dataset paths come from
  config/env (`RML2016_ROOT`), never hard-coded in `src/`.
- `splits/rml2016.10a_seed42.json` (split algorithm `per-group-permutation/v1`, seed 42) is the
  official RML2016.10a split. Results on any earlier split are not comparable.
- Preserve every experiment. Never overwrite the best model.
- Do not download datasets, start training, push to GitHub, or use paid services/API keys
  without explicit approval.
- Never claim a target is met without a recorded test-split evaluation result.
- Run `pytest` before proposing a commit.
