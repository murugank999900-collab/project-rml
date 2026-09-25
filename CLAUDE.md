# Project RML rules

- Primary metric is `peak_accuracy_highest_snr` on the held-out test split
  (RML2016.10a: +18 dB, target >= 0.90; RML2018.01a: +30 dB, target >= 0.94).
  Overall accuracy is secondary and never the selection metric.
- Select models on the validation split only. Never tune on the test split.
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
