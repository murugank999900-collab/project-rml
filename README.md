# Project RML

Automatic Modulation Classification (AMC) research on RML2016.10a and RML2018.01a.

## Primary metric

**Peak accuracy = held-out test accuracy at the highest SNR present in the test data.**
Overall accuracy is reported but is secondary and is not used for experiment selection.

- Development / checkpoint selection metric: `val_peak_accuracy_highest_snr` (validation split).
- Final research metric: test `peak_accuracy_highest_snr`, computed once per finished run by
  `scripts/evaluate_final.py`. Test results are never used for training, early stopping,
  hyperparameter, architecture or checkpoint selection.

| Dataset      | Highest SNR | Target peak accuracy |
|--------------|-------------|----------------------|
| RML2016.10a  | +18 dB      | >= 90%               |
| RML2018.01a  | +30 dB      | >= 94%               |

No target has been achieved yet.

## Layout

```
configs/rml2016/base.yaml          dataset + split + evaluation config (no runtime paths)
configs/rml2016/exp001_cldnn.yaml  first experiment (cldnn_v1)
src/rml/config.py                  config loading/merging/overrides, dataset path resolution
src/rml/data/                      loader, frozen split, train/val vs test views, input transforms
src/rml/evaluation/                overall / per-SNR / peak accuracy, CI, confusion matrix, report
src/rml/models/                    model registry + architectures (cldnn_v1)
src/rml/training/                  trainer, inference, validation-only selection, seeding
src/rml/experiment/                run directories, metadata/hashes, append-only registry
scripts/prepare_rml2016.py         verify dataset structure, create or verify the frozen split
scripts/train.py                   train one experiment (train + val only)
scripts/evaluate_final.py          one-time held-out test evaluation of a finished run
splits/                            frozen split index files (no samples)
experiments/<dataset>/<run_id>/    one directory per run (checkpoints are git-ignored)
results/registry.csv               append-only log of train / final_test events
tests/                             unit tests on synthetic data
```

## Dataset location

Datasets are never stored in Git. Point the code at the dataset at runtime:

```bash
export RML2016_ROOT=/path/to/dir/containing/pkl      # or pass --data /path/to/file.pkl
```

## Training (GPU, e.g. Kaggle)

```bash
pip install -r requirements-train.txt   # PyTorch is preinstalled on Kaggle/Colab
python scripts/train.py --config configs/rml2016/exp001_cldnn.yaml
python scripts/train.py --config configs/rml2016/exp001_cldnn.yaml --seed 1 --set train.epochs=50
```

Each run creates `experiments/rml2016.10a/<UTC time>_<experiment>_s<seed>_<git sha>/` with
`config.resolved.yaml`, `env.json` (Git SHA, environment/GPU, seeding, dataset and split
hashes), `history.csv`, `val_metrics.json`, `summary.json` and `checkpoints/`
(`best.pt`, `last.pt`, `SHA256SUMS`), and appends a `train` row to `results/registry.csv`.

After all choices for a run are final:

```bash
python scripts/evaluate_final.py --run experiments/rml2016.10a/<run_id> --final
```

## Adding an architecture

Create a module in `src/rml/models/`, decorate the `nn.Module` with
`@register_model("name")`, add the module to `_BUILTIN_MODULES` in
`src/rml/models/registry.py`, and set `model.name` / `model.params` in a new config.
Data loading, splits, training, selection and evaluation are unchanged.

## Tests

```bash
pip install -r requirements-dev.txt
pytest            # model/trainer tests are skipped if PyTorch is not installed
```
