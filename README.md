# Project RML

Automatic Modulation Classification (AMC) research on RML2016.10a and RML2018.01a.

## Primary metric

**Peak accuracy = held-out test accuracy at the highest SNR present in the test data.**
Overall accuracy is reported but is secondary and is not used for experiment selection.
Model selection uses the validation split only; the test split is for final reporting.

| Dataset      | Highest SNR | Target peak accuracy |
|--------------|-------------|----------------------|
| RML2016.10a  | +18 dB      | >= 90%               |
| RML2018.01a  | +30 dB      | >= 94%               |

No target has been achieved yet.

## Layout

```
configs/rml2016/base.yaml   dataset + split + evaluation config (no runtime paths)
src/rml/config.py           config loading, dataset path resolution
src/rml/data/rml2016.py     RML2016.10a pickle loader
src/rml/data/splits.py      deterministic per-(mod, SNR) 80/10/10 split, JSON export
src/rml/evaluation/         overall / per-SNR / peak accuracy, confusion matrix, report
scripts/prepare_rml2016.py  verify dataset structure, create or verify the frozen split
splits/                     frozen split index files (no samples)
tests/                      unit tests on synthetic data
```

## Dataset location

Datasets are never stored in Git. Point the code at the dataset at runtime:

```bash
export RML2016_ROOT=/path/to/dir/containing/pkl      # or
python scripts/prepare_rml2016.py --data /path/to/RML2016.10a_dict_optimized.pkl
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
