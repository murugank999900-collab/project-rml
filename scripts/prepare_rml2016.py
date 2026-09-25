"""Verify RML2016.10a and create or verify the frozen split file.

Run where the dataset is available (e.g. Kaggle). The dataset path comes from
--data or the environment variable named in the config (RML2016_ROOT by default):

    RML2016_ROOT=/path/to/dataset/dir python scripts/prepare_rml2016.py
    python scripts/prepare_rml2016.py --data /path/to/RML2016.10a_dict_optimized.pkl --write-split

Without --write-split the split is generated and checked in memory only. If the
split file already exists it is never overwritten; the regenerated split must
match it exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rml.config import load_config, resolve_dataset_file, resolve_project_path  # noqa: E402
from rml.data import check_disjoint, export_split, load_rml2016, load_split, make_split, validate_structure  # noqa: E402
from rml.evaluation import verify_highest_snr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/rml2016/base.yaml")
    parser.add_argument("--data", help="dataset directory or .pkl file (overrides env/config)")
    parser.add_argument("--split-file", help="override the split file path from the config")
    parser.add_argument("--write-split", action="store_true", help="write the split file if it does not exist")
    args = parser.parse_args()

    cfg = load_config(resolve_project_path(args.config))
    dcfg = cfg["dataset"]
    exp = dcfg["expected"]

    data_file = resolve_dataset_file(dcfg, override=args.data)
    print(f"Loading {data_file}")
    ds = load_rml2016(data_file, sample_shape=exp["sample_shape"])
    validate_structure(ds, exp["num_classes"], exp["snrs"], exp["samples_per_group"])
    print(f"  samples: {len(ds)}  shape: {ds.X.shape}  dtype: {ds.X.dtype}")
    print(f"  classes ({len(ds.classes)}): {list(ds.classes)}")
    print(f"  SNRs ({len(ds.snrs)}): {list(ds.snrs)}")

    split = make_split(
        ds.group_sizes,
        seed=dcfg["split_seed"],
        val_fraction=dcfg["split"]["val_fraction"],
        test_fraction=dcfg["split"]["test_fraction"],
    )
    check_disjoint(split)
    counts = split.counts()
    print(f"  split counts: {counts}")
    if counts != exp["counts"]:
        raise SystemExit(f"Split counts {counts} != expected {exp['counts']}")

    _, y_test, snr_test = ds.take(split.test)
    top = verify_highest_snr(snr_test, exp["highest_snr"])
    n_top = int(np.sum(snr_test == top))
    expected_top = len(ds.classes) * round(exp["samples_per_group"] * dcfg["split"]["test_fraction"])
    per_class_top = np.bincount(y_test[snr_test == top], minlength=len(ds.classes))
    print(f"  highest test SNR: {top:+d} dB with {n_top} samples, per class: {per_class_top.tolist()}")
    if n_top != expected_top:
        raise SystemExit(f"Expected {expected_top} test samples at {top:+d} dB, found {n_top}")

    split_path = resolve_project_path(args.split_file or dcfg["split"]["file"])
    if split_path.exists():
        frozen = load_split(split_path)
        if frozen.group_sizes != split.group_sizes:
            raise SystemExit(f"Frozen split {split_path} was made for different groups than this dataset")
        if not frozen.same_indices(split):
            raise SystemExit(f"Regenerated split differs from frozen split {split_path}; the frozen file wins")
        print(f"  frozen split verified: {split_path}")
    elif args.write_split:
        export_split(split, split_path, dcfg["name"])
        print(f"  split written: {split_path}")
    else:
        print(f"  split file {split_path} not present; rerun with --write-split to create it")

    print(f"  sha256: {split.sha256()}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
