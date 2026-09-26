"""Verify RML2018.01a against the frozen split. Never creates or modifies a split file.

Run where the dataset is available (e.g. Kaggle). The dataset path comes from
--data or the environment variable named in the config (RML2018_ROOT by default):

    RML2018_ROOT=/path/to/dataset/dir python scripts/prepare_rml2018.py
    python scripts/prepare_rml2018.py --data /path/to/GOLD_XYZ_OSC.0001_1024.hdf5
    python scripts/prepare_rml2018.py --split-only     # split file checks, no dataset needed

Requires h5py (pip install -e .[rml2018]) unless --split-only is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rml.config import load_config, resolve_dataset_file, resolve_project_path  # noqa: E402
from rml.data.rml2018 import (  # noqa: E402
    check_split_against_labels,
    load_rml2018,
    load_rml2018_split,
    validate_rml2018_structure,
)
from rml.evaluation import verify_highest_snr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/rml2018/base.yaml")
    parser.add_argument("--data", help="dataset directory or .hdf5 file (overrides env/config)")
    parser.add_argument("--split-only", action="store_true", help="verify the split file without the dataset")
    args = parser.parse_args()

    dcfg = load_config(resolve_project_path(args.config))["dataset"]
    exp = dcfg["expected"]

    split_path = resolve_project_path(dcfg["split"]["file"])
    split = load_rml2018_split(split_path, expected_sha256=dcfg["split"]["sha256"])
    if split.seed != dcfg["split_seed"]:
        raise SystemExit(f"Split file seed {split.seed} != configured split_seed {dcfg['split_seed']}")
    counts = split.counts()
    if counts != exp["counts"]:
        raise SystemExit(f"Split counts {counts} != expected {exp['counts']}")
    print(f"Frozen split verified: {split_path}")
    print(f"  counts: {counts}")
    print(f"  sha256: {split.sha256()}")
    if args.split_only:
        print("OK (split only; dataset labels not checked)")
        return 0

    data_file = resolve_dataset_file(dcfg, override=args.data)
    print(f"Loading {data_file}")
    ds = load_rml2018(data_file)
    if len(ds) != exp["num_samples"]:
        raise SystemExit(f"Dataset has {len(ds)} rows, expected {exp['num_samples']}")
    validate_rml2018_structure(ds, exp["num_classes"], exp["snrs"], exp["samples_per_group"])
    print(f"  samples: {len(ds)}  classes: {len(ds.classes)}  SNRs ({len(ds.snrs)}): {list(ds.snrs)}")

    check_split_against_labels(split, ds.y, ds.snr, exp["per_group"])
    print(f"  per (class, SNR) group: {exp['per_group']}")

    for part in ("val", "test"):
        part_snr = ds.snr[getattr(split, part)]
        top = verify_highest_snr(part_snr, exp["highest_snr"])
        n_top = int(np.sum(part_snr == top))
        print(f"  {part}: {n_top} samples at {top:+d} dB")
        if n_top != exp["highest_snr_counts"][part]:
            raise SystemExit(f"Expected {exp['highest_snr_counts'][part]} {part} samples at {top:+d} dB, found {n_top}")

    X, _, _ = ds.take(np.arange(min(8, len(ds))))
    if X.shape[1:] != tuple(exp["sample_shape"]) or not np.all(np.isfinite(X)):
        raise SystemExit(f"Unexpected sample shape {X.shape[1:]} or non-finite values")
    print(f"  sample shape: {X.shape[1:]}  dtype: {X.dtype}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
