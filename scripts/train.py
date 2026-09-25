"""Train one experiment on the official train split; select checkpoints on validation.

    RML2016_ROOT=/path/to/dataset/dir python scripts/train.py --config configs/rml2016/exp001_cldnn.yaml
    python scripts/train.py --config ... --data /path/to/file.pkl --seed 1 --set train.epochs=5

This script never reads the test split. Final test evaluation is a separate,
explicit step: scripts/evaluate_final.py --run <run_dir> --final
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from rml.config import PROJECT_ROOT, load_experiment_config, resolve_dataset_file, resolve_project_path  # noqa: E402
from rml.data import feature_channels, load_rml2016, load_split, make_split, make_train_val, prepare_inputs  # noqa: E402
from rml.data import validate_structure  # noqa: E402
from rml.evaluation import verify_highest_snr  # noqa: E402
from rml.experiment.metadata import environment_info, git_info, sha256_file, write_json  # noqa: E402
from rml.experiment.registry import append_registry_row  # noqa: E402
from rml.experiment.run_dir import create_run_dir, make_run_id  # noqa: E402
from rml.training.seeding import seed_everything  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment config YAML")
    p.add_argument("--data", help="dataset directory or file (overrides env/config)")
    p.add_argument("--seed", type=int, help="override experiment.seed")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.add_argument("--device", default=None, help="cuda, cpu, ... (default: cuda if available)")
    p.add_argument("--skip-dataset-hash", action="store_true", help="do not SHA-256 the dataset file")
    return p.parse_args(argv)


def load_frozen_split(dcfg: dict):
    """Load the official split file and check it is the configured, unmodified split."""
    split_path = resolve_project_path(dcfg["split"]["file"])
    split = load_split(split_path)  # verifies recorded counts and index hashes
    if split.seed != dcfg["split_seed"]:
        raise SystemExit(f"Split file seed {split.seed} != configured split_seed {dcfg['split_seed']}")
    regenerated = make_split(split.group_sizes, dcfg["split_seed"], split.val_fraction, split.test_fraction)
    if not regenerated.same_indices(split):
        raise SystemExit("Split file does not match the split algorithm; refusing to train")
    return split, split_path


def main(argv=None) -> Path:
    args = parse_args(argv)
    overrides = list(args.set) + ([f"experiment.seed={args.seed}"] if args.seed is not None else [])
    cfg = load_experiment_config(resolve_project_path(args.config), overrides)
    dcfg, exp = cfg["dataset"], cfg["dataset"]["expected"]
    seed = cfg["experiment"]["seed"]

    import torch  # deferred so --help works without PyTorch

    from rml.models import build_model
    from rml.training.trainer import count_parameters, fit

    # Record Git state before anything is written into the repository.
    git = git_info(PROJECT_ROOT)
    if git["dirty"]:
        print(f"WARNING: uncommitted source changes: {git['dirty_files']}")

    split, split_path = load_frozen_split(dcfg)
    data_file = resolve_dataset_file(dcfg, override=args.data)
    print(f"Loading {data_file}")
    ds = load_rml2016(data_file, sample_shape=exp["sample_shape"])
    validate_structure(ds, exp["num_classes"], exp["snrs"], exp["samples_per_group"])
    tv = make_train_val(ds, split)
    classes = tv.classes
    del ds  # only train/val copies are kept

    counts = {"train": len(tv.train), "val": len(tv.val)}
    expected_counts = {k: exp["counts"][k] for k in counts}
    if counts != expected_counts:
        raise SystemExit(f"Train/val sizes {counts} != expected {expected_counts}")
    val_top = verify_highest_snr(tv.val.snr, exp["highest_snr"])

    seeding = seed_everything(seed, cfg["experiment"].get("deterministic", True))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dcfg_data = cfg["data"]
    train = tv.train.with_inputs(prepare_inputs(tv.train.X, dcfg_data["normalize"], dcfg_data["features"]))
    val = tv.val.with_inputs(prepare_inputs(tv.val.X, dcfg_data["normalize"], dcfg_data["features"]))
    del tv

    model = build_model(
        cfg["model"]["name"],
        in_channels=feature_channels(dcfg_data["features"]),
        num_classes=len(classes),
        **cfg["model"].get("params", {}),
    )

    run_root = resolve_project_path(cfg["output"]["root"]) / dcfg["name"].lower()
    run_dir = create_run_dir(run_root, make_run_id(cfg["experiment"]["name"], seed, git["sha"]))
    run_id = run_dir.name
    print(f"Run directory: {run_dir}")

    with open(run_dir / "config.resolved.yaml", "x", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    split_hashes = split.sha256()
    write_json(
        run_dir / "env.json",
        {
            "run_id": run_id,
            "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
            "git": git,
            "environment": environment_info(),
            "seeding": seeding,
            "device": str(device),
            "dataset": {
                "name": dcfg["name"],
                "file": str(data_file),
                "file_sha256": None if args.skip_dataset_hash else sha256_file(data_file),
                "classes": list(classes),
            },
            "split": {
                "file": str(split_path),
                "seed": split.seed,
                "index_sha256": split_hashes,
                "counts_used": counts,
            },
            "val_highest_snr": val_top,
            "model": {"name": cfg["model"]["name"], "parameters": count_parameters(model)},
        },
    )

    result = fit(
        model,
        train,
        val,
        classes=classes,
        train_cfg=cfg["train"],
        selection_cfg=cfg["selection"],
        expected_highest_snr=exp["highest_snr"],
        run_dir=run_dir,
        device=device,
        seed=seed,
    )

    write_json(run_dir / "val_metrics.json", result.best_val_metrics)
    checksums = {p.name: sha256_file(p) for p in (result.best_checkpoint, result.last_checkpoint)}
    with open(run_dir / "checkpoints" / "SHA256SUMS", "x", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{h}  {name}\n" for name, h in checksums.items())
    best = result.best_val_metrics
    summary = {
        "run_id": run_id,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "selection_metric": cfg["selection"]["metric"],
        "val_highest_snr": best["highest_snr"],
        "val_peak_accuracy_highest_snr": best["peak_accuracy_highest_snr"],
        "val_peak_accuracy_highest_snr_ci95": best["peak_accuracy_highest_snr_ci95"],
        "val_overall_accuracy": best["overall_accuracy"],
        "checkpoint_sha256": checksums,
        "test_evaluated": False,
    }
    write_json(run_dir / "summary.json", summary)

    append_registry_row(
        resolve_project_path(cfg["output"]["registry"]),
        {
            "event": "train",
            "timestamp_utc": environment_info()["timestamp_utc"],
            "run_id": run_id,
            "dataset": dcfg["name"],
            "experiment": cfg["experiment"]["name"],
            "model": cfg["model"]["name"],
            "seed": seed,
            "git_sha": git["sha"],
            "git_dirty": git["dirty"],
            "split_sha256_train": split_hashes["train"],
            "split_sha256_val": split_hashes["val"],
            "split_sha256_test": split_hashes["test"],
            "epochs_run": result.epochs_run,
            "best_epoch": result.best_epoch,
            "val_highest_snr": best["highest_snr"],
            "val_peak_accuracy_highest_snr": best["peak_accuracy_highest_snr"],
            "val_overall_accuracy": best["overall_accuracy"],
            "checkpoint_sha256": checksums["best.pt"],
            "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix() if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
        },
    )
    lo, hi = best["peak_accuracy_highest_snr_ci95"]
    print(
        f"Best epoch {result.best_epoch}: val peak @ {best['highest_snr']:+d} dB = "
        f"{best['peak_accuracy_highest_snr']:.4f} (95% CI {lo:.4f}-{hi:.4f}), "
        f"val overall = {best['overall_accuracy']:.4f}"
    )
    print("Test split NOT evaluated. Final evaluation: scripts/evaluate_final.py --run <run_dir> --final")
    return run_dir


if __name__ == "__main__":
    main()
