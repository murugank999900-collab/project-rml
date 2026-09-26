"""Train one RML2018.01a experiment on the official train split; select checkpoints on validation.

    RML2018_ROOT=/path/to/dir python scripts/train_rml2018.py --config configs/rml2018/exp001_cldnn.yaml
    python scripts/train_rml2018.py --config ... --data /path/to/GOLD_XYZ_OSC.0001_1024.hdf5 --seed 1
    python scripts/train_rml2018.py --resume experiments/rml2018.01a/<run_id> --data ...
    python scripts/train_rml2018.py --config ... --data ... --smoke     # short throughput check, no registry

Train and validation samples are read from the HDF5 file in one sequential
pass into compact host-memory arrays (float16 by default, ~9.4 GB); batches
are transformed on the GPU. This script never reads the test split. Final test
evaluation is a separate, explicit step:
scripts/evaluate_final_rml2018.py --run <run_dir> --final
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from rml.config import PROJECT_ROOT, load_experiment_config, resolve_dataset_file, resolve_project_path  # noqa: E402
from rml.data import feature_channels  # noqa: E402
from rml.data.rml2018 import (  # noqa: E402
    check_split_against_labels,
    load_rml2018,
    load_rml2018_split,
    make_train_val_compact,
    validate_rml2018_structure,
)
from rml.evaluation import verify_highest_snr  # noqa: E402
from rml.experiment.metadata import environment_info, git_info, sha256_file, write_json  # noqa: E402
from rml.experiment.registry import append_registry_row  # noqa: E402
from rml.experiment.run_dir import create_run_dir, make_run_id  # noqa: E402
from rml.training.seeding import seed_everything  # noqa: E402

DATASET = "RML2018.01a"
MAX_FP16_RELATIVE_RMS_ERROR = 1e-3  # float16 keeps ~11 bits; typical error is ~2e-4
SMOKE_BATCHES = 200


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="experiment config YAML")
    src.add_argument("--resume", metavar="RUN_DIR", help="continue an unfinished run from checkpoints/last.pt")
    p.add_argument("--data", help="dataset directory or .hdf5 file (overrides env/config)")
    p.add_argument("--seed", type=int, help="override experiment.seed")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.add_argument("--device", default=None, help="cuda, cpu, ... (default: cuda if available)")
    p.add_argument("--skip-dataset-hash", action="store_true", help="do not SHA-256 the dataset file")
    p.add_argument("--smoke", action="store_true", help=f"1 epoch of {SMOKE_BATCHES} batches; no registry row")
    args = p.parse_args(argv)
    if args.resume and (args.seed is not None or args.set or args.smoke):
        p.error("--resume uses the run's saved config; --seed/--set/--smoke are not allowed")
    return args


def available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            return next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return None


def main(argv=None) -> Path:
    args = parse_args(argv)
    if args.resume:
        run_dir = resolve_project_path(args.resume)
        if (run_dir / "summary.json").exists():
            raise SystemExit(f"{run_dir} already finished (summary.json exists); start a new run instead")
        cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    else:
        overrides = list(args.set) + ([f"experiment.seed={args.seed}"] if args.seed is not None else [])
        cfg = load_experiment_config(resolve_project_path(args.config), overrides)
        if args.smoke:
            cfg["train"]["epochs"] = 1
    dcfg, exp = cfg["dataset"], cfg["dataset"]["expected"]
    if dcfg["name"] != DATASET:
        raise SystemExit(f"This script trains {DATASET}; config is for {dcfg['name']}")
    seed = cfg["experiment"]["seed"]
    dtype = np.dtype(cfg["data"].get("storage_dtype", "float16"))

    import torch  # deferred so --help works without PyTorch

    from rml.models import build_model
    from rml.training.batched import fit_batched
    from rml.training.trainer import count_parameters

    git = git_info(PROJECT_ROOT)
    if git["dirty"]:
        print(f"WARNING: uncommitted source changes: {git['dirty_files']}")
    if args.resume:
        started = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))["git"]
        if (started["sha"], started["dirty"]) != (git["sha"], git["dirty"]):
            raise SystemExit(f"Run started at commit {started['sha']} (dirty={started['dirty']}); "
                             f"resume from the same clean commit, not {git['sha']} (dirty={git['dirty']})")

    # Frozen split: file and index hashes are checked against the config.
    split_path = resolve_project_path(dcfg["split"]["file"])
    split = load_rml2018_split(split_path, expected_sha256=dcfg["split"]["sha256"])
    if split.seed != dcfg["split_seed"]:
        raise SystemExit(f"Split file seed {split.seed} != configured split_seed {dcfg['split_seed']}")
    split_hashes = split.sha256()

    data_file = resolve_dataset_file(dcfg, override=args.data)
    print(f"Loading labels from {data_file}")
    ds = load_rml2018(data_file)
    if len(ds) != exp["num_samples"]:
        raise SystemExit(f"Dataset has {len(ds)} rows, expected {exp['num_samples']}")
    validate_rml2018_structure(ds, exp["num_classes"], exp["snrs"], exp["samples_per_group"])
    check_split_against_labels(split, ds.y, ds.snr, exp["per_group"], parts=("train", "val"))

    need = (exp["counts"]["train"] + exp["counts"]["val"]) * int(np.prod(exp["sample_shape"])) * dtype.itemsize
    avail = available_memory_bytes()
    print(f"Train+val samples need {need / 1e9:.1f} GB as {dtype.name}; available: "
          f"{'unknown' if avail is None else f'{avail / 1e9:.1f} GB'}")
    if avail is not None and need > 0.85 * avail:
        raise SystemExit("Not enough host memory for the train+val arrays")

    t = time.perf_counter()
    tv, read_stats = make_train_val_compact(ds, split, dtype=dtype, log=print)
    read_stats["seconds"] = round(time.perf_counter() - t, 1)
    print(f"Read train+val samples in {read_stats['seconds']}s: {read_stats}")
    if dtype == np.float16 and not (
        read_stats["all_finite"] and read_stats["relative_rms_rounding_error"] <= MAX_FP16_RELATIVE_RMS_ERROR
    ):
        raise SystemExit(f"float16 storage is lossy for this data: {read_stats}")
    classes = tv.classes
    del ds

    counts = {"train": len(tv.train), "val": len(tv.val)}
    if counts != {k: exp["counts"][k] for k in counts}:
        raise SystemExit(f"Train/val sizes {counts} != expected {exp['counts']}")
    val_top = verify_highest_snr(tv.val.snr, exp["highest_snr"])
    n_val_top = int(np.sum(tv.val.snr == val_top))
    if n_val_top != exp["highest_snr_counts"]["val"]:
        raise SystemExit(f"Expected {exp['highest_snr_counts']['val']} val samples at {val_top:+d} dB, found {n_val_top}")

    seeding = seed_everything(seed, cfg["experiment"].get("deterministic", True))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(
        cfg["model"]["name"],
        in_channels=feature_channels(cfg["data"]["features"]),
        num_classes=len(classes),
        **cfg["model"].get("params", {}),
    )
    n_params = count_parameters(model)

    if not args.resume:
        root = resolve_project_path(cfg["output"]["root"]) / ("_smoke" if args.smoke else "") / dcfg["name"].lower()
        run_dir = create_run_dir(root, make_run_id(cfg["experiment"]["name"], seed, git["sha"]))
        with open(run_dir / "config.resolved.yaml", "x", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    run_id = run_dir.name
    print(f"Run directory: {run_dir}")

    env = {
        "run_id": run_id,
        "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
        "smoke": args.smoke,
        "git": git,
        "environment": environment_info(),
        "seeding": seeding,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset": {
            "name": dcfg["name"],
            "file": str(data_file),
            "file_sha256": None if args.skip_dataset_hash else sha256_file(data_file),
            "classes": list(classes),
            "storage": read_stats,
        },
        "split": {"file": str(split_path), "seed": split.seed, "index_sha256": split_hashes, "counts_used": counts},
        "val_highest_snr": val_top,
        "val_n_highest_snr": n_val_top,
        "model": {"name": cfg["model"]["name"], "parameters": n_params},
    }
    if args.resume:
        n = len(list(run_dir.glob("env.resume*.json"))) + 1
        write_json(run_dir / f"env.resume{n}.json", env)
    else:
        write_json(run_dir / "env.json", env)

    t_fit = time.perf_counter()
    result = fit_batched(
        model,
        tv.train,
        tv.val,
        classes=classes,
        train_cfg=cfg["train"],
        selection_cfg=cfg["selection"],
        data_cfg=cfg["data"],
        expected_highest_snr=exp["highest_snr"],
        run_dir=run_dir,
        device=device,
        seed=seed,
        resume=bool(args.resume),
        limit_train_batches=SMOKE_BATCHES if args.smoke else None,
    )
    session_seconds = time.perf_counter() - t_fit

    best = result.best_val_metrics
    write_json(run_dir / "val_metrics.json", best)
    checksums = {p.name: sha256_file(p) for p in (result.best_checkpoint, result.last_checkpoint)}
    with open(run_dir / "checkpoints" / "SHA256SUMS", "x", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{h}  {name}\n" for name, h in checksums.items())
    tcfg = cfg["train"]
    summary = {
        "run_id": run_id,
        "smoke": args.smoke,
        "dataset": dcfg["name"],
        "experiment": cfg["experiment"]["name"],
        "seed": seed,
        "git_sha": git["sha"],
        "git_dirty": git["dirty"],
        "split_sha256": split_hashes,
        "model": cfg["model"]["name"],
        "model_params": cfg["model"].get("params", {}),
        "parameters": n_params,
        "features": cfg["data"]["features"],
        "normalize": cfg["data"]["normalize"],
        "storage_dtype": dtype.name,
        "optimizer": tcfg["optimizer"],
        "scheduler": tcfg.get("scheduler"),
        "batch_size": tcfg["batch_size"],
        "epochs_configured": tcfg["epochs"],
        "amp": bool(tcfg.get("amp")),
        "augmentation": tcfg.get("augmentation", "none"),
        "label_smoothing": tcfg.get("label_smoothing", 0.0),
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "resumed_from_epoch": result.resumed_from_epoch,
        "train_seconds_total": round(result.train_seconds, 1),
        "session_seconds": round(session_seconds, 1),
        "selection_metric": cfg["selection"]["metric"],
        "val_highest_snr": best["highest_snr"],
        "val_peak_accuracy_highest_snr": best["peak_accuracy_highest_snr"],
        "val_peak_accuracy_highest_snr_ci95": best["peak_accuracy_highest_snr_ci95"],
        "val_overall_accuracy": best["overall_accuracy"],
        "target_peak_accuracy": dcfg["target_peak_accuracy"],
        "best_checkpoint": result.best_checkpoint.relative_to(run_dir).as_posix(),
        "checkpoint_sha256": checksums,
        "test_evaluated": False,
    }
    write_json(run_dir / "summary.json", summary)

    if not args.smoke:
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
                "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix()
                if run_dir.is_relative_to(PROJECT_ROOT)
                else str(run_dir),
            },
        )
    lo, hi = best["peak_accuracy_highest_snr_ci95"]
    print(
        f"Best epoch {result.best_epoch}: val peak @ {best['highest_snr']:+d} dB = "
        f"{best['peak_accuracy_highest_snr']:.4f} (95% CI {lo:.4f}-{hi:.4f}), "
        f"val overall = {best['overall_accuracy']:.4f}; training time {result.train_seconds / 3600:.2f} h"
    )
    print("Test split NOT evaluated. Final evaluation: scripts/evaluate_final_rml2018.py --run <run_dir> --final")
    return run_dir


if __name__ == "__main__":
    main()
