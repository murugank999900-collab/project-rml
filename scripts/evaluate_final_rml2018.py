"""Final, one-time held-out TEST evaluation of a finished RML2018.01a training run.

    python scripts/evaluate_final_rml2018.py --run experiments/rml2018.01a/<run_id> --final --data ...

Only run this after all training, tuning and checkpoint/architecture choices
for the run are finished; those choices must be made on validation metrics.
Writes test_metrics.json (refuses if it already exists) and appends a
``final_test`` event to the registry. Test results must never feed back into
model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from rml.config import PROJECT_ROOT, resolve_dataset_file, resolve_project_path  # noqa: E402
from rml.data import feature_channels  # noqa: E402
from rml.data.rml2018 import load_rml2018, load_rml2018_split, make_test_compact, validate_rml2018_structure  # noqa: E402
from rml.evaluation import evaluate_predictions, format_classification_report, verify_highest_snr  # noqa: E402
from rml.experiment.metadata import environment_info, sha256_file, write_json  # noqa: E402
from rml.experiment.registry import append_registry_row  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory created by scripts/train_rml2018.py")
    p.add_argument("--final", action="store_true", help="confirm this is the final test evaluation")
    p.add_argument("--data", help="dataset directory or .hdf5 file (overrides env/config)")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if not args.final:
        raise SystemExit("Refusing to evaluate the test split without --final.")
    run_dir = resolve_project_path(args.run)
    out_path = run_dir / "test_metrics.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; the test split is evaluated once per run.")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("smoke"):
        raise SystemExit("Smoke runs are not evaluated on the test split.")

    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    dcfg, exp = cfg["dataset"], cfg["dataset"]["expected"]

    best_ckpt = run_dir / "checkpoints" / "best.pt"
    recorded = dict(line.split()[::-1] for line in (run_dir / "checkpoints" / "SHA256SUMS").read_text().splitlines())
    ckpt_sha = sha256_file(best_ckpt)
    if recorded.get("best.pt") != ckpt_sha:
        raise SystemExit("best.pt does not match its recorded SHA-256; refusing to evaluate")

    split = load_rml2018_split(resolve_project_path(dcfg["split"]["file"]), expected_sha256=dcfg["split"]["sha256"])
    if split.sha256() != env["split"]["index_sha256"]:
        raise SystemExit("Split file differs from the one used for training")

    import torch

    from rml.models import build_model
    from rml.training.batched import make_transform, predict_logits_batched
    from rml.training.trainer import load_model_weights

    data_file = resolve_dataset_file(dcfg, override=args.data)
    ds = load_rml2018(data_file)
    validate_rml2018_structure(ds, exp["num_classes"], exp["snrs"], exp["samples_per_group"])
    classes = ds.classes
    if list(classes) != env["dataset"]["classes"]:
        raise SystemExit("Dataset classes differ from those used for training")
    dtype = np.dtype(cfg["data"].get("storage_dtype", "float16"))
    test, _ = make_test_compact(ds, split, dtype=dtype)
    del ds

    top = verify_highest_snr(test.snr, exp["highest_snr"])
    n_top = int(np.sum(test.snr == top))
    if len(test) != exp["counts"]["test"] or n_top != exp["highest_snr_counts"]["test"]:
        raise SystemExit(f"Unexpected test set: {len(test)} samples, {n_top} at {top:+d} dB")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(
        cfg["model"]["name"],
        in_channels=feature_channels(cfg["data"]["features"]),
        num_classes=len(classes),
        **cfg["model"].get("params", {}),
    ).to(device)
    ckpt = load_model_weights(model, best_ckpt, device)
    logits = predict_logits_batched(
        model,
        test.X,
        device,
        make_transform(cfg["data"]),
        int(cfg["train"].get("eval_batch_size", 2048)),
        bool(cfg["train"].get("amp")),
    )
    y_pred = logits.argmax(1).numpy()

    result = evaluate_predictions(
        test.y,
        y_pred,
        test.snr,
        classes,
        split="test",
        expected_highest_snr=exp["highest_snr"],
        target_peak_accuracy=dcfg.get("target_peak_accuracy"),
    )
    result.update(
        {
            "run_id": env["run_id"],
            "checkpoint": "best.pt",
            "checkpoint_epoch": ckpt["epoch"],
            "checkpoint_sha256": ckpt_sha,
            "evaluated_utc": environment_info()["timestamp_utc"],
        }
    )
    write_json(out_path, result)

    append_registry_row(
        resolve_project_path(cfg["output"]["registry"]),
        {
            "event": "final_test",
            "timestamp_utc": result["evaluated_utc"],
            "run_id": env["run_id"],
            "dataset": dcfg["name"],
            "experiment": cfg["experiment"]["name"],
            "model": cfg["model"]["name"],
            "seed": cfg["experiment"]["seed"],
            "git_sha": env["git"]["sha"],
            "git_dirty": env["git"]["dirty"],
            "split_sha256_train": env["split"]["index_sha256"]["train"],
            "split_sha256_val": env["split"]["index_sha256"]["val"],
            "split_sha256_test": env["split"]["index_sha256"]["test"],
            "best_epoch": ckpt["epoch"],
            "test_highest_snr": result["highest_snr"],
            "test_peak_accuracy_highest_snr": result["peak_accuracy_highest_snr"],
            "test_overall_accuracy": result["overall_accuracy"],
            "checkpoint_sha256": ckpt_sha,
            "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix() if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
        },
    )

    lo, hi = result["peak_accuracy_highest_snr_ci95"]
    print(format_classification_report(result["classification_report_highest_snr"]))
    print(
        f"TEST peak accuracy @ {top:+d} dB: {result['peak_accuracy_highest_snr']:.4f} "
        f"(95% CI {lo:.4f}-{hi:.4f}, n={n_top}); overall: {result['overall_accuracy']:.4f}"
    )
    if "meets_target" in result:
        print(f"Target {result['target_peak_accuracy']:.2f} at {top:+d} dB met by this run: {result['meets_target']}")
    return result


if __name__ == "__main__":
    main()
