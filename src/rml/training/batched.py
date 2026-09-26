"""Mini-batch trainer for datasets too large to transform ahead of time (RML2018.01a).

Train and validation samples stay in host memory in a compact dtype (float16
for RML2018). Each batch is gathered on the CPU, copied to the device and
transformed there (:mod:`rml.training.torch_transforms`). Optimizer, schedule,
validation-only selection and metrics are those of :mod:`rml.training.trainer`.

Every epoch writes ``history.csv`` (one row), ``val_epochs.jsonl`` (full
validation metrics), ``checkpoints/last.pt`` (complete resumable state) and, on
a new best by the validation selection metric, ``checkpoints/best.pt`` and
``val_metrics_best.json``. ``resume=True`` continues from ``last.pt``.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from rml.data.views import Subset
from rml.evaluation import evaluate_predictions
from rml.training.selection import BestTracker
from rml.training.torch_transforms import prepare_inputs_torch
from rml.training.trainer import CHECKPOINT_FORMAT, HISTORY_COLUMNS, _val_scalars, build_optimizer, build_scheduler

Transform = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class BatchedFitResult:
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    best_val_metrics: dict  # full evaluate_predictions() output at the best epoch
    best_checkpoint: Path
    last_checkpoint: Path
    train_seconds: float  # summed over sessions when resumed
    resumed_from_epoch: int | None


def make_transform(data_cfg: Mapping) -> Transform:
    normalize, features = data_cfg["normalize"], data_cfg["features"]
    return lambda xb: prepare_inputs_torch(xb, normalize, features)


@torch.no_grad()
def predict_logits_batched(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    transform: Transform,
    batch_size: int = 2048,
    amp: bool = False,
) -> torch.Tensor:
    """Float32 CPU logits for host-memory samples, transformed per batch on the device."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    X_t = torch.from_numpy(X)
    out = []
    for start in range(0, X_t.shape[0], batch_size):
        xb = transform(X_t[start : start + batch_size].to(device, non_blocking=True))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out.append(model(xb).float().cpu())
    return torch.cat(out)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _truncate_log(path: Path, keep_epochs: int, csv_header: bool) -> None:
    """Drop rows for epochs after ``keep_epochs`` (written after the last checkpoint)."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    head, rows = (lines[:1], lines[1:]) if csv_header else ([], lines)
    epoch_of = (lambda r: int(r.split(",", 1)[0])) if csv_header else (lambda r: int(json.loads(r)["epoch"]))
    path.write_text("".join(head + [r for r in rows if r.strip() and epoch_of(r) <= keep_epochs]), encoding="utf-8")


def fit_batched(
    model: nn.Module,
    train: Subset,
    val: Subset,
    *,
    classes: Sequence[str],
    train_cfg: Mapping,
    selection_cfg: Mapping,
    data_cfg: Mapping,
    expected_highest_snr: int | None,
    run_dir: Path,
    device: torch.device,
    seed: int,
    resume: bool = False,
    limit_train_batches: int | None = None,
    log: Callable[[str], None] = print,
) -> BatchedFitResult:
    if train.name != "train" or val.name != "val":
        raise ValueError(f"fit_batched() takes the train and val subsets, got {train.name!r} and {val.name!r}")
    if train_cfg.get("augmentation", "none") != "none":
        raise ValueError("Augmentation is not implemented yet; set train.augmentation: none")

    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    history_path, epochs_path, best_json = run_dir / "history.csv", run_dir / "val_epochs.jsonl", run_dir / "val_metrics_best.json"

    epochs = int(train_cfg["epochs"])
    if epochs < 1:
        raise ValueError("train.epochs must be >= 1")
    batch_size = min(int(train_cfg["batch_size"]), len(train))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 2048))
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    grad_clip = train_cfg.get("grad_clip")
    high_snr_min = int(selection_cfg.get("high_snr_min", 10))
    transform = make_transform(data_cfg)

    model.to(device)
    X_tr = torch.from_numpy(train.X)  # host memory, compact dtype
    y_tr = torch.from_numpy(train.y)
    y_val_t = torch.from_numpy(val.y)
    n = X_tr.shape[0]
    # Drop the incomplete last batch (reshuffled every epoch) so BatchNorm never sees a batch of 1.
    steps_per_epoch = max(1, n // batch_size)
    if limit_train_batches:
        steps_per_epoch = min(steps_per_epoch, int(limit_train_batches))

    optimizer = build_optimizer(model, train_cfg["optimizer"])
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", {}), steps_per_epoch, epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    tracker = BestTracker(
        selection_cfg["metric"], selection_cfg.get("tie_breaker"), selection_cfg.get("early_stopping_patience")
    )
    shuffle_gen = torch.Generator().manual_seed(seed)

    start_epoch, train_seconds, resumed_from, best_full = 1, 0.0, None, {}
    if resume:
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        if ckpt.get("format") != CHECKPOINT_FORMAT or "tracker" not in ckpt:
            raise ValueError(f"{last_path} is not a resumable checkpoint")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        tracker.best_key = tuple(ckpt["tracker"]["best_key"])
        tracker.best_epoch = int(ckpt["tracker"]["best_epoch"])
        tracker.epochs_since_improvement = int(ckpt["tracker"]["epochs_since_improvement"])
        shuffle_gen.set_state(ckpt["rng"]["shuffle"])
        torch.set_rng_state(ckpt["rng"]["torch_cpu"])
        if device.type == "cuda" and ckpt["rng"].get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(list(ckpt["rng"]["torch_cuda"]))
        resumed_from = int(ckpt["epoch"])
        start_epoch, train_seconds = resumed_from + 1, float(ckpt["train_seconds"])
        _truncate_log(history_path, resumed_from, csv_header=True)
        _truncate_log(epochs_path, resumed_from, csv_header=False)
        best_full = json.loads(best_json.read_text(encoding="utf-8"))
        log(f"Resumed after epoch {resumed_from} (best epoch {tracker.best_epoch})")
        if tracker.should_stop or resumed_from >= epochs:
            log("Nothing left to train")
    else:
        with open(history_path, "x", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(HISTORY_COLUMNS)

    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1) if not tracker.should_stop else ():
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=shuffle_gen)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        seen = 0
        lr = optimizer.param_groups[0]["lr"]
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            xb = transform(X_tr.index_select(0, idx).to(device, non_blocking=True))
            yb = y_tr.index_select(0, idx).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits.float(), yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach() * yb.shape[0]
            correct += (logits.argmax(1) == yb).sum()
            seen += yb.shape[0]
        train_time = time.perf_counter() - t0

        val_logits = predict_logits_batched(model, val.X, device, transform, eval_batch_size, use_amp)
        val_loss = float(criterion(val_logits, y_val_t).item())
        val_pred = val_logits.argmax(1).numpy()
        full = evaluate_predictions(
            val.y, val_pred, val.snr, classes, split="val", expected_highest_snr=expected_highest_snr
        )
        scalars = _val_scalars(val.y, val_pred, val.snr, full, val_loss, high_snr_min)
        is_best = tracker.update(epoch, scalars)
        epoch_seconds = time.perf_counter() - t0
        train_seconds += epoch_seconds

        if is_best:
            best_full = {**full, "epoch": epoch}
            _atomic_save(
                {"format": CHECKPOINT_FORMAT, "epoch": epoch, "model_state": model.state_dict(), "val_metrics": scalars},
                best_path,
            )
            _write_json(best_json, best_full)
        _atomic_save(
            {
                "format": CHECKPOINT_FORMAT,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_metrics": scalars,
                "tracker": {
                    "best_key": list(tracker.best_key),
                    "best_epoch": tracker.best_epoch,
                    "epochs_since_improvement": tracker.epochs_since_improvement,
                },
                "rng": {
                    "shuffle": shuffle_gen.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                },
                "train_seconds": train_seconds,
            },
            last_path,
        )

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": float(loss_sum.item()) / seen,
            "train_accuracy": int(correct.item()) / seen,
            **scalars,
            "is_best": int(is_best),
            "epoch_seconds": round(epoch_seconds, 3),
        }
        with open(history_path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow([row[c] for c in HISTORY_COLUMNS])
        with open(epochs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train_seconds": round(train_time, 3), **row, "val": full}) + "\n")
        log(
            f"epoch {epoch:3d}/{epochs}  loss {row['train_loss']:.4f}  train_acc {row['train_accuracy']:.4f}  "
            f"val_acc {scalars['val_overall_accuracy']:.4f}  "
            f"val_peak@{scalars['val_highest_snr']:+d}dB {scalars['val_peak_accuracy_highest_snr']:.4f}"
            f"{'  *best*' if is_best else ''}  ({epoch_seconds:.1f}s, train {train_time:.1f}s)"
        )
        if tracker.should_stop:
            log(f"Early stopping: no improvement in {tracker.patience} epochs")
            break

    return BatchedFitResult(
        best_epoch=int(tracker.best_epoch or 0),
        epochs_run=epoch,
        stopped_early=tracker.should_stop,
        best_val_metrics={**best_full, "best_epoch": tracker.best_epoch},
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        train_seconds=train_seconds,
        resumed_from_epoch=resumed_from,
    )
