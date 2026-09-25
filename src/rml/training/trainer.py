"""Architecture-agnostic training loop with validation-only checkpoint selection.

``fit`` receives the training and validation subsets only. Each epoch it
evaluates the validation split, tracks the best epoch by the configured
validation metric (``val_peak_accuracy_highest_snr`` by default), saves
``best.pt`` / ``last.pt`` and appends a row to ``history.csv``.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from rml.data.views import Subset
from rml.evaluation import accuracy_at_or_above, evaluate_predictions
from rml.training.inference import predict_logits
from rml.training.selection import BestTracker

CHECKPOINT_FORMAT = "rml-checkpoint/v1"

HISTORY_COLUMNS = (
    "epoch",
    "lr",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_overall_accuracy",
    "val_peak_accuracy_highest_snr",
    "val_highest_snr",
    "val_accuracy_high_snr",
    "is_best",
    "epoch_seconds",
)


@dataclass
class FitResult:
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    best_val_metrics: dict  # full evaluate_predictions() output at the best epoch
    best_checkpoint: Path
    last_checkpoint: Path


def build_optimizer(model: nn.Module, cfg: Mapping) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw")
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"Unknown optimizer {name!r}")


def build_scheduler(optimizer, cfg: Mapping, steps_per_epoch: int, epochs: int):
    """Per-step LR schedule: linear warmup then cosine decay to ``min_lr_ratio``, or constant."""
    name = cfg.get("name", "cosine")
    warmup = int(cfg.get("warmup_epochs", 0) * steps_per_epoch)
    total = max(1, epochs * steps_per_epoch)
    min_ratio = float(cfg.get("min_lr_ratio", 0.0))

    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    if name != "cosine":
        raise ValueError(f"Unknown scheduler {name!r}")

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _val_scalars(y_true, y_pred, snr, full: Mapping, val_loss: float, high_snr_min: int) -> dict[str, float]:
    return {
        "val_loss": val_loss,
        "val_overall_accuracy": full["overall_accuracy"],
        "val_peak_accuracy_highest_snr": full["peak_accuracy_highest_snr"],
        "val_highest_snr": full["highest_snr"],
        "val_accuracy_high_snr": accuracy_at_or_above(y_true, y_pred, snr, high_snr_min),
    }


def fit(
    model: nn.Module,
    train: Subset,
    val: Subset,
    *,
    classes: Sequence[str],
    train_cfg: Mapping,
    selection_cfg: Mapping,
    expected_highest_snr: int | None,
    run_dir: Path,
    device: torch.device,
    seed: int,
    log: Callable[[str], None] = print,
) -> FitResult:
    if train.name != "train" or val.name != "val":
        raise ValueError(f"fit() takes the train and val subsets, got {train.name!r} and {val.name!r}")
    if train_cfg.get("augmentation", "none") != "none":
        raise ValueError("Augmentation is not implemented yet; set train.augmentation: none")

    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"

    epochs = int(train_cfg["epochs"])
    if epochs < 1:
        raise ValueError("train.epochs must be >= 1")
    batch_size = int(train_cfg["batch_size"])
    eval_batch_size = int(train_cfg.get("eval_batch_size", 1024))
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    grad_clip = train_cfg.get("grad_clip")

    model.to(device)
    X_tr = torch.from_numpy(train.X).to(device)
    y_tr = torch.from_numpy(train.y).to(device)
    y_val_t = torch.from_numpy(val.y)
    n = X_tr.shape[0]
    # Drop the incomplete last batch (reshuffled every epoch) so BatchNorm never sees a batch of 1.
    steps_per_epoch = max(1, n // batch_size)
    batch_size = min(batch_size, n)

    optimizer = build_optimizer(model, train_cfg["optimizer"])
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", {}), steps_per_epoch, epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    tracker = BestTracker(
        selection_cfg["metric"],
        selection_cfg.get("tie_breaker"),
        selection_cfg.get("early_stopping_patience"),
    )
    shuffle_gen = torch.Generator().manual_seed(seed)

    history_path = run_dir / "history.csv"
    with open(history_path, "x", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(HISTORY_COLUMNS)

    best_full: dict = {}
    epoch = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=shuffle_gen).to(device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        seen = 0
        lr = optimizer.param_groups[0]["lr"]
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
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

        val_logits = predict_logits(model, val.X, device, eval_batch_size, use_amp)
        val_loss = float(criterion(val_logits, y_val_t).item())
        val_pred = val_logits.argmax(1).numpy()
        full = evaluate_predictions(
            val.y, val_pred, val.snr, classes, split="val", expected_highest_snr=expected_highest_snr
        )
        scalars = _val_scalars(val.y, val_pred, val.snr, full, val_loss, int(selection_cfg.get("high_snr_min", 10)))
        is_best = tracker.update(epoch, scalars)
        if is_best:
            best_full = full
            torch.save(
                {"format": CHECKPOINT_FORMAT, "epoch": epoch, "model_state": model.state_dict(), "val_metrics": scalars},
                best_path,
            )
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_metrics": scalars,
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
            "epoch_seconds": round(time.perf_counter() - t0, 3),
        }
        with open(history_path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow([row[c] for c in HISTORY_COLUMNS])
        log(
            f"epoch {epoch:3d}/{epochs}  loss {row['train_loss']:.4f}  train_acc {row['train_accuracy']:.4f}  "
            f"val_acc {scalars['val_overall_accuracy']:.4f}  "
            f"val_peak@{scalars['val_highest_snr']:+d}dB {scalars['val_peak_accuracy_highest_snr']:.4f}"
            f"{'  *best*' if is_best else ''}  ({row['epoch_seconds']:.1f}s)"
        )
        if tracker.should_stop:
            log(f"Early stopping: no improvement in {tracker.patience} epochs")
            break

    return FitResult(
        best_epoch=int(tracker.best_epoch or 0),  # epochs >= 1, so always set
        epochs_run=epoch,
        stopped_early=tracker.should_stop,
        best_val_metrics={**best_full, "best_epoch": tracker.best_epoch},
        best_checkpoint=best_path,
        last_checkpoint=last_path,
    )


def load_model_weights(model: nn.Module, checkpoint: Path, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint}")
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


__all__ = ["FitResult", "HISTORY_COLUMNS", "count_parameters", "fit", "load_model_weights"]
