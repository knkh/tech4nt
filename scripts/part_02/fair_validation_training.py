"""Shared, chronological training and evaluation for the TCN/PatchTST comparison.

This module deliberately centralises the experimental protocol.  Changing a
split, scaler, class weight, metric, or purge rule in one trainer but not the
other would make a model comparison ambiguous.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Both trainers import these same values.  `LABEL_HORIZON_BARS` must match the
# horizon used when Y.npy was created.  The supplied feature pipeline uses a
# 12-bar horizon; change this one number if your label-generation rule changes.
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
LABEL_HORIZON_BARS = 12
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 8
SEED = 20260923

LABEL_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}


class TrendDataset(Dataset):
    """A simple wrapper around already-windowed `(sample, time, feature)` data."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 3:
            raise ValueError(f"X must have 3 dimensions (N, T, F), got {x.shape}")
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


@dataclass(frozen=True)
class Split:
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    purge_samples: int


def set_reproducible_seed(seed: int = SEED) -> None:
    """Make reruns more repeatable; GPU kernels can still vary by platform."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(data_dir: str):
    root = Path(data_dir)
    x = np.load(root / "X.npy")
    y = np.load(root / "Y.npy")
    with (root / "meta.json").open() as handle:
        meta = json.load(handle)

    if len(x) != len(y):
        raise ValueError(f"X has {len(x)} samples but Y has {len(y)} labels")
    if x.ndim != 3:
        raise ValueError(f"X must be (N, T, F), got {x.shape}")
    if set(np.unique(y)) - {0, 1, 2}:
        raise ValueError("Y must contain only the labels 0=DOWN, 1=FLAT, 2=UP")
    if "lookback" not in meta:
        raise ValueError("meta.json must contain the window length as 'lookback'")
    return x, y, meta


def make_split(total_samples: int, lookback: int) -> Split:
    """Create identical chronological partitions for both architectures.

    The purge removes samples around each boundary.  It prevents the final
    training windows and their future labels from overlapping the beginning of
    validation/test.  The gap is deliberately conservative: input lookback +
    label horizon.
    """
    purge = int(lookback) + LABEL_HORIZON_BARS
    train_end = int(total_samples * TRAIN_FRACTION)
    validation_end = int(total_samples * (TRAIN_FRACTION + VALIDATION_FRACTION))
    validation_start = train_end + purge
    test_start = validation_end + purge
    if validation_start >= validation_end or test_start >= total_samples:
        raise ValueError(
            "Dataset is too short for the selected 60/20/20 split and purge. "
            "Collect more data or reduce the explicitly documented split settings."
        )
    return Split(train_end, validation_start, validation_end, test_start, purge)


def normalise_from_training_data(x_train, x_validation, x_test):
    """Scale every period using train-only statistics; never fit on future data."""
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (
        (x_train - mean) / std,
        (x_validation - mean) / std,
        (x_test - mean) / std,
        mean.squeeze(),
        std.squeeze(),
    )


def class_weights(y_train: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    weights = len(y_train) / (3 * np.maximum(counts, 1.0))
    return (weights / weights.mean()).astype(np.float32)


def safe_divide(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    prediction = np.argmax(logits, axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual, predicted in zip(y_true, prediction):
        matrix[int(actual), int(predicted)] += 1

    per_class = {}
    f1s, recalls = [], []
    for cls in range(3):
        true_positive = matrix[cls, cls]
        false_positive = matrix[:, cls].sum() - true_positive
        false_negative = matrix[cls, :].sum() - true_positive
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[LABEL_NAMES[cls]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[cls, :].sum()),
        }
        f1s.append(f1)
        recalls.append(recall)

    # This is the single shared checkpoint metric.  It rewards predictions in
    # both trade-relevant directions and cannot be maximised by predicting FLAT.
    directional_f1 = (per_class["UP"]["f1"] + per_class["DOWN"]["f1"]) / 2
    return {
        "accuracy": float((prediction == y_true).mean()),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "directional_f1": directional_f1,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, logits_all, labels_all = [], [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            losses.append(criterion(logits, labels).item())
            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    result = metrics(labels, logits)
    result["loss"] = float(np.mean(losses))
    return result, logits, labels


def print_metrics(title: str, result: dict) -> None:
    print(
        f"{title}: loss={result['loss']:.4f}  accuracy={result['accuracy']:.4f}  "
        f"macro_f1={result['macro_f1']:.4f}  directional_f1={result['directional_f1']:.4f}"
    )
    for name in ("DOWN", "FLAT", "UP"):
        item = result["per_class"][name]
        print(
            f"  {name:<4} precision={item['precision']:.3f} recall={item['recall']:.3f} "
            f"f1={item['f1']:.3f} support={item['support']}"
        )
    print_confusion_matrix(result["confusion_matrix"])


def print_confusion_matrix(matrix: list[list[int]]) -> None:
    """Print a readable three-class matrix; rows are actual labels, columns predicted."""
    print("  Confusion matrix (rows = actual, columns = predicted)")
    print(f"  {'':<13}{'Pred DOWN':>11}{'Pred FLAT':>11}{'Pred UP':>11}")
    for actual_class, name in LABEL_NAMES.items():
        row = matrix[actual_class]
        print(f"  {f'Actual {name}':<13}{row[0]:>11,}{row[1]:>11,}{row[2]:>11,}")


def train_and_test(*, model, model_name: str, data_dir: str, out_dir: str, optimizer_kind: str):
    """Train one architecture under the shared protocol and save comparable artefacts."""
    set_reproducible_seed()
    device = choose_device()
    x, y, meta = load_data(data_dir)
    split = make_split(len(x), int(meta["lookback"]))

    x_train, y_train = x[:split.train_end], y[:split.train_end]
    x_validation = x[split.validation_start:split.validation_end]
    y_validation = y[split.validation_start:split.validation_end]
    x_test, y_test = x[split.test_start:], y[split.test_start:]
    x_train, x_validation, x_test, mean, std = normalise_from_training_data(
        x_train, x_validation, x_test
    )

    print(f"Model: {model_name} | device: {device}")
    print(f"Dataset shape: {x.shape}; labels: {dict(zip(LABEL_NAMES.values(), np.bincount(y, minlength=3)))}")
    print(
        f"Split: train=[0:{split.train_end}), purge={split.purge_samples}, "
        f"validation=[{split.validation_start}:{split.validation_end}), "
        f"test=[{split.test_start}:{len(x)})"
    )

    weights = class_weights(y_train)
    print(f"Train class weights: {dict(zip(LABEL_NAMES.values(), weights.round(4)))}")
    loaders = {
        "train": DataLoader(TrendDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True),
        "train_eval": DataLoader(TrendDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=False),
        "validation": DataLoader(TrendDataset(x_validation, y_validation), batch_size=BATCH_SIZE, shuffle=False),
        "test": DataLoader(TrendDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False),
    }

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    if optimizer_kind == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_kind == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_kind}")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_model.pth"
    best_score, best_epoch, stale_epochs = -1.0, 0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for features, labels in loaders["train"]:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        validation, _, _ = evaluate(model, loaders["validation"], criterion, device)
        print(
            f"Epoch {epoch:02d}: train_loss={np.mean(losses):.4f} "
            f"validation_directional_f1={validation['directional_f1']:.4f}"
        )
        if validation["directional_f1"] > best_score:
            best_score, best_epoch, stale_epochs = validation["directional_f1"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping. Best epoch: {best_epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_result, _, _ = evaluate(model, loaders["train_eval"], criterion, device)
    validation_result, _, _ = evaluate(model, loaders["validation"], criterion, device)
    test_result, test_logits, test_labels = evaluate(model, loaders["test"], criterion, device)
    print_metrics("Training", train_result)
    print_metrics("Validation", validation_result)
    print_metrics("FINAL UNTOUCHED TEST", test_result)

    probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    with (output / "scaler.json").open("w") as handle:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, handle, indent=2)
    with (output / "evaluation.json").open("w") as handle:
        json.dump(
            {
                "model": model_name,
                "best_epoch": best_epoch,
                "best_validation_directional_f1": best_score,
                "protocol": {
                    "train_fraction": TRAIN_FRACTION,
                    "validation_fraction": VALIDATION_FRACTION,
                    "lookback": int(meta["lookback"]),
                    "label_horizon_bars": LABEL_HORIZON_BARS,
                    "purge_samples": split.purge_samples,
                    "seed": SEED,
                    "checkpoint_metric": "directional_f1",
                },
                "split": asdict(split),
                "training": train_result,
                "validation": validation_result,
                "test": test_result,
            },
            handle,
            indent=2,
        )
    np.savez_compressed(
        output / "test_predictions.npz",
        y_true=test_labels,
        y_pred=np.argmax(test_logits, axis=1),
        logits=test_logits,
        probabilities=probabilities,
    )
    print(f"Saved comparable checkpoint and results to: {output}")
