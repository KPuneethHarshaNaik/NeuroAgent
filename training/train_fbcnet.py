from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.fbcnet import FBCNet


TASKS = {
    "all_actions": ({index: index for index in range(8)}, {0: "actual_left_hand", 1: "actual_right_hand", 2: "imagined_left_hand", 3: "imagined_right_hand", 4: "actual_both_fists", 5: "actual_both_feet", 6: "imagined_both_fists", 7: "imagined_both_feet"}),
    "movement_type": ({0: 0, 1: 0, 4: 0, 5: 0, 2: 1, 3: 1, 6: 1, 7: 1}, {0: "actual", 1: "imagined"}),
    "effector": ({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}, {0: "hands", 1: "feet"}),
    "hand_side": ({0: 0, 2: 0, 1: 1, 3: 1}, {0: "left_hand", 1: "right_hand"}),
}


class EpochDataset(Dataset):
    def __init__(self, epochs: np.ndarray, labels: np.ndarray, indices: list[int], augment: bool = False, noise_std: float = 0.1) -> None:
        self.epochs, self.labels, self.indices = epochs, labels, indices
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        epoch_index = self.indices[index]
        data = torch.from_numpy(self.epochs[epoch_index]).unsqueeze(0)
        if self.augment:
            data = data + torch.randn_like(data) * self.noise_std
        return data, torch.tensor(self.labels[epoch_index], dtype=torch.long)


def indices_by_split(manifest_path: Path, splits_path: Path) -> dict[str, list[int]]:
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    subject_split = {subject: split for split, subjects in splits.items() for subject in subjects}
    indices = {split: [] for split in splits}
    with manifest_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            indices[subject_split[row["subject_id"]]].append(int(row["epoch_index"]))
    return indices


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, classes: int, loss_function: nn.Module) -> tuple[float, float, np.ndarray]:
    model.eval()
    predictions, targets, total_loss = [], [], 0.0
    for inputs, labels in loader:
        outputs = model(inputs.to(device))
        total_loss += loss_function(outputs, labels.to(device)).item() * len(labels)
        predictions.extend(outputs.argmax(1).cpu().tolist())
        targets.extend(labels.tolist())
    matrix = np.zeros((classes, classes), dtype=int)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target, prediction] += 1
    accuracy = float(matrix.trace() / matrix.sum())
    return accuracy, total_loss / len(loader.dataset), matrix


@torch.no_grad()
def accuracy_by_source_label(model: nn.Module, loader: DataLoader, device: torch.device, targets: np.ndarray, source_labels: np.ndarray, source_names: dict[int, str]) -> dict[str, dict[str, float | int]]:
    """Test accuracy inside each label of the *source* labelling a task was projected from.

    ``hand_side`` folds imagined and executed trials into one target, but the deployed classifier
    only ever sees imagined recordings. Splitting the held-out accuracy by source label keeps the
    headline number honest instead of letting executed trials inflate it. ``targets`` and
    ``source_labels`` must be aligned with the loader's order, which holds because evaluation
    loaders never shuffle.
    """
    model.eval()
    predictions: list[int] = []
    for inputs, _ in loader:
        predictions.extend(model(inputs.to(device)).argmax(1).cpu().tolist())
    correct = np.asarray(predictions) == targets
    grouped: dict[str, dict[str, float | int]] = {}
    for value, name in sorted(source_names.items()):
        members = source_labels == value
        if not members.any():
            continue
        grouped[name] = {"n": int(members.sum()), "accuracy": round(float(correct[members].mean()), 4)}
    return grouped


def labels_for(data_dir: Path) -> dict[int, str]:
    path = data_dir / "labels.json"
    return {int(index): label for index, label in json.loads(path.read_text(encoding="utf-8")).items()} if path.exists() else {0: "left_hand", 1: "right_hand"}


def task_mapping(labels: np.ndarray, task: str, default_labels: dict[int, str]) -> tuple[dict[int, int], dict[int, str], np.ndarray]:
    """Resolve a task into its label map, class names and the epochs it selects.

    Shared by ``task_data`` and the per-source-label breakdown in the training report, so the
    two can never disagree about which epochs a task covers.
    """
    label_map, class_labels = TASKS.get(task, ({}, {}))
    if not label_map:
        raise ValueError(f"Unknown task {task!r}; choose from {', '.join(TASKS)}")
    if task == "all_actions" and len(np.unique(labels)) == 2:
        class_labels = default_labels
        label_map = {index: index for index in class_labels}
    return label_map, class_labels, np.asarray([label in label_map for label in labels])


def task_data(epochs: np.ndarray, labels: np.ndarray, source_splits: dict[str, list[int]], task: str, default_labels: dict[int, str]) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], dict[int, str]]:
    label_map, class_labels, selected = task_mapping(labels, task, default_labels)
    old_to_new = np.full(len(labels), -1, dtype=int)
    old_to_new[selected] = np.arange(selected.sum())
    splits = {split: old_to_new[np.asarray(indices)][old_to_new[np.asarray(indices)] >= 0].tolist() for split, indices in source_splits.items()}
    return epochs[selected], np.asarray([label_map[int(label)] for label in labels[selected]], dtype=np.int64), splits, class_labels


def normalize_epochs(epochs: np.ndarray, train_indices: list[int], mode: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Normalize without using labels or statistics from held-out subjects."""
    normalized = epochs.astype(np.float32, copy=True)
    if mode == "epoch_channel":
        mean = normalized.mean(axis=2, keepdims=True)
        std = normalized.std(axis=2, keepdims=True).clip(1e-6)
        return (normalized - mean) / std, None, None
    if mode == "global_channel":
        mean = normalized[train_indices].mean(axis=(0, 2), keepdims=True)
        std = normalized[train_indices].std(axis=(0, 2), keepdims=True).clip(1e-6)
        return (normalized - mean) / std, mean, std
    raise ValueError(f"Unknown normalization {mode!r}")


def train(data_dir: Path, model_path: Path, epochs_to_train: int = 100, batch_size: int = 64, seed: int = 7, learning_rate: float = 1e-3, weight_decay: float = 1e-4, bands: int = 4, spatial_filters: int = 8, task: str = "all_actions", normalization: str = "global_channel") -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for FBCNet training. Install a CUDA-enabled PyTorch build and verify it with `python -c \"import torch; print(torch.cuda.is_available())\"`.")
    data = np.load(data_dir / "epochs.npz")
    epochs, labels = data["epochs"], data["labels"]
    source_splits = indices_by_split(data_dir / "manifest.csv", data_dir / "subject_splits.json")
    dataset_labels = labels_for(data_dir)
    epoch_source_labels = labels.copy()  # What the dataset itself calls each epoch, before the task projects it.
    epochs, labels, splits, class_labels = task_data(epochs, labels, source_splits, task, dataset_labels)
    classes = len(class_labels)
    if set(np.unique(labels)) != set(class_labels):
        raise ValueError("labels.json does not describe every target in epochs.npz")
    epochs, mean, std = normalize_epochs(epochs, splits["train"], normalization)

    device = torch.device("cuda")
    loaders = {split: DataLoader(EpochDataset(epochs, labels, indexes, augment=(split == "train"), noise_std=0.1), batch_size=batch_size, shuffle=split == "train", pin_memory=True) for split, indexes in splits.items()}
    model = FBCNet(channels=epochs.shape[1], samples=epochs.shape[2], classes=classes, bands=bands, spatial_filters=spatial_filters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_train, eta_min=1e-5)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_accuracy, best_state, stale_epochs = -1.0, None, 0
    history = []
    device_details = {"name": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda, "memory_gb": round(torch.cuda.get_device_properties(device).total_memory / 1024 ** 3, 2)}
    split_class_counts = {split: np.bincount(labels[indexes], minlength=classes).tolist() for split, indexes in splits.items()}
    print(json.dumps({"device": device_details, "parameters": sum(parameter.numel() for parameter in model.parameters()), "split_class_counts": split_class_counts}), flush=True)
    for epoch in range(1, epochs_to_train + 1):
        model.train()
        train_loss, train_correct, train_examples = 0.0, 0, 0
        for inputs, targets in loaders["train"]:
            optimizer.zero_grad()
            outputs = model(inputs.to(device, non_blocking=True))
            targets = targets.to(device, non_blocking=True)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_examples += len(targets)
        scheduler.step()
        validation_accuracy, validation_loss, _ = evaluate(model, loaders["validation"], device, classes, loss_function)
        metrics = {"epoch": epoch, "train_loss": train_loss / train_examples, "train_accuracy": train_correct / train_examples, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "lr": optimizer.param_groups[0]["lr"]}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if validation_accuracy > best_accuracy:
            best_accuracy, best_state, stale_epochs = validation_accuracy, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, 0
        else:
            stale_epochs += 1
            if stale_epochs == 15:
                break
    model.load_state_dict(best_state)
    test_accuracy, test_loss, matrix = evaluate(model, loaders["test"], device, classes, loss_function)
    by_source_label = accuracy_by_source_label(model, loaders["test"], device, labels[splits["test"]], epoch_source_labels[splits["test"]], dataset_labels)
    model_path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "channels": epochs.shape[1], "samples": epochs.shape[2], "normalization": normalization, "normalization_mean": mean, "normalization_std": std, "labels": class_labels, "architecture": {"bands": bands, "spatial_filters": spatial_filters}}, model_path)
    preprocessing_path = data_dir / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8")) if preprocessing_path.exists() else {"bandpass_hz": [8, 30], "epoch_seconds": [0.5, 3.5], "reference": "average"}
    report = {"model": model_path.stem, "task": task, "device": device_details, "parameters": sum(parameter.numel() for parameter in model.parameters()), "best_validation_accuracy": best_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy, "test_confusion_matrix": matrix.tolist(),        "test_per_class_accuracy": {class_labels[index]: matrix[index, index] / matrix[index].sum() for index in range(classes)},
        "test_accuracy_by_source_label": by_source_label, "labels": class_labels, "hyperparameters": {"epochs": epochs_to_train, "batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay, "bands": bands, "spatial_filters": spatial_filters, "seed": seed, "normalization": normalization}, "split_epoch_counts": {split: len(indexes) for split, indexes in splits.items()}, "split_class_counts": split_class_counts, "training_history": history, "preprocessing": preprocessing}
    model_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FBCNet on prepared NeuroAgent epochs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/motor_imagery"))
    parser.add_argument("--model-path", type=Path, default=Path("models/fbcnet_v1.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--spatial-filters", type=int, default=8)
    parser.add_argument("--task", choices=TASKS, default="all_actions")
    parser.add_argument("--normalization", choices=("global_channel", "epoch_channel"), default="global_channel")
    args = parser.parse_args()
    report = train(args.data_dir, args.model_path, args.epochs, args.batch_size, args.seed, args.learning_rate, args.weight_decay, args.bands, args.spatial_filters, args.task, args.normalization)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
