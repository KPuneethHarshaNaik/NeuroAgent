"""Train the classical CSP + LDA baseline on the prepared motor-imagery epochs.

Uses the exact subject-wise split and 8-30 Hz, 0.5-3.5 s epoch contract already
used by FBCNet. This is a comparator, not a replacement for the deployed model.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from training.train_fbcnet import TASKS, indices_by_split, labels_for, task_data


def metrics(model: object, epochs: np.ndarray, labels: np.ndarray) -> dict:
    prediction = model.predict(epochs)
    return {
        "accuracy": round(float(accuracy_score(labels, prediction)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, prediction)), 4),
        "f1": round(float(f1_score(labels, prediction, average="macro")), 4),
        "confusion_matrix": confusion_matrix(labels, prediction).tolist(),
        "epochs": int(len(labels)),
    }


def train(data_dir: Path, model_path: Path, task: str = "all_actions") -> dict:
    from sklearn.pipeline import Pipeline

    archive = np.load(data_dir / "epochs.npz")
    epochs, labels = archive["epochs"], archive["labels"]
    splits = indices_by_split(data_dir / "manifest.csv", data_dir / "subject_splits.json")
    epochs, labels, splits, class_labels = task_data(epochs, labels, splits, task, labels_for(data_dir))
    pipeline = Pipeline(
        [
            ("csp", mne.decoding.CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    pipeline.fit(epochs[splits["train"]], labels[splits["train"]])
    report = {
        "model": "csp_lda_v1",
        "task": task,
        "labels": class_labels,
        "preprocessing": {"bandpass_hz": [8, 30], "epoch_seconds": [0.5, 3.5], "reference": "average"},
        "split_epoch_counts": {split: len(indexes) for split, indexes in splits.items()},
        "validation": metrics(pipeline, epochs[splits["validation"]], labels[splits["validation"]]),
        "test": metrics(pipeline, epochs[splits["test"]], labels[splits["test"]]),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump(pipeline, stream)
    model_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/motor_imagery"))
    parser.add_argument("--model-path", type=Path, default=Path("models/csp_lda_v1.pkl"))
    parser.add_argument("--task", choices=TASKS, default="all_actions")
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.model_path, args.task), indent=2))


if __name__ == "__main__":
    main()
