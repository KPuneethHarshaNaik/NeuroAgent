"""Check that prepared EEG data is complete, balanced, and subject-disjoint."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def audit(data_dir: Path) -> dict:
    archive = np.load(data_dir / "epochs.npz")
    epochs, labels = archive["epochs"], archive["labels"]
    with (data_dir / "manifest.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    splits = json.loads((data_dir / "subject_splits.json").read_text(encoding="utf-8"))
    split_for = {subject: split for split, subjects in splits.items() for subject in subjects}
    coverage: dict[str, set[tuple[str, str]]] = defaultdict(set)
    split_labels: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        subject, label = row["subject_id"], row["label"]
        coverage[subject].add((row["source_run"], label))
        split_labels[split_for[subject]][label] += 1
    expected = {(run, label) for run in ("R04", "R08", "R12") for label in ("left_hand", "right_hand")}
    missing = {subject: sorted(expected - values) for subject, values in coverage.items() if expected - values}
    all_subjects = [subject for members in splits.values() for subject in members]
    report = {
        "data_dir": str(data_dir),
        "epochs_shape": list(epochs.shape),
        "epochs_dtype": str(epochs.dtype),
        "epochs": int(len(labels)),
        "label_counts": np.bincount(labels).tolist(),
        "manifest_rows": len(rows),
        "indexes_contiguous": sorted(int(row["epoch_index"]) for row in rows) == list(range(len(labels))),
        "subjects": {split: len(members) for split, members in splits.items()},
        "subject_split_disjoint": len(all_subjects) == len(set(all_subjects)),
        "split_label_counts": {split: dict(counts) for split, counts in split_labels.items()},
        "subjects_missing_a_class_or_run": missing,
        "passed": len(rows) == len(labels) and not missing and len(all_subjects) == len(set(all_subjects)),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/motor_imagery"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data_dir)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
