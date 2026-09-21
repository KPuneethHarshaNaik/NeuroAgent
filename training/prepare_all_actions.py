"""Prepare eight actual/imagined motor-action classes from PhysioNet EEGMMIDB."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.prepare_dataset import process_recording, subject_splits

RUN_LABELS = {
    3: ("actual_left_hand", "actual_right_hand"), 7: ("actual_left_hand", "actual_right_hand"), 11: ("actual_left_hand", "actual_right_hand"),
    4: ("imagined_left_hand", "imagined_right_hand"), 8: ("imagined_left_hand", "imagined_right_hand"), 12: ("imagined_left_hand", "imagined_right_hand"),
    5: ("actual_both_fists", "actual_both_feet"), 9: ("actual_both_fists", "actual_both_feet"), 13: ("actual_both_fists", "actual_both_feet"),
    6: ("imagined_both_fists", "imagined_both_feet"), 10: ("imagined_both_fists", "imagined_both_feet"), 14: ("imagined_both_fists", "imagined_both_feet"),
}
LABELS = {index: label for index, label in enumerate(("actual_left_hand", "actual_right_hand", "imagined_left_hand", "imagined_right_hand", "actual_both_fists", "actual_both_feet", "imagined_both_fists", "imagined_both_feet"))}
LABEL_INDEX = {label: index for index, label in LABELS.items()}


def prepare(dataset_root: Path, output_dir: Path, overwrite: bool = False, low_hz: float = 8.0, high_hz: float = 30.0, tmin: float = 0.5, tmax: float = 3.5) -> dict[str, int]:
    subjects = sorted(path.name for path in dataset_root.glob("S[0-9][0-9][0-9]") if path.is_dir())
    if not subjects:
        raise ValueError(f"No S001-style subject folders found in {dataset_root}")
    archive, manifest_path = output_dir / "epochs.npz", output_dir / "manifest.csv"
    if (archive.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"{output_dir} already contains prepared data. Use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    batches, targets, rows = [], [], []
    for subject in subjects:
        for run, names in RUN_LABELS.items():
            path = dataset_root / subject / f"{subject}R{run:02d}.edf"
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"Preparing {subject} R{run:02d}", flush=True)
            data, event_labels, samples = process_recording(path, low_hz, high_hz, tmin, tmax)
            encoded = np.asarray([LABEL_INDEX[names[event]] for event in event_labels], dtype=np.int64)
            batches.append(data)
            start = len(targets)
            targets.extend(encoded.tolist())
            rows.extend({"epoch_index": start + index, "label": LABELS[int(label)], "subject_id": subject, "source_run": f"R{run:02d}", "event_sample": int(sample), "source_file": str(path)} for index, (label, sample) in enumerate(zip(encoded, samples, strict=True)))
    np.savez_compressed(archive, epochs=np.concatenate(batches), labels=np.asarray(targets, dtype=np.int64))
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch_index", "label", "subject_id", "source_run", "event_sample", "source_file"])
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "labels.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")
    (output_dir / "subject_splits.json").write_text(json.dumps(subject_splits(subjects), indent=2), encoding="utf-8")
    (output_dir / "preprocessing.json").write_text(json.dumps({"bandpass_hz": [low_hz, high_hz], "epoch_seconds": [tmin, tmax], "reference": "average"}, indent=2), encoding="utf-8")
    return {"subjects": len(subjects), "epochs": len(targets), "preprocessing": {"bandpass_hz": [low_hz, high_hz], "epoch_seconds": [tmin, tmax]}, **{label: int(np.count_nonzero(np.asarray(targets) == index)) for index, label in LABELS.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/all_actions"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--low-hz", type=float, default=8.0)
    parser.add_argument("--high-hz", type=float, default=30.0)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=3.5)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root, args.output_dir, args.overwrite, args.low_hz, args.high_hz, args.tmin, args.tmax), indent=2))


if __name__ == "__main__":
    main()
