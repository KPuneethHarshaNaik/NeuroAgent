"""Prepare left/right motor-imagery epochs from the PhysioNet EEGMMIDB data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUNS = (4, 8, 12)
EVENT_IDS = {"T1": 0, "T2": 1}  # T1 = imagined left fist; T2 = imagined right fist.
LABELS = {0: "left_hand", 1: "right_hand"}


def subject_splits(subjects: list[str]) -> dict[str, list[str]]:
    """Create a fixed 70/15/15 subject-wise split."""
    ordered = sorted(subjects)
    train_end, validation_end = round(len(ordered) * 0.70), round(len(ordered) * 0.85)
    return {"train": ordered[:train_end], "validation": ordered[train_end:validation_end], "test": ordered[validation_end:]}


def process_recording(path: Path, low_hz: float = 8.0, high_hz: float = 30.0, tmin: float = 0.5, tmax: float = 3.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick("eeg")
    if raw.info["sfreq"] != 160:
        raw.resample(160, verbose=False)
    raw.filter(low_hz, high_hz, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_IDS, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=EVENT_IDS, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    return epochs.get_data(copy=False).astype(np.float32), epochs.events[:, 2], epochs.events[:, 0]


def prepare(dataset_root: Path, output_dir: Path, overwrite: bool = False, low_hz: float = 8.0, high_hz: float = 30.0, tmin: float = 0.5, tmax: float = 3.5) -> dict[str, int]:
    subjects = sorted(path.name for path in dataset_root.glob("S[0-9][0-9][0-9]") if path.is_dir())
    if not subjects:
        raise ValueError(f"No S001-style subject folders found in {dataset_root}")
    archive = output_dir / "epochs.npz"
    manifest_path = output_dir / "manifest.csv"
    if (archive.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"{output_dir} already contains prepared data. Use --overwrite to replace it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_batches: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, str | int]] = []
    for subject in subjects:
        for run in RUNS:
            path = dataset_root / subject / f"{subject}R{run:02d}.edf"
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"Preparing {subject} R{run:02d}", flush=True)
            data, event_labels, samples = process_recording(path, low_hz, high_hz, tmin, tmax)
            epoch_batches.append(data)
            start = len(labels)
            labels.extend(event_labels.tolist())
            rows.extend(
                {
                    "epoch_index": start + index,
                    "label": LABELS[int(label)],
                    "subject_id": subject,
                    "source_run": f"R{run:02d}",
                    "event_sample": int(sample),
                    "source_file": str(path),
                }
                for index, (label, sample) in enumerate(zip(event_labels, samples, strict=True))
            )

    data = np.concatenate(epoch_batches)
    targets = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(archive, epochs=data, labels=targets)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch_index", "label", "subject_id", "source_run", "event_sample", "source_file"])
        writer.writeheader()
        writer.writerows(rows)
    splits = subject_splits(subjects)
    (output_dir / "subject_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (output_dir / "preprocessing.json").write_text(json.dumps({"bandpass_hz": [low_hz, high_hz], "epoch_seconds": [tmin, tmax], "reference": "average"}, indent=2), encoding="utf-8")
    return {"subjects": len(subjects), "epochs": len(targets), "left_hand": int((targets == 0).sum()), "right_hand": int((targets == 1).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing S001...S109.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/motor_imagery"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--low-hz", type=float, default=8.0)
    parser.add_argument("--high-hz", type=float, default=30.0)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=3.5)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root, args.output_dir, args.overwrite, args.low_hz, args.high_hz, args.tmin, args.tmax), indent=2))


if __name__ == "__main__":
    main()
