"""Convert Cho2017 MATLAB structs into NeuroAgent's subject-wise NPZ format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from training.prepare_dataset import subject_splits


def bandpass(signal: np.ndarray, low_hz: float, high_hz: float, fs: int, order: int = 5) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the last axis."""
    sos = butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)


def trials(signal: np.ndarray, events: np.ndarray, source_rate: int, target_rate: int, tmin: float, tmax: float) -> np.ndarray:
    starts = np.flatnonzero(events > 0)
    length = int(round((tmax - tmin) * source_rate))
    offset = int(round(tmin * source_rate))
    windows = [signal[:, start + offset : start + offset + length] for start in starts]
    if not windows or any(window.shape != (signal.shape[0], length) for window in windows):
        raise ValueError("Cho2017 contains an incomplete imagery epoch")
    data = np.stack(windows).astype(np.float32)
    if source_rate != target_rate:
        data = resample_poly(data, target_rate, source_rate, axis=2).astype(np.float32)
    return data


def prepare(dataset_dir: Path, output_dir: Path, overwrite: bool = False, channels: int = 64, target_rate: int = 160, tmin: float = 0.5, tmax: float = 3.5, low_hz: float = 8.0, high_hz: float = 30.0) -> dict:
    files = sorted(dataset_dir.glob("s[0-9][0-9].mat"))
    if not files:
        raise ValueError(f"No s01.mat-style files found in {dataset_dir}")
    if (output_dir / "epochs.npz").exists() and not overwrite:
        raise FileExistsError(f"{output_dir} already contains prepared data; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    batches, labels, rows = [], [], []
    subjects = [path.stem.upper() for path in files]
    for path in files:
        record = loadmat(path, struct_as_record=False, squeeze_me=True)["eeg"]
        source_rate = int(record.srate)
        if source_rate != 512:
            raise ValueError(f"{path.name}: expected 512 Hz, got {record.srate}")
        if record.imagery_left.shape[0] < channels or record.imagery_right.shape[0] < channels:
            raise ValueError(f"{path.name}: expected at least {channels} EEG channels")
        events = np.asarray(record.imagery_event)
        start_index = sum(len(batch) for batch in batches)
        for label, name, signal in ((0, "left_hand", record.imagery_left), (1, "right_hand", record.imagery_right)):
            filtered = bandpass(np.asarray(signal, dtype=np.float64)[:channels], low_hz, high_hz, source_rate)
            data = trials(filtered, events, source_rate, target_rate, tmin, tmax)
            batches.append(data)
            labels.extend([label] * len(data))
            rows.extend({"epoch_index": start_index + i, "label": name, "subject_id": path.stem.upper(), "source_run": "imagery", "event_sample": int(sample), "source_file": str(path)} for i, sample in enumerate(np.flatnonzero(events > 0)))
            start_index += len(data)
        print(f"Prepared {path.name}", flush=True)
    data = np.concatenate(batches).astype(np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(output_dir / "epochs.npz", epochs=data, labels=targets)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch_index", "label", "subject_id", "source_run", "event_sample", "source_file"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "subject_splits.json").write_text(json.dumps(subject_splits(subjects), indent=2), encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps({"0": "left_hand", "1": "right_hand"}, indent=2), encoding="utf-8")
    (output_dir / "preprocessing.json").write_text(json.dumps({"dataset": "Cho2017", "bandpass_hz": [low_hz, high_hz], "epoch_seconds": [tmin, tmax], "source_rate_hz": 512, "target_rate_hz": target_rate, "channels_used": channels, "reference": "source dataset"}, indent=2), encoding="utf-8")
    return {"subjects": len(subjects), "epochs": len(targets), "left_hand": int((targets == 0).sum()), "right_hand": int((targets == 1).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cho2017"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--low-hz", type=float, default=8.0)
    parser.add_argument("--high-hz", type=float, default=30.0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_dir, args.output_dir, args.overwrite, low_hz=args.low_hz, high_hz=args.high_hz), indent=2))


if __name__ == "__main__":
    main()
