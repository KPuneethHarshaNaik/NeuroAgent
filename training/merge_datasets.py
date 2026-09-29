"""Merge PhysioNet and Cho2017 prepared datasets into a single unified training set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.prepare_dataset import subject_splits


def merge(
    physionet_dir: Path,
    cho2017_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    archive = output_dir / "epochs.npz"
    if archive.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} already contains merged data; use --overwrite")

    # ── Load both datasets ──────────────────────────────────────────────
    print("Loading PhysioNet epochs...", flush=True)
    phy = np.load(physionet_dir / "epochs.npz")
    phy_epochs, phy_labels = phy["epochs"], phy["labels"]

    print("Loading Cho2017 epochs...", flush=True)
    cho = np.load(cho2017_dir / "epochs.npz")
    cho_epochs, cho_labels = cho["epochs"], cho["labels"]

    print(f"PhysioNet shape: {phy_epochs.shape}, Cho2017 shape: {cho_epochs.shape}", flush=True)

    # ── Align sample counts (trim to the shorter) ──────────────────────
    min_samples = min(phy_epochs.shape[2], cho_epochs.shape[2])
    if phy_epochs.shape[2] != min_samples:
        print(f"Trimming PhysioNet from {phy_epochs.shape[2]} to {min_samples} samples", flush=True)
        phy_epochs = phy_epochs[:, :, :min_samples]
    if cho_epochs.shape[2] != min_samples:
        print(f"Trimming Cho2017 from {cho_epochs.shape[2]} to {min_samples} samples", flush=True)
        cho_epochs = cho_epochs[:, :, :min_samples]

    # ── Verify channel count matches ───────────────────────────────────
    if phy_epochs.shape[1] != cho_epochs.shape[1]:
        raise ValueError(
            f"Channel mismatch: PhysioNet has {phy_epochs.shape[1]}, "
            f"Cho2017 has {cho_epochs.shape[1]}"
        )

    # ── Concatenate ────────────────────────────────────────────────────
    phy_count = len(phy_epochs)
    combined_epochs = np.concatenate([phy_epochs, cho_epochs], axis=0).astype(np.float32)
    combined_labels = np.concatenate([phy_labels, cho_labels], axis=0).astype(np.int64)
    print(f"Combined shape: {combined_epochs.shape}, labels: {combined_labels.shape}", flush=True)

    # Free memory
    del phy_epochs, cho_epochs, phy_labels, cho_labels, phy, cho

    # ── Merge manifests with prefixed subject IDs ──────────────────────
    rows: list[dict] = []

    def read_manifest(manifest_path: Path, prefix: str, index_offset: int) -> list[str]:
        subjects = set()
        with manifest_path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                new_index = int(row["epoch_index"]) + index_offset
                subject = f"{prefix}_{row['subject_id']}"
                subjects.add(subject)
                rows.append({
                    "epoch_index": new_index,
                    "label": row["label"],
                    "subject_id": subject,
                    "source_run": row["source_run"],
                    "event_sample": row["event_sample"],
                    "source_file": row["source_file"],
                })
        return sorted(subjects)

    phy_subjects = read_manifest(physionet_dir / "manifest.csv", "PHY", 0)
    cho_subjects = read_manifest(cho2017_dir / "manifest.csv", "CHO", phy_count)
    all_subjects = phy_subjects + cho_subjects

    print(f"Total subjects: {len(all_subjects)} (PhysioNet: {len(phy_subjects)}, Cho2017: {len(cho_subjects)})", flush=True)

    # ── Save ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, epochs=combined_epochs, labels=combined_labels)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch_index", "label", "subject_id", "source_run", "event_sample", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    splits = subject_splits(all_subjects)
    (output_dir / "subject_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps({"0": "left_hand", "1": "right_hand"}, indent=2), encoding="utf-8")
    (output_dir / "preprocessing.json").write_text(json.dumps({
        "datasets": ["PhysioNet_EEGMMIDB", "Cho2017"],
        "bandpass_hz": [8.0, 30.0],
        "epoch_seconds": [0.5, 3.5],
        "target_rate_hz": 160,
        "channels": combined_epochs.shape[1],
        "samples": min_samples,
        "reference": "average",
    }, indent=2), encoding="utf-8")

    summary = {
        "subjects": len(all_subjects),
        "physionet_subjects": len(phy_subjects),
        "cho2017_subjects": len(cho_subjects),
        "total_epochs": len(combined_labels),
        "left_hand": int((combined_labels == 0).sum()),
        "right_hand": int((combined_labels == 1).sum()),
        "channels": combined_epochs.shape[1],
        "samples": min_samples,
        "splits": {split: len(subjects) for split, subjects in splits.items()},
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physionet-dir", type=Path, default=Path("data/processed/motor_imagery"))
    parser.add_argument("--cho2017-dir", type=Path, default=Path("data/processed/cho2017"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/combined"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    merge(args.physionet_dir, args.cho2017_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
