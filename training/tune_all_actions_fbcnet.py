"""Select a compact FBCNet configuration using validation accuracy only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.train_fbcnet import train

CANDIDATES = (
    {"learning_rate": 1e-3, "weight_decay": 1e-4, "bands": 4, "spatial_filters": 8},
    {"learning_rate": 5e-4, "weight_decay": 1e-4, "bands": 4, "spatial_filters": 8},
    {"learning_rate": 1e-3, "weight_decay": 5e-4, "bands": 4, "spatial_filters": 12},
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/all_actions"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    models = Path("models")
    reports = []
    for index, config in enumerate(CANDIDATES, start=1):
        path = models / f"fbcnet_all_actions_candidate_{index}.pt"
        report = train(args.data_dir, path, args.epochs, args.batch_size, seed=7, **config)
        reports.append(report)
    winner = max(enumerate(reports, start=1), key=lambda item: item[1]["best_validation_accuracy"])
    candidate_index, winner_report = winner
    candidate_path = models / f"fbcnet_all_actions_candidate_{candidate_index}.pt"
    final_path = models / "fbcnet_all_actions_v1.pt"
    shutil.copy2(candidate_path, final_path)
    final_report = {**winner_report, "model": final_path.stem, "tuning": {"selection_metric": "validation_accuracy", "candidates": [{"model": report["model"], "best_validation_accuracy": report["best_validation_accuracy"], "hyperparameters": report["hyperparameters"]} for report in reports]}}
    final_path.with_suffix(".json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
