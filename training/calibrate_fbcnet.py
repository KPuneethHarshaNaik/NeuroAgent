"""Measure how honest the trained FBCNet's confidence is, and calibrate it.

A reviewer agent cannot reason about a probability it cannot trust. This script answers
two separate questions about ``models/fbcnet_v1.pt`` and records both answers in
``models/fbcnet_v1_calibration.json``:

1. **Is the reported probability honest?** (calibration) — fixed with temperature
   scaling, fitted on the validation split and scored on the held-out test split.
2. **Does a higher probability mean a more likely correct prediction?**
   (discrimination) — measured as the accuracy gain of the most confident quartile of
   trials over the base rate. A model can be perfectly calibrated yet useless as
   evidence if its confidence does not separate correct from incorrect trials, so this
   flag is what decides whether the policy layer is allowed to trust confidence at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.fbcnet import FBCNet
from training.train_fbcnet import indices_by_split

DISCRIMINATION_GAIN = 0.10  # Top-quartile accuracy must beat the base rate by this much.
CONFIDENCE_BANDS = (("low", 0.0, 0.55), ("medium", 0.55, 0.65), ("high", 0.65, 1.01))


def expected_calibration_error(probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> float:
    """Mean gap between claimed confidence and observed accuracy, weighted by bin size."""
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        members = (confidence > low) & (confidence <= high)
        if not members.any():
            continue
        weight = members.float().mean().item()
        error += weight * abs(correct[members].float().mean().item() - confidence[members].mean().item())
    return error


def brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    one_hot = torch.zeros_like(probabilities).scatter_(1, targets.unsqueeze(1), 1.0)
    return float(((probabilities - one_hot) ** 2).sum(dim=1).mean())


def score(probabilities: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return {
        "accuracy": round(float(probabilities.argmax(dim=1).eq(targets).float().mean()), 4),
        "expected_calibration_error": round(expected_calibration_error(probabilities, targets), 4),
        "brier_score": round(brier_score(probabilities, targets), 4),
        "mean_confidence": round(float(probabilities.max(dim=1).values.mean()), 4),
        "n": int(len(targets)),
    }


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, maximum: float = 100.0) -> float:
    """Fit a single temperature that makes the validation logits least surprising.

    The parameter is log-temperature, so positivity holds by construction and no clamp can
    saturate the gradient (a clamped T lets the line search stall at the initial value).
    """
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=200)
    criterion = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return round(float(log_temperature.detach().exp().clamp(min=1e-3, max=maximum)), 4)


def confidence_is_discriminative(probabilities: torch.Tensor, targets: torch.Tensor, quartiles: int = 4) -> tuple[bool, dict[str, float]]:
    """Does the most confident slice of trials actually get predicted better?"""
    margins = probabilities.max(dim=1).values
    correct = probabilities.argmax(dim=1).eq(targets)
    base_rate = float(correct.float().mean())
    order = torch.argsort(margins)
    top = order[-(len(order) // quartiles):]
    top_accuracy = float(correct[top].float().mean())
    gain = top_accuracy - base_rate
    return gain >= DISCRIMINATION_GAIN, {"base_rate": round(base_rate, 4), "top_quartile_accuracy": round(top_accuracy, 4), "gain": round(gain, 4)}


def reliability_bands(probabilities: torch.Tensor, targets: torch.Tensor) -> list[dict[str, float | str | int]]:
    """Observed held-out accuracy inside each confidence band the policy may quote."""
    confidence = probabilities.max(dim=1).values
    correct = probabilities.argmax(dim=1).eq(targets)
    bands: list[dict[str, float | str | int]] = []
    for name, low, high in CONFIDENCE_BANDS:
        members = (confidence >= low) & (confidence < high)
        bands.append(
            {
                "band": name,
                "min_probability": low,
                "max_probability": high,
                "n": int(members.sum()),
                "observed_accuracy": round(float(correct[members].float().mean()), 4) if members.any() else None,
            }
        )
    return bands


def calibrate(data_dir: Path, checkpoint: Path, output: Path) -> dict:
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = FBCNet(channels=bundle["channels"], samples=bundle["samples"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    data = np.load(data_dir / "epochs.npz")
    epochs, labels = data["epochs"], data["labels"]
    epochs = (epochs - bundle["normalization_mean"]) / np.clip(bundle["normalization_std"], 1e-6, None)
    splits = indices_by_split(data_dir / "manifest.csv", data_dir / "subject_splits.json")

    with torch.no_grad():
        validation_logits = model(torch.from_numpy(epochs[splits["validation"]]).unsqueeze(1))
        test_logits = model(torch.from_numpy(epochs[splits["test"]]).unsqueeze(1))
    validation_targets = torch.from_numpy(labels[splits["validation"]]).long()
    test_targets = torch.from_numpy(labels[splits["test"]]).long()

    raw_test_probabilities = torch.softmax(test_logits, dim=1)
    temperature = fit_temperature(validation_logits, validation_targets)
    calibrated_test = torch.softmax(test_logits / temperature, dim=1)
    discriminative, discrimination = confidence_is_discriminative(calibrated_test, test_targets)

    report = {
        "model": checkpoint.stem,
        "method": "temperature_scaling",
        "temperature": temperature,
        "fit_on": "validation",
        "evaluated_on": "test",
        "before": score(raw_test_probabilities, test_targets),
        "after": score(calibrated_test, test_targets),
        "confidence_discriminative": discriminative,
        "discrimination": discrimination,
        "discrimination_rule": f"top-quartile accuracy must exceed the base rate by >= {DISCRIMINATION_GAIN}",
        "reliability_bands": reliability_bands(calibrated_test, test_targets),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/motor_imagery"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/fbcnet_v1.pt"))
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <checkpoint stem>_calibration.json.")
    args = parser.parse_args()
    output = args.output or args.checkpoint.with_name(f"{args.checkpoint.stem}_calibration.json")
    print(json.dumps(calibrate(args.data_dir, args.checkpoint, output), indent=2))


if __name__ == "__main__":
    main()
