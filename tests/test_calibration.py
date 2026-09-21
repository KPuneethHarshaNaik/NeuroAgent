"""Calibration measurement tests, including the finding that drives the acceptance policy."""

import json
import unittest
from pathlib import Path

import torch

from backend.inference import calibration_path
from training.calibrate_fbcnet import (
    confidence_is_discriminative,
    expected_calibration_error,
    fit_temperature,
    reliability_bands,
)

CHECKPOINT = Path("models/fbcnet_v1.pt")
ARTIFACT = calibration_path(CHECKPOINT)


class CalibrationMetricTests(unittest.TestCase):
    def test_perfect_confidence_has_no_calibration_error(self) -> None:
        probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        self.assertEqual(expected_calibration_error(probabilities, targets), 0.0)

    def test_half_right_at_full_confidence_is_maximally_miscalibrated(self) -> None:
        probabilities = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        targets = torch.tensor([0, 1])
        self.assertAlmostEqual(expected_calibration_error(probabilities, targets), 0.5, places=6)

    def test_overconfident_logits_are_softened_by_temperature(self) -> None:
        # Three maximally confident correct trials and one maximally confident wrong one:
        # the honest probability is 0.75, so the temperature must rise above 1.
        logits = torch.tensor([[9.0, -9.0], [9.0, -9.0], [9.0, -9.0], [-9.0, 9.0]])
        targets = torch.tensor([0, 0, 0, 0])
        before = torch.softmax(logits, dim=1)
        temperature = fit_temperature(logits, targets)
        after = torch.softmax(logits / temperature, dim=1)
        self.assertGreater(temperature, 1.0)
        self.assertLess(expected_calibration_error(after, targets), expected_calibration_error(before, targets))

    def test_temperature_never_changes_the_prediction(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [-3.0, 0.5], [0.1, 0.2]])
        temperature = fit_temperature(logits, torch.tensor([0, 0, 1]))
        self.assertTrue(torch.equal(torch.softmax(logits, dim=1).argmax(1), torch.softmax(logits / temperature, dim=1).argmax(1)))

    def test_accurate_confident_predictions_are_discriminative(self) -> None:
        probabilities = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.51, 0.49], [0.9, 0.1], [0.49, 0.51], [0.1, 0.9]])
        targets = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1])
        discriminative, detail = confidence_is_discriminative(probabilities, targets)
        self.assertTrue(discriminative, detail)
        self.assertGreater(detail["gain"], 0.0)

    def test_confidently_wrong_predictions_are_not_discriminative(self) -> None:
        probabilities = torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.4, 0.6], [0.95, 0.05], [0.6, 0.4], [0.05, 0.95], [0.4, 0.6]])
        targets = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1])
        discriminative, detail = confidence_is_discriminative(probabilities, targets)
        self.assertFalse(discriminative, detail)
        self.assertLess(detail["gain"], 0.0)

    def test_reliability_bands_account_for_every_trial(self) -> None:
        probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.52, 0.48], [0.51, 0.49]])
        bands = reliability_bands(probabilities, torch.tensor([0, 1, 0, 1]))
        self.assertEqual([band["band"] for band in bands], ["low", "medium", "high"])
        self.assertEqual(sum(int(band["n"]) for band in bands), 4)


@unittest.skipUnless(ARTIFACT.exists(), f"{ARTIFACT} is not present; run training.calibrate_fbcnet")
class ShippedCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_calibration_improves_honesty_without_changing_accuracy(self) -> None:
        before, after = self.artifact["before"], self.artifact["after"]
        self.assertLess(after["expected_calibration_error"], before["expected_calibration_error"])
        self.assertLess(after["brier_score"], before["brier_score"])
        self.assertGreater(before["mean_confidence"] - after["mean_confidence"], 0.15)
        self.assertEqual(before["accuracy"], after["accuracy"])

    def test_shipped_checkpoint_confidence_is_not_discriminative(self) -> None:
        """If this ever flips, the policy may start issuing plain ACCEPT — so assert it."""
        self.assertFalse(self.artifact["confidence_discriminative"])
        self.assertLess(self.artifact["discrimination"]["gain"], 0.10)

    def test_higher_confidence_is_not_more_accurate(self) -> None:
        bands = {band["band"]: band for band in self.artifact["reliability_bands"]}
        self.assertLessEqual(bands["high"]["observed_accuracy"], bands["low"]["observed_accuracy"])


if __name__ == "__main__":
    unittest.main()
