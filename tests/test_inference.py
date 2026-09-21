"""Left/right motor-imagery inference over purpose-built recordings."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np

from backend.inference import (
    DEFAULT_MODEL_PATH,
    TRAINING_EPOCH_SECONDS,
    TRAINING_SAMPLING_RATE_HZ,
    classify,
    load_classifier,
)

CHECKPOINT_AVAILABLE = DEFAULT_MODEL_PATH.exists()
CHANNELS = 64
TRIALS = 12
RECORDING_SECONDS = 60.0


def synthetic_raw(channels: int = CHANNELS, sampling_rate: float = TRAINING_SAMPLING_RATE_HZ, descriptions: tuple[str, ...] = ("T1", "T2")) -> mne.io.BaseRaw:
    """A noise recording with evenly spaced annotations, long enough to yield every trial."""
    rng = np.random.default_rng(0)
    info = mne.create_info([f"EEG{index:03d}" for index in range(channels)], sampling_rate, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((channels, int(sampling_rate * RECORDING_SECONDS))) * 1e-5, info, verbose=False)
    onsets = np.arange(2.0, 2.0 + 4.0 * TRIALS, 4.0)
    labels = [descriptions[index % len(descriptions)] for index in range(TRIALS)]
    raw.set_annotations(mne.Annotations(onsets, [0.0] * TRIALS, labels))
    return raw


class CheckpointContractTests(unittest.TestCase):
    @unittest.skipUnless(CHECKPOINT_AVAILABLE, f"{DEFAULT_MODEL_PATH} is not present")
    def test_checkpoint_matches_the_inference_window(self) -> None:
        classifier, problem = load_classifier(DEFAULT_MODEL_PATH)
        self.assertIsNone(problem)
        assert classifier is not None
        expected_samples = int((TRAINING_EPOCH_SECONDS[1] - TRAINING_EPOCH_SECONDS[0]) * TRAINING_SAMPLING_RATE_HZ) + 1
        self.assertEqual((classifier.channels, classifier.samples), (CHANNELS, expected_samples))
        self.assertEqual(classifier.labels, {0: "left_hand", 1: "right_hand"})


class MissingCheckpointTests(unittest.TestCase):
    def test_missing_checkpoint_reports_its_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = classify(synthetic_raw(), path=Path(directory) / "absent.pt")
        self.assertEqual(report.status, "unavailable")
        self.assertEqual(report.reason_code, "no_model")
        self.assertIsNone(report.predicted_label)
        self.assertEqual(report.trial_count, 0)
        self.assertIn("absent.pt", report.reason or "")


@unittest.skipUnless(CHECKPOINT_AVAILABLE, f"{DEFAULT_MODEL_PATH} is not present")
class ClassifyTests(unittest.TestCase):
    def test_classifies_motor_imagery_trials(self) -> None:
        report = classify(synthetic_raw())
        self.assertEqual(report.status, "classified")
        self.assertEqual(report.model, DEFAULT_MODEL_PATH.stem)
        self.assertEqual(report.trial_count, TRIALS)
        self.assertIn(report.predicted_label, {"left_hand", "right_hand"})
        self.assertEqual(set(report.probabilities), {"left_hand", "right_hand"})
        self.assertAlmostEqual(sum(report.probabilities.values()), 1.0, places=3)
        self.assertEqual(sum(report.vote_counts.values()), TRIALS)
        self.assertIsNotNone(report.agreement)
        assert report.agreement is not None
        self.assertGreaterEqual(report.agreement, 0.0)
        self.assertLessEqual(report.agreement, 1.0)

    def test_resamples_recordings_recorded_at_another_rate(self) -> None:
        report = classify(synthetic_raw(sampling_rate=250.0))
        self.assertEqual(report.status, "classified")
        self.assertEqual(report.trial_count, TRIALS)

    def test_reports_unavailable_without_motor_imagery_annotations(self) -> None:
        report = classify(synthetic_raw(descriptions=("T0",)))
        self.assertEqual(report.status, "unavailable")
        self.assertEqual(report.reason_code, "no_annotations")
        self.assertIn("T1/T2", report.reason or "")
        self.assertEqual(report.trial_count, 0)

    def test_reports_unavailable_for_a_different_montage(self) -> None:
        report = classify(synthetic_raw(channels=32))
        self.assertEqual(report.status, "unavailable")
        self.assertEqual(report.reason_code, "montage_mismatch")
        self.assertIn(f"{CHANNELS} EEG channels", report.reason or "")
        self.assertIn("32", report.reason or "")

    def test_reports_calibrated_confidence_with_held_out_reliability(self) -> None:
        report = classify(synthetic_raw())
        calibration = report.calibration
        self.assertIsNotNone(calibration)
        assert calibration is not None
        self.assertEqual(calibration.method, "temperature_scaling")
        self.assertEqual(calibration.evaluated_on, "test")
        self.assertIn(calibration.band, {"low", "medium", "high"})
        self.assertGreater(calibration.band_sample_size or 0, 0)
        # The shipped checkpoint's confidence does not separate correct from incorrect trials.
        self.assertFalse(calibration.confidence_discriminative)
        self.assertTrue(any("cannot support acceptance" in warning for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()
