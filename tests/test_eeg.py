"""Signal-quality checks are measured per event-locked epoch, not inferred globally."""

import unittest

import mne
import numpy as np

from backend.eeg import epoch_and_measure, validate


def raw_with_artifacts(artifact_epochs: int) -> mne.io.BaseRaw:
    sampling_rate = 100.0
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((4, 1_000)) * 1e-6, mne.create_info([f"EEG{index}" for index in range(4)], sampling_rate, "eeg"), verbose=False)
    onsets = np.array([1.0, 3.0, 5.0, 7.0])
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), ["T1"] * len(onsets)))
    for onset in onsets[:artifact_epochs]:
        start = int(onset * sampling_rate)
        raw._data[0, start : start + 20] = 500e-6
    return raw


class EpochQualityTests(unittest.TestCase):
    def test_artifact_ratio_marks_many_bad_epochs_as_poor(self) -> None:
        _, quality = epoch_and_measure(raw_with_artifacts(2))
        self.assertEqual(quality.rejected_epoch_ratio, 0.5)
        self.assertEqual(quality.status, "poor")

    def test_one_bad_epoch_is_a_warning(self) -> None:
        _, quality = epoch_and_measure(raw_with_artifacts(1))
        self.assertEqual(quality.rejected_epoch_ratio, 0.25)
        self.assertEqual(quality.status, "warning")

    def test_recording_without_events_is_invalid(self) -> None:
        raw = mne.io.RawArray(np.zeros((2, 100)), mne.create_info(["EEG0", "EEG1"], 100, "eeg"), verbose=False)
        report = validate(raw, ".fif")
        self.assertEqual(report.status, "invalid")
        self.assertIn("No event markers", report.warnings[0])

    def test_recording_without_eeg_channels_is_invalid(self) -> None:
        raw = mne.io.RawArray(np.zeros((1, 100)), mne.create_info(["MEG0"], 100, "mag"), verbose=False)
        raw.set_annotations(mne.Annotations([0.2], [0.0], ["T1"]))
        report = validate(raw, ".fif")
        self.assertEqual(report.status, "invalid")
        self.assertIn("No EEG channels", report.warnings[0])


if __name__ == "__main__":
    unittest.main()
