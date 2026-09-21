"""Upload a recording through POST /jobs and check the verdict and evidence bundle."""

from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
from fastapi import UploadFile

import backend.main as api
from backend.inference import DEFAULT_MODEL_PATH
from backend.schemas import HumanReviewRequest

CHANNELS = 64
SAMPLING_RATE_HZ = 160.0
TRIALS = 12


def write_recording(path: Path, descriptions: tuple[str, ...] = ("T1", "T2")) -> Path:
    rng = np.random.default_rng(0)
    info = mne.create_info([f"EEG{index:03d}" for index in range(CHANNELS)], SAMPLING_RATE_HZ, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((CHANNELS, int(SAMPLING_RATE_HZ * 60))) * 1e-5, info, verbose=False)
    if descriptions:
        onsets = np.arange(2.0, 2.0 + 4.0 * TRIALS, 4.0)
        labels = [descriptions[index % len(descriptions)] for index in range(TRIALS)]
        raw.set_annotations(mne.Annotations(onsets, [0.0] * TRIALS, labels))
    raw.save(path, overwrite=True, verbose=False)
    return path


class JobsEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_temp = Path.cwd() / "runtime" / "test_tmp"
        self.workspace_temp.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=self.workspace_temp)
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(setattr, api, "RUNTIME_DIR", api.RUNTIME_DIR)
        api.RUNTIME_DIR = Path(self.directory.name) / "runtime"

    def submit(self, path: Path) -> api.JobReport:
        upload = UploadFile(file=io.BytesIO(path.read_bytes()), filename=path.name)
        return asyncio.run(api.create_job(upload))

    def recording(self, name: str, descriptions: tuple[str, ...] = ("T1", "T2")) -> api.JobReport:
        return self.submit(write_recording(Path(self.directory.name) / name, descriptions))

    def stages(self, report: api.JobReport) -> list[str]:
        return [entry["stage"] for entry in report.audit_events]

    @unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), f"{DEFAULT_MODEL_PATH} is not present")
    def test_report_carries_prediction_verdict_and_evidence(self) -> None:
        report = self.recording("S001R04_eeg.fif")
        self.assertEqual(report.status, "completed")
        assert report.classification is not None and report.evidence is not None
        self.assertEqual(report.classification.status, "classified")
        self.assertEqual(report.classification.model, DEFAULT_MODEL_PATH.stem)
        self.assertEqual(report.classification.trial_count, TRIALS)
        self.assertIn(report.classification.predicted_label, {"left_hand", "right_hand"})
        # Plain ACCEPT is withheld because this checkpoint's confidence is not discriminative.
        self.assertEqual(report.decision, "ACCEPT_WITH_WARNING")
        self.assertFalse(report.evidence.measurements["confidence_discriminative"])
        self.assertEqual(report.evidence.provenance["input_hash"], report.input_hash)
        self.assertIn("classification", self.stages(report))
        self.assertIn("policy", self.stages(report))

    def test_recording_without_motor_imagery_trials_is_refused_not_guessed(self) -> None:
        report = self.recording("rest_eeg.fif", descriptions=("T0",))
        self.assertEqual(report.status, "completed")
        assert report.classification is not None and report.evidence is not None
        self.assertEqual(report.classification.status, "unavailable")
        self.assertEqual(report.classification.reason_code, "no_annotations")
        self.assertEqual(report.decision, "RETURN_UNCERTAIN")
        self.assertNotIn("publish", report.evidence.allowed_actions)
        self.assertIn("classification_skipped", self.stages(report))

    def test_recording_without_events_is_rejected_with_evidence(self) -> None:
        report = self.recording("silent_eeg.fif", descriptions=())
        self.assertEqual(report.status, "invalid_input")
        self.assertEqual(report.decision, "REJECT_INVALID_INPUT")
        assert report.evidence is not None
        gates = {gate.name: gate for gate in report.evidence.gates}
        self.assertFalse(gates["input_valid"].passed)
        self.assertEqual(report.evidence.allowed_actions, ["reject"])

    def test_unsupported_format_is_refused_before_reading(self) -> None:
        upload = UploadFile(file=io.BytesIO(b"not eeg"), filename="notes.txt")
        with self.assertRaises(Exception):
            asyncio.run(api.create_job(upload))

    @unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), f"{DEFAULT_MODEL_PATH} is not present")
    def test_human_reviewer_can_mark_a_prediction_uncertain(self) -> None:
        report = self.recording("review_eeg.fif")
        reviewed = api.review_job(report.job_id, HumanReviewRequest(action="mark_uncertain", reviewer_comment="Trial evidence is insufficient."))
        assert reviewed.human_review is not None
        self.assertEqual(reviewed.human_review.approved_label, "uncertain")
        self.assertIn("human_review", self.stages(reviewed))

    @unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), f"{DEFAULT_MODEL_PATH} is not present")
    def test_override_requires_a_hand_label(self) -> None:
        report = self.recording("override_eeg.fif")
        with self.assertRaises(Exception):
            api.review_job(report.job_id, HumanReviewRequest(action="override", approved_label="uncertain"))


if __name__ == "__main__":
    unittest.main()
