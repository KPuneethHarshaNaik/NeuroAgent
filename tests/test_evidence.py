"""Evidence bundle tests: provenance an agent cannot forge, and thresholds that travel with it."""

import unittest

from backend.evidence import EVIDENCE_SCHEMA_VERSION, PIPELINE_VERSION, build_evidence
from backend.policy import DEFAULT_THRESHOLDS, POLICY_VERSION, evaluate
from backend.schemas import CalibrationInfo, ClassificationReport, QualityReport, ValidationReport

VALID = ValidationReport(status="valid", file_format="edf", sampling_rate=160.0, channel_count=64, eeg_channel_count=64, duration_seconds=60.0, has_events=True, event_count=30)
UNAVAILABLE = ClassificationReport(status="unavailable", model="fbcnet_v1", reason_code="montage_mismatch", reason="not a 64-channel recording")


def prediction() -> ClassificationReport:
    return ClassificationReport(
        status="classified",
        model="fbcnet_v1",
        predicted_label="left_hand",
        probabilities={"left_hand": 0.52, "right_hand": 0.48},
        vote_counts={"left_hand": 8, "right_hand": 7},
        agreement=0.5333,
        trial_count=15,
        calibration=CalibrationInfo(method="temperature_scaling", temperature=26.5483, evaluated_on="test", confidence_discriminative=False, band="low", band_observed_accuracy=0.599, band_sample_size=591),
    )


def bundle(quality: QualityReport | None = None, classification: ClassificationReport | None = None):
    quality = quality if quality is not None else QualityReport(status="acceptable", peak_to_peak_uv=120.0, flat_channel_count=0)
    classification = classification if classification is not None else prediction()
    outcome = evaluate(VALID, quality, classification)
    return build_evidence("job-1", "recording.edf", "abc123", VALID, {"bandpass_hz": [8.0, 30.0], "reference": "average"}, {"epoch_count": 15, "window_seconds": [0.0, 2.0]}, quality, classification, outcome)


class ProvenanceTests(unittest.TestCase):
    def test_bundle_records_names_the_artifacts_it_was_judged_by(self) -> None:
        provenance = bundle().provenance
        self.assertEqual(provenance["input_hash"], "abc123")
        self.assertEqual(provenance["pipeline_version"], PIPELINE_VERSION)
        self.assertEqual(provenance["policy_version"], POLICY_VERSION)
        self.assertEqual(provenance["thresholds"], DEFAULT_THRESHOLDS.as_dict())
        self.assertEqual(provenance["model"], "fbcnet_v1")
        self.assertEqual(bundle().schema_version, EVIDENCE_SCHEMA_VERSION)

    def test_thresholds_travel_with_the_evidence(self) -> None:
        self.assertEqual(bundle().provenance["thresholds"]["minimum_trials"], 5)
        self.assertEqual(bundle().provenance["thresholds"]["recheck_limit"], 1)


class MeasurementTests(unittest.TestCase):
    def test_measurements_carry_the_numbers_a_reviewer_needs(self) -> None:
        measurements = bundle().measurements
        self.assertEqual(measurements["eeg_channel_count"], 64)
        self.assertEqual(measurements["trial_count"], 15)
        self.assertEqual(measurements["agreement"], 0.5333)
        self.assertEqual(measurements["probability_left_hand"], 0.52)
        self.assertEqual(measurements["predicted_label"], "left_hand")

    def test_measurements_include_rejected_epoch_ratio(self) -> None:
        self.assertEqual(bundle().measurements["rejected_epoch_ratio"], 0.0)

    def test_measurements_record_calibration_context(self) -> None:
        measurements = bundle().measurements
        self.assertFalse(measurements["confidence_discriminative"])
        self.assertEqual(measurements["confidence_band"], "low")
        self.assertEqual(measurements["band_sample_size"], 591)
        self.assertEqual(measurements["calibration_temperature"], 26.5483)

    def test_unknown_values_are_omitted_rather_than_nulled(self) -> None:
        measurements = bundle(classification=UNAVAILABLE).measurements
        self.assertNotIn("agreement", measurements)
        self.assertNotIn("predicted_label", measurements)
        self.assertEqual(measurements["failure_code"], "montage_mismatch")

    def test_quality_warnings_become_signals(self) -> None:
        quality = QualityReport(status="warning", peak_to_peak_uv=480.0, flat_channel_count=0, warnings=["High peak-to-peak amplitude detected."])
        self.assertIn("High peak-to-peak amplitude detected.", bundle(quality=quality).signals)


class GateSerializationTests(unittest.TestCase):
    def test_gates_are_recorded_with_their_requirement(self) -> None:
        gates = {gate.name: gate for gate in bundle().gates}
        self.assertEqual(gates["input_valid"].requirement, "valid EEG channels and event markers")
        self.assertTrue(gates["input_valid"].passed)
        self.assertFalse(gates["confidence_discriminative"].passed)
        self.assertIn("separate correct from incorrect", gates["confidence_discriminative"].requirement)

    def test_allowed_actions_follow_the_verdict(self) -> None:
        evidence = bundle(classification=UNAVAILABLE)
        self.assertEqual(evidence.allowed_actions, ["report", "report_uncertain", "request_human_review"])
        self.assertEqual(evidence.recheck_budget, {"limit": 1, "used": 0, "remaining": 1})


if __name__ == "__main__":
    unittest.main()
