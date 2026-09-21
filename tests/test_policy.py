"""Deterministic policy tests: gates, the bounded recheck budget, and the action allowlist."""

import unittest

from backend.policy import ALLOWED_ACTIONS, RecheckState, authorize, decide, evaluate
from backend.schemas import CalibrationInfo, ClassificationReport, QualityReport, ValidationReport

VALID = ValidationReport(status="valid", file_format="edf", eeg_channel_count=64, event_count=30)
INVALID = ValidationReport(status="invalid", file_format="edf", eeg_channel_count=0, event_count=0)


def calibration(discriminative: bool) -> CalibrationInfo:
    return CalibrationInfo(method="temperature_scaling", temperature=26.5483, evaluated_on="test", confidence_discriminative=discriminative, band="low", band_observed_accuracy=0.599, band_sample_size=591)


def prediction(trials: int = 15, agreement: float = 0.8, discriminative: bool = True, calibrated: bool = True) -> ClassificationReport:
    return ClassificationReport(status="classified", model="fbcnet_v1", predicted_label="left_hand", probabilities={"left_hand": 0.6, "right_hand": 0.4}, vote_counts={"left_hand": 12, "right_hand": 3}, agreement=agreement, trial_count=trials, calibration=calibration(discriminative) if calibrated else None)


UNAVAILABLE = ClassificationReport(status="unavailable", model="fbcnet_v1", reason_code="montage_mismatch", reason="not a 64-channel recording")


class DecisionTests(unittest.TestCase):
    def test_rejects_invalid_input(self) -> None:
        self.assertEqual(decide(INVALID, None, None), "REJECT_INVALID_INPUT")

    def test_rechecks_poor_quality_while_budget_remains(self) -> None:
        self.assertEqual(decide(VALID, QualityReport(status="poor"), prediction()), "RECHECK_QUALITY")

    def test_escalates_to_human_review_when_the_budget_is_spent(self) -> None:
        spent = RecheckState(limit=1, used=1)
        self.assertEqual(spent.remaining, 0)
        self.assertEqual(decide(VALID, QualityReport(status="poor"), prediction(), spent), "REQUEST_HUMAN_REVIEW")

    def test_returns_uncertain_when_no_prediction_could_be_made(self) -> None:
        self.assertEqual(decide(VALID, QualityReport(status="acceptable"), UNAVAILABLE), "RETURN_UNCERTAIN")

    def test_returns_uncertain_when_there_are_too_few_trials(self) -> None:
        self.assertEqual(decide(VALID, QualityReport(status="acceptable"), prediction(trials=2)), "RETURN_UNCERTAIN")

    def test_accepts_clean_evidence(self) -> None:
        self.assertEqual(decide(VALID, QualityReport(status="acceptable"), prediction()), "ACCEPT")

    def test_warns_when_confidence_cannot_support_acceptance(self) -> None:
        self.assertEqual(decide(VALID, QualityReport(status="acceptable"), prediction(discriminative=False)), "ACCEPT_WITH_WARNING")

    def test_warns_when_a_prediction_has_no_calibration_evidence(self) -> None:
        self.assertEqual(decide(VALID, QualityReport(status="acceptable"), prediction(calibrated=False)), "ACCEPT_WITH_WARNING")

    def test_warns_on_disagreeing_trials(self) -> None:
        self.assertEqual(decide(VALID, QualityReport(status="acceptable"), prediction(agreement=0.5)), "ACCEPT_WITH_WARNING")

    def test_warns_on_borderline_quality(self) -> None:
        self.assertEqual(decide(VALID, QualityReport(status="warning"), prediction()), "ACCEPT_WITH_WARNING")

    def test_every_declared_decision_is_reachable(self) -> None:
        """Guards the Phase 1 wart where half the verdict enum could never be produced."""
        reached = {
            decide(INVALID, None, None),
            decide(VALID, QualityReport(status="poor"), prediction()),
            decide(VALID, QualityReport(status="poor"), prediction(), RecheckState(limit=1, used=1)),
            decide(VALID, QualityReport(status="acceptable"), UNAVAILABLE),
            decide(VALID, QualityReport(status="warning"), prediction()),
            decide(VALID, QualityReport(status="acceptable"), prediction()),
        }
        self.assertEqual(reached, set(ALLOWED_ACTIONS))


class GateTests(unittest.TestCase):
    def gates(self, outcome) -> dict:
        return {gate.name: gate for gate in outcome.gates}

    def test_failing_gates_name_what_they_saw(self) -> None:
        gates = self.gates(evaluate(VALID, QualityReport(status="acceptable"), UNAVAILABLE))
        self.assertFalse(gates["classification_available"].passed)
        self.assertEqual(gates["classification_available"].observed, "unavailable")
        self.assertTrue(gates["classification_available"].blocking)
        self.assertTrue(gates["input_valid"].passed)

    def test_soft_gates_do_not_block(self) -> None:
        gates = self.gates(evaluate(VALID, QualityReport(status="acceptable"), prediction(discriminative=False)))
        self.assertFalse(gates["confidence_discriminative"].passed)
        self.assertFalse(gates["confidence_discriminative"].blocking)

    def test_signals_explain_a_non_discriminative_confidence(self) -> None:
        outcome = evaluate(VALID, QualityReport(status="acceptable"), prediction(discriminative=False))
        self.assertTrue(any("does NOT separate" in signal for signal in outcome.signals), outcome.signals)


class AllowlistTests(unittest.TestCase):
    def test_unknown_actions_are_refused(self) -> None:
        self.assertFalse(authorize("delete_labels", "ACCEPT"))

    def test_actions_outside_the_verdict_are_refused(self) -> None:
        self.assertFalse(authorize("recheck_quality", "ACCEPT"))
        self.assertFalse(authorize("publish", "REJECT_INVALID_INPUT"))
        self.assertTrue(authorize("report", "ACCEPT"))

    def test_recheck_is_refused_without_budget(self) -> None:
        self.assertTrue(authorize("recheck_quality", "RECHECK_QUALITY", RecheckState(limit=1, used=0)))
        self.assertFalse(authorize("recheck_quality", "RECHECK_QUALITY", RecheckState(limit=1, used=1)))
        self.assertFalse(authorize("recheck_quality", "RECHECK_QUALITY"))

    def test_review_escalation_is_always_permitted(self) -> None:
        self.assertTrue(authorize("request_human_review", "REQUEST_HUMAN_REVIEW"))
        self.assertTrue(authorize("request_human_review", "RETURN_UNCERTAIN"))

    def test_spending_the_budget_is_monotonic(self) -> None:
        state = RecheckState(limit=2)
        self.assertEqual((state.remaining, state.spend().remaining, state.spend().spend().remaining), (2, 1, 0))
        self.assertEqual(state.spend().spend().spend().remaining, 0)


if __name__ == "__main__":
    unittest.main()
