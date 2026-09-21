"""Deterministic acceptance policy.

Agents may *recommend*; only this module *authorizes*. Nothing here consults a language
model, and every threshold is a frozen constant that gets recorded in the evidence bundle,
so no agent can quietly change what "acceptable" means.

Two rules drive the shape of this policy, both of them measured rather than assumed
(see ``training/calibrate_fbcnet.py`` and ``models/fbcnet_v1_calibration.json``):

* A reported probability is only trustworthy if it is *discriminative* — if more confident
  predictions are correspondingly more often correct. The shipped checkpoint is not, so
  plain ``ACCEPT`` is unreachable for it and the honest verdict is ``ACCEPT_WITH_WARNING``.
* A recheck is a bounded diagnostic re-run, never a search for a friendlier answer. Once
  the budget is spent the only escalation left is a human.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from backend.schemas import ClassificationReport, Decision, GateResult, QualityReport, ValidationReport

POLICY_VERSION = "2.1"


@dataclass(frozen=True)
class PolicyThresholds:
    """Acceptance criteria. Recorded verbatim in the evidence bundle."""

    minimum_trials: int = 5
    minimum_agreement: float = 0.6
    maximum_peak_to_peak_uv: float = 300.0
    maximum_flat_channels: int = 0
    maximum_rejected_epoch_ratio: float = 0.25
    recheck_limit: int = 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "minimum_trials": self.minimum_trials,
            "minimum_agreement": self.minimum_agreement,
            "maximum_peak_to_peak_uv": self.maximum_peak_to_peak_uv,
            "maximum_flat_channels": self.maximum_flat_channels,
            "maximum_rejected_epoch_ratio": self.maximum_rejected_epoch_ratio,
            "recheck_limit": self.recheck_limit,
        }


DEFAULT_THRESHOLDS = PolicyThresholds()

# What each verdict permits. An action absent from the verdict's list is refused outright,
# so a supervisor agent cannot invent capabilities the policy never granted it.
ALLOWED_ACTIONS: dict[str, tuple[str, ...]] = {
    "ACCEPT": ("report", "publish"),
    "ACCEPT_WITH_WARNING": ("report", "publish_with_warning"),
    "RECHECK_QUALITY": ("recheck_quality",),
    "RETURN_UNCERTAIN": ("report", "report_uncertain", "request_human_review"),
    "REQUEST_HUMAN_REVIEW": ("request_human_review",),
    "REJECT_INVALID_INPUT": ("reject",),
}


@dataclass(frozen=True)
class RecheckState:
    """How many diagnostic re-runs this job has left."""

    limit: int = DEFAULT_THRESHOLDS.recheck_limit
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    def spend(self) -> RecheckState:
        return replace(self, used=self.used + 1)

    def as_dict(self) -> dict[str, int]:
        return {"limit": self.limit, "used": self.used, "remaining": self.remaining}


@dataclass(frozen=True)
class PolicyOutcome:
    decision: Decision
    gates: tuple[GateResult, ...]
    signals: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    recheck: RecheckState
    thresholds: PolicyThresholds


def authorize(action: str, decision: Decision, recheck: RecheckState | None = None) -> bool:
    """The validator every proposed action must pass. Unknown actions are refused."""
    if action not in ALLOWED_ACTIONS[decision]:
        return False
    if action == "recheck_quality" and (recheck is None or recheck.remaining <= 0):
        return False
    return True


def confidence_is_trustworthy(classification: ClassificationReport | None) -> bool:
    """Without a calibration artifact we cannot claim confidence means anything."""
    if classification is None or classification.calibration is None:
        return False
    return classification.calibration.confidence_discriminative


def _gate(name: str, passed: bool, observed: str, requirement: str, blocking: bool = True) -> GateResult:
    return GateResult(name=name, passed=passed, blocking=blocking, observed=observed, requirement=requirement)


def build_gates(validation: ValidationReport, quality: QualityReport | None, classification: ClassificationReport | None, thresholds: PolicyThresholds = DEFAULT_THRESHOLDS) -> tuple[GateResult, ...]:
    discriminating = confidence_is_trustworthy(classification)
    gates = [
        _gate("input_valid", validation.status == "valid", validation.status, "valid EEG channels and event markers"),
        _gate("signal_quality", quality is not None and quality.status != "poor", "quality unavailable" if quality is None else quality.status, "quality better than poor"),
        _gate("classification_available", classification is not None and classification.status == "classified", "unavailable" if classification is None else classification.status, "a motor-imagery prediction"),
        _gate("enough_trials", bool(classification and classification.trial_count >= thresholds.minimum_trials), "0 trials" if classification is None else f"{classification.trial_count} trials", f"at least {thresholds.minimum_trials} trials"),
        _gate("trial_agreement", bool(classification and classification.agreement is not None and classification.agreement >= thresholds.minimum_agreement), "no prediction" if not classification or classification.agreement is None else f"{classification.agreement:.0%} agreement", f"at least {thresholds.minimum_agreement:.0%} of trials agree", blocking=False),
        _gate("confidence_discriminative", discriminating, "no calibration evidence" if classification is None or classification.calibration is None else str(discriminating), "held-out confidence must separate correct from incorrect trials", blocking=False),
    ]
    return tuple(gates)


def build_signals(validation: ValidationReport, quality: QualityReport | None, classification: ClassificationReport | None) -> tuple[str, ...]:
    """Plain-language observations a reviewer can reason over without re-deriving anything."""
    signals = [f"Recording has {validation.eeg_channel_count} EEG channels and {validation.event_count} event markers."]
    if quality is not None:
        signals.extend(quality.warnings)
        signals.append(f"Peak-to-peak amplitude {quality.peak_to_peak_uv} uV with {quality.flat_channel_count} flat channel(s); {quality.rejected_epoch_ratio or 0:.0%} of epochs exceed the amplitude limit.")
    if classification is None:
        signals.append("No classification was attempted.")
        return tuple(signals)
    if classification.status == "unavailable":
        signals.append(f"No prediction was produced: {classification.reason}")
        return tuple(signals)
    signals.append(f"{classification.model} predicted {classification.predicted_label} from {classification.trial_count} trials with {(classification.agreement or 0):.0%} agreement.")
    calibration = classification.calibration
    if calibration is None:
        signals.append("No calibration evidence: confidence is reported uncalibrated and must not be treated as a reliability estimate.")
    else:
        band = f"{calibration.band} band scored {calibration.band_observed_accuracy} on {calibration.band_sample_size} held-out trials"
        note = "Confidence separates correct from incorrect predictions." if calibration.confidence_discriminative else "Confidence does NOT separate correct from incorrect predictions, so it cannot justify acceptance."
        signals.append(f"Probability is temperature-calibrated (T={calibration.temperature}); {band}. {note}")
    return tuple(signals)


def decide(validation: ValidationReport, quality: QualityReport | None, classification: ClassificationReport | None, recheck: RecheckState | None = None, thresholds: PolicyThresholds = DEFAULT_THRESHOLDS) -> Decision:
    """The verdict, in strict order of precedence. Every branch is reachable."""
    state = recheck or RecheckState(limit=thresholds.recheck_limit)
    if validation.status != "valid":
        return "REJECT_INVALID_INPUT"
    if quality is None or quality.status == "poor":
        return "RECHECK_QUALITY" if state.remaining > 0 else "REQUEST_HUMAN_REVIEW"
    if classification is None or classification.status != "classified":
        return "RETURN_UNCERTAIN"
    if classification.trial_count < thresholds.minimum_trials:
        return "RETURN_UNCERTAIN"
    if quality.status == "warning":
        return "ACCEPT_WITH_WARNING"
    if classification.agreement is not None and classification.agreement < thresholds.minimum_agreement:
        return "ACCEPT_WITH_WARNING"
    if not confidence_is_trustworthy(classification):
        return "ACCEPT_WITH_WARNING"
    return "ACCEPT"


def evaluate(validation: ValidationReport, quality: QualityReport | None, classification: ClassificationReport | None, recheck: RecheckState | None = None, thresholds: PolicyThresholds = DEFAULT_THRESHOLDS) -> PolicyOutcome:
    state = recheck or RecheckState(limit=thresholds.recheck_limit)
    decision = decide(validation, quality, classification, state, thresholds)
    return PolicyOutcome(
        decision=decision,
        gates=build_gates(validation, quality, classification, thresholds),
        signals=build_signals(validation, quality, classification),
        allowed_actions=ALLOWED_ACTIONS[decision],
        recheck=state,
        thresholds=thresholds,
    )
