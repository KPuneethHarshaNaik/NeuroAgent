"""Small, auditable agents that each own one EEG review responsibility."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter

from backend.eeg import epoch_and_measure, preprocess, validate
from backend.inference import classify
from backend.policy import PolicyOutcome, evaluate
from backend.schemas import AgentUpdate, ClassificationReport, QualityReport, ValidationReport


def update(agent: str, status: str, message: str, findings: dict | None = None, started_at: float | None = None) -> AgentUpdate:
    return AgentUpdate(
        agent=agent,
        status=status,
        message=message,
        findings=findings or {},
        timestamp=datetime.now(UTC).isoformat(),
        duration_ms=round((perf_counter() - started_at) * 1_000) if started_at is not None else 0,
    )


def failure_update(agent: str, error: Exception) -> AgentUpdate:
    return update(agent, "failed", f"{type(error).__name__}: {error}")


def validation_agent(raw: object, extension: str) -> tuple[ValidationReport, AgentUpdate]:
    started_at = perf_counter()
    report = validate(raw, extension)
    message = "EEG channels and event markers are ready." if report.status == "valid" else "The recording cannot be used for trial-based analysis."
    return report, update("validation", "completed" if report.status == "valid" else "blocked", message, {"eeg_channels": report.eeg_channel_count, "event_markers": report.event_count}, started_at)


def signal_agent(raw: object) -> tuple[object, dict, dict, QualityReport, AgentUpdate]:
    started_at = perf_counter()
    cleaned, preprocessing = preprocess(raw)
    epochs, quality = epoch_and_measure(cleaned)
    message = f"Created {epochs['epoch_count']} epochs; signal quality is {quality.status}."
    return cleaned, preprocessing, epochs, quality, update("signal", "completed", message, {"epochs": epochs["epoch_count"], "quality": quality.status, "rejected_epoch_ratio": quality.rejected_epoch_ratio}, started_at)


def prediction_agent(raw: object) -> tuple[ClassificationReport, AgentUpdate]:
    started_at = perf_counter()
    report = classify(raw)
    if report.status == "classified":
        message = f"Predicted {report.predicted_label} from {report.trial_count} trials."
        findings = {"prediction": report.predicted_label, "agreement": report.agreement, "trials": report.trial_count}
        return report, update("prediction", "completed", message, findings, started_at)
    return report, update("prediction", "blocked", report.reason or "Prediction is unavailable.", {"reason_code": report.reason_code}, started_at)


def decision_agent(validation: ValidationReport, quality: QualityReport | None, classification: ClassificationReport | None) -> tuple[PolicyOutcome, AgentUpdate]:
    started_at = perf_counter()
    outcome = evaluate(validation, quality, classification)
    return outcome, update("decision", "completed", f"Policy decision: {outcome.decision.replace('_', ' ').lower()}.", {"decision": outcome.decision, "allowed_actions": list(outcome.allowed_actions)}, started_at)


def report_agent(decision: str) -> AgentUpdate:
    return update("report", "completed", "Evidence bundle prepared for reviewer.", {"decision": decision})
