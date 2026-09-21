"""Build the single evidence artifact the reviewer agent consumes.

The bundle exists so an agent never has to reassemble facts from sibling fields, and so
the thresholds it is being judged against travel with the data. Provenance is *derived*
here, not accepted as an argument, which is what stops a caller from misreporting it.
"""

from __future__ import annotations

from backend.policy import POLICY_VERSION, PolicyOutcome
from backend.schemas import ClassificationReport, EvidenceBundle, QualityReport, ValidationReport

PIPELINE_VERSION = "0.2.0"
EVIDENCE_SCHEMA_VERSION = "1.0"


def _measurements(validation: ValidationReport, epoching: dict, quality: QualityReport | None, classification: ClassificationReport | None) -> dict:
    measurements: dict[str, float | int | bool | str | None] = {
        "sampling_rate_hz": validation.sampling_rate,
        "channel_count": validation.channel_count,
        "eeg_channel_count": validation.eeg_channel_count,
        "duration_seconds": validation.duration_seconds,
        "event_count": validation.event_count,
        "epoch_count": epoching.get("epoch_count"),
    }
    if quality is not None:
        measurements.update(
            {
                "peak_to_peak_uv": quality.peak_to_peak_uv,
                "flat_channel_count": quality.flat_channel_count,
                "rejected_epoch_ratio": quality.rejected_epoch_ratio,
            }
        )
    if classification is not None:
        measurements.update(
            {
                "trial_count": classification.trial_count,
                "agreement": classification.agreement,
                "predicted_label": classification.predicted_label,
                "failure_code": classification.reason_code,
            }
        )
        for label, probability in classification.probabilities.items():
            measurements[f"probability_{label}"] = probability
        if classification.calibration is not None:
            measurements.update(
                {
                    "calibration_temperature": classification.calibration.temperature,
                    "confidence_band": classification.calibration.band,
                    "confidence_discriminative": classification.calibration.confidence_discriminative,
                    "band_observed_accuracy": classification.calibration.band_observed_accuracy,
                    "band_sample_size": classification.calibration.band_sample_size,
                }
            )
    return {key: value for key, value in measurements.items() if value is not None}


def build_evidence(
    job_id: str,
    filename: str,
    digest: str,
    validation: ValidationReport,
    preprocessing: dict,
    epoching: dict,
    quality: QualityReport | None,
    classification: ClassificationReport | None,
    outcome: PolicyOutcome,
) -> EvidenceBundle:
    provenance = {
        "job_id": job_id,
        "input_filename": filename,
        "input_hash": digest,
        "pipeline_version": PIPELINE_VERSION,
        "policy_version": POLICY_VERSION,
        "model": None if classification is None else classification.model,
        "model_temperature": None if classification is None or classification.calibration is None else classification.calibration.temperature,
        "preprocessing": preprocessing,
        "epoch_seconds": epoching.get("window_seconds"),
        "thresholds": outcome.thresholds.as_dict(),
    }
    return EvidenceBundle(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        provenance=provenance,
        measurements=_measurements(validation, epoching, quality, classification),
        signals=list(outcome.signals),
        gates=list(outcome.gates),
        recheck_budget=outcome.recheck.as_dict(),
        allowed_actions=list(outcome.allowed_actions),
    )
