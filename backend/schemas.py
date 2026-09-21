from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ValidationReport(BaseModel):
    status: Literal["valid", "invalid"]
    file_format: str
    sampling_rate: float | None = None
    channel_count: int = 0
    eeg_channel_count: int = 0
    duration_seconds: float | None = None
    has_events: bool = False
    event_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    status: Literal["acceptable", "warning", "poor"]
    peak_to_peak_uv: float | None = None
    flat_channel_count: int = 0
    rejected_epoch_ratio: float = 0.0
    warnings: list[str] = Field(default_factory=list)


FailureCode = Literal["no_model", "no_annotations", "montage_mismatch", "trial_window", "unknown"]


class CalibrationInfo(BaseModel):
    """What a held-out split says about the probability we are reporting."""

    method: str
    temperature: float
    evaluated_on: str
    confidence_discriminative: bool
    band: Literal["low", "medium", "high"]
    band_observed_accuracy: float | None = None
    band_sample_size: int | None = None


class ClassificationReport(BaseModel):
    status: Literal["classified", "unavailable"]
    model: str
    reason_code: FailureCode | None = None
    reason: str | None = None
    predicted_label: Literal["left_hand", "right_hand"] | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    vote_counts: dict[str, int] = Field(default_factory=dict)
    agreement: float | None = None
    trial_count: int = 0
    calibration: CalibrationInfo | None = None
    warnings: list[str] = Field(default_factory=list)


Decision = Literal["ACCEPT", "ACCEPT_WITH_WARNING", "RECHECK_QUALITY", "RETURN_UNCERTAIN", "REQUEST_HUMAN_REVIEW", "REJECT_INVALID_INPUT"]


class GateResult(BaseModel):
    """One deterministic acceptance rule and what it saw."""

    name: str
    passed: bool
    blocking: bool
    observed: str
    requirement: str


class EvidenceBundle(BaseModel):
    """The single artifact the reviewer agent consumes, with provenance it cannot alter."""

    schema_version: str
    provenance: dict = Field(default_factory=dict)
    measurements: dict = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)
    recheck_budget: dict = Field(default_factory=dict)
    allowed_actions: list[str] = Field(default_factory=list)


ReviewAction = Literal["approve", "mark_uncertain", "override"]


class HumanReviewRequest(BaseModel):
    """A human decision recorded separately from the automated policy verdict."""

    action: ReviewAction
    reviewer_comment: str = Field(default="", max_length=2_000)
    approved_label: Literal["left_hand", "right_hand", "uncertain"] | None = None


class HumanReview(BaseModel):
    action: ReviewAction
    reviewer_comment: str
    approved_label: Literal["left_hand", "right_hand", "uncertain"]
    created_at: str


class AgentUpdate(BaseModel):
    """A visible, factual status update from one bounded pipeline agent."""

    agent: Literal["validation", "signal", "prediction", "decision", "report"]
    status: Literal["working", "completed", "blocked", "failed"]
    message: str
    findings: dict = Field(default_factory=dict)
    timestamp: str
    duration_ms: int = 0


class JobReport(BaseModel):
    job_id: str
    status: Literal["completed", "invalid_input", "processing_failed"]
    input_filename: str
    input_hash: str
    validation: ValidationReport
    decision: Decision | None = None
    preprocessing: dict = Field(default_factory=dict)
    epoching: dict = Field(default_factory=dict)
    quality: QualityReport | None = None
    classification: ClassificationReport | None = None
    evidence: EvidenceBundle | None = None
    human_review: HumanReview | None = None
    warnings: list[str] = Field(default_factory=list)
    audit_events: list[dict] = Field(default_factory=list)
    agent_updates: list[AgentUpdate] = Field(default_factory=list)


class LiveJobStatus(BaseModel):
    """A short-lived view for the live review workspace while a job runs."""

    job_id: str
    status: Literal["processing", "completed", "invalid_input", "processing_failed"]
    agent_updates: list[AgentUpdate] = Field(default_factory=list)
    report: JobReport | None = None
