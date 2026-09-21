"""Durable reviewer corrections, ready to become labelled training data after audit."""

from __future__ import annotations

import json
from pathlib import Path

from backend.schemas import JobReport


def save_reviewer_feedback(path: Path, report: JobReport) -> None:
    """Append only; original model output is preserved alongside the human correction."""
    if report.human_review is None:
        raise ValueError("A reviewer decision is required before feedback can be saved.")
    record = {
        "job_id": report.job_id,
        "input_hash": report.input_hash,
        "automated_label": report.classification.predicted_label if report.classification else None,
        "review_action": report.human_review.action,
        "reviewed_label": report.human_review.approved_label,
        "comment": report.human_review.reviewer_comment,
        "created_at": report.human_review.created_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
