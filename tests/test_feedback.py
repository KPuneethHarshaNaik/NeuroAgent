import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from backend.feedback import save_reviewer_feedback
from backend.schemas import HumanReview, JobReport, ValidationReport


class FeedbackTests(unittest.TestCase):
    def test_reviewer_feedback_preserves_the_input_hash(self) -> None:
        report = JobReport(job_id="job-1", status="invalid_input", input_filename="sample.fif", input_hash="hash-1", validation=ValidationReport(status="invalid", file_format="fif"), human_review=HumanReview(action="mark_uncertain", reviewer_comment="Noisy", approved_label="uncertain", created_at="2026-09-20T00:00:00Z"))
        path = Path("runtime") / "feedback.jsonl"
        with patch.object(Path, "open", mock_open()) as opened:
            save_reviewer_feedback(path, report)
            written = opened.return_value.__enter__.return_value.write.call_args.args[0]
            self.assertEqual(json.loads(written)["input_hash"], "hash-1")
