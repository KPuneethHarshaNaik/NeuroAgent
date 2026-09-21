"""Unit and integration tests for backend/graph.py (LangGraph orchestration)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np

from backend.graph import (
    PipelineState,
    RunContext,
    build_graph,
    run_pipeline,
    stream_pipeline,
)
from backend.main import legacy_process_job, process_job
from backend.schemas import AgentUpdate, JobReport


def create_dummy_fif(
    filepath: Path,
    num_channels: int = 64,
    sfreq: float = 160.0,
    duration: float = 10.0,
    with_events: bool = True,
    event_type: str = "T1",
) -> None:
    """Helper to generate dummy MNE raw FIF file for testing."""
    ch_names = [f"EEG {i+1:03d}" for i in range(num_channels)]
    ch_types = ["eeg"] * num_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.random.randn(num_channels, int(sfreq * duration)) * 1e-5
    raw = mne.io.RawArray(data, info)

    if with_events:
        # Add annotations
        onset = [1.0, 4.0, 7.0]
        duration_annot = [2.0, 2.0, 2.0]
        description = [event_type] * 3
        annotations = mne.Annotations(onset=onset, duration=duration_annot, description=description)
        raw.set_annotations(annotations)

    raw.save(filepath, overwrite=True)


class LangGraphPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create test FIF files
        self.valid_fif = self.dir_path / "valid_eeg.fif"
        create_dummy_fif(self.valid_fif, with_events=True, event_type="T1")

        self.no_events_fif = self.dir_path / "no_events_eeg.fif"
        create_dummy_fif(self.no_events_fif, with_events=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_graph_shape_and_mermaid(self) -> None:
        """Assert graph topology matches expected mermaid nodes and structure."""
        graph = build_graph(mode="rules")
        mermaid = graph.get_graph().draw_mermaid()

        self.assertIn("validation", mermaid)
        self.assertIn("signal", mermaid)
        self.assertIn("prediction", mermaid)
        self.assertIn("decision", mermaid)
        self.assertIn("report", mermaid)

    def test_parity_between_legacy_and_langgraph(self) -> None:
        """Assert JobReport from legacy_process_job matches run_pipeline, ignoring generated IDs and timestamps."""
        job_id = "test-parity-123"
        filename = "valid_eeg.fif"
        digest = "sha256-dummy-hash"

        legacy_report = legacy_process_job(job_id, filename, self.valid_fif, digest)
        graph_report = run_pipeline(job_id, filename, self.valid_fif, digest)

        self.assertEqual(legacy_report.status, graph_report.status)
        self.assertEqual(legacy_report.decision, graph_report.decision)
        self.assertEqual(legacy_report.validation.status, graph_report.validation.status)
        self.assertEqual(legacy_report.validation.eeg_channel_count, graph_report.validation.eeg_channel_count)
        if legacy_report.quality and graph_report.quality:
            self.assertEqual(legacy_report.quality.status, graph_report.quality.status)

    def test_routing_invalid_file_skips_signal_and_prediction(self) -> None:
        """Assert an invalid recording routes directly to decision and report, skipping signal and prediction."""
        job_id = "test-invalid-route"
        filename = "no_events_eeg.fif"
        digest = "sha256-dummy-hash"

        with patch("backend.graph.signal_node") as mock_signal, patch("backend.graph.prediction_node") as mock_pred:
            report = run_pipeline(job_id, filename, self.no_events_fif, digest)
            mock_signal.assert_not_called()
            mock_pred.assert_not_called()

        self.assertEqual(report.status, "invalid_input")
        self.assertEqual(report.decision, "REJECT_INVALID_INPUT")

    def test_failure_routing_when_stage_raises(self) -> None:
        """Assert an exception in a stage sets status='failed' and routes cleanly to decision and report without crashing."""
        job_id = "test-stage-failure"
        filename = "valid_eeg.fif"
        digest = "sha256-dummy-hash"

        with patch("backend.graph.preprocess", side_effect=RuntimeError("Signal processing corrupted")):
            report = run_pipeline(job_id, filename, self.valid_fif, digest)

        self.assertEqual(report.status, "processing_failed")
        self.assertTrue(any("Signal processing corrupted" in w or "failed" in w for w in report.warnings))

        # Check agent_updates contains a failure update for signal
        failed_updates = [u for u in report.agent_updates if u.agent == "signal" and u.status == "failed"]
        self.assertEqual(len(failed_updates), 1)

    def test_reducers_accumulate_agent_updates_and_audit_events_in_order(self) -> None:
        """Assert agent_updates and audit_events accumulate entries sequentially across nodes."""
        job_id = "test-reducers"
        filename = "valid_eeg.fif"
        digest = "sha256-dummy-hash"

        report = run_pipeline(job_id, filename, self.valid_fif, digest)

        agents_in_order = [u.agent for u in report.agent_updates if u.status in ("completed", "blocked", "failed")]
        expected_agents = ["validation", "signal", "prediction", "decision", "report"]
        self.assertEqual(agents_in_order, expected_agents)

        self.assertGreater(len(report.audit_events), 0)

    def test_streaming_yields_working_and_completed_updates_in_order(self) -> None:
        """Assert stream_pipeline yields progressive AgentUpdates and ends with JobReport."""
        job_id = "test-streaming"
        filename = "valid_eeg.fif"
        digest = "sha256-dummy-hash"

        items = list(stream_pipeline(job_id, filename, self.valid_fif, digest))
        agent_updates = [item for item in items if isinstance(item, AgentUpdate)]
        final_reports = [item for item in items if isinstance(item, JobReport)]

        self.assertEqual(len(final_reports), 1)
        self.assertGreater(len(agent_updates), 0)

        statuses = [u.status for u in agent_updates]
        self.assertIn("working", statuses)
        self.assertIn("completed", statuses)

        # Last item yielded must be JobReport
        self.assertIsInstance(items[-1], JobReport)


if __name__ == "__main__":
    unittest.main()
