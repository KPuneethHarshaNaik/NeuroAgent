import unittest

from backend.agents import decision_agent, update
from backend.schemas import ValidationReport


class AgentTests(unittest.TestCase):
    def test_agents_emit_structured_factual_updates(self) -> None:
        event = update("signal", "completed", "Created 12 epochs.", {"epochs": 12})
        self.assertEqual(event.agent, "signal")
        self.assertEqual(event.findings["epochs"], 12)

    def test_decision_agent_cannot_accept_invalid_input(self) -> None:
        validation = ValidationReport(status="invalid", file_format="fif")
        outcome, event = decision_agent(validation, None, None)
        self.assertEqual(outcome.decision, "REJECT_INVALID_INPUT")
        self.assertEqual(event.findings["decision"], "REJECT_INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()
