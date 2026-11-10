from types import SimpleNamespace
import unittest
from unittest.mock import patch

import app
from agno.run.team import TeamRunEvent


class RunResearchStreamingTests(unittest.TestCase):
    def test_streams_accumulated_team_content_only(self):
        member_event = SimpleNamespace(event=object(), content="member notes")
        team_events = [
            member_event,
            SimpleNamespace(event=TeamRunEvent.run_content, content="First "),
            SimpleNamespace(event=TeamRunEvent.run_content, content="answer"),
        ]

        with patch.object(app.research_team, "run", return_value=iter(team_events)) as run:
            updates = list(app.run_research("  research question  "))

        self.assertEqual(
            updates,
            [
                ("_Researching — this may take a minute..._", "Running..."),
                ("First ", "Running..."),
                ("First answer", "Running..."),
                ("First answer", "Done."),
            ],
        )
        run.assert_called_once_with("research question", stream=True)

    def test_empty_query_does_not_start_a_run(self):
        with patch.object(app.research_team, "run") as run:
            updates = list(app.run_research("   "))

        self.assertEqual(updates, [(app.PLACEHOLDER, "")])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
