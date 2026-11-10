import unittest

from session_results import PIPELINE_RESULT_KEYS, clear_pipeline_results


class ClearPipelineResultsTests(unittest.TestCase):
    def test_removes_only_pipeline_outputs(self):
        session_state = {
            "plan": "old plan",
            "draft_code": "old draft",
            "review": "old review",
            "final_code": "old final",
            "task_input": "new task",
            "copy_triggered": True,
        }

        clear_pipeline_results(session_state)

        self.assertTrue(all(key not in session_state for key in PIPELINE_RESULT_KEYS))
        self.assertEqual(session_state, {"task_input": "new task", "copy_triggered": True})

    def test_handles_missing_pipeline_outputs(self):
        session_state = {"task_input": "new task"}

        clear_pipeline_results(session_state)

        self.assertEqual(session_state, {"task_input": "new task"})


if __name__ == "__main__":
    unittest.main()
