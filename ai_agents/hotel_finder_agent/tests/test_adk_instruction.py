import ast
import unittest
from pathlib import Path


class AdkInstructionTests(unittest.TestCase):
    def test_instruction_includes_children_in_guest_count(self) -> None:
        agent_path = Path(__file__).parents[1] / "hotel_finder_agent" / "agent.py"
        tree = ast.parse(agent_path.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "INSTRUCTION"
                for target in node.targets
            )
        )
        instruction = "".join(
            part.value
            for part in assignment.value.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )

        self.assertIn("Number of adults, children, and rooms", instruction)


if __name__ == "__main__":
    unittest.main()
