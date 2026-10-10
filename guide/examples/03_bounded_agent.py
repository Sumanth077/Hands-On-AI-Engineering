"""허용 도구와 단계 예산을 코드로 강제하는 작은 에이전트 루프."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass


KNOWLEDGE = {"rag": "검색 증강 생성", "ocr": "광학 문자 인식"}


def lookup(term: str) -> str:
    return KNOWLEDGE.get(term.lower(), "알 수 없는 용어")


def calculate(expression: str) -> str:
    """eval 대신 허용한 산술 노드만 해석해 임의 코드 실행을 막는다."""
    operations: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("허용되지 않은 계산식입니다.")

    return str(visit(ast.parse(expression, mode="eval")))


TOOLS: dict[str, Callable[[str], str]] = {"lookup": lookup, "calculate": calculate}


@dataclass(frozen=True)
class Action:
    tool: str
    argument: str


def run(actions: list[Action], max_steps: int = 3) -> list[str]:
    observations: list[str] = []
    for step, action in enumerate(actions, start=1):
        if step > max_steps:
            observations.append("중단: 단계 예산 초과")
            break
        tool = TOOLS.get(action.tool)
        if tool is None:
            observations.append(f"거부: 허용되지 않은 도구 {action.tool}")
            continue
        try:
            observations.append(f"{action.tool}: {tool(action.argument)}")
        except (SyntaxError, ValueError, ZeroDivisionError) as error:
            observations.append(f"실패: {error}")
    return observations


if __name__ == "__main__":
    plan = [
        Action("lookup", "RAG"),
        Action("calculate", "(12 + 8) / 2"),
        Action("shell", "delete everything"),
        Action("lookup", "OCR"),
    ]
    print("\n".join(run(plan)))
