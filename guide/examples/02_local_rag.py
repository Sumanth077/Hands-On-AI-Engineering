"""외부 패키지 없이 검색과 근거 답변의 경계를 보여주는 최소 RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


DOCUMENTS = [
    Document("rag", "RAG는 질문과 관련된 문서를 검색해 모델의 답변 근거로 제공한다."),
    Document("agent", "에이전트는 목표를 위해 도구를 선택하고 결과를 관찰하며 작업을 반복한다."),
    Document("security", "외부 부작용이 있는 도구는 최소 권한과 명시적 사용자 승인이 필요하다."),
]


def tokens(text: str) -> set[str]:
    """비교 과정을 눈으로 확인할 수 있도록 단순한 영문·한글 토큰화를 사용한다."""
    return set(re.findall(r"[0-9A-Za-z가-힣]+", text.lower()))


def retrieve(query: str, limit: int = 2) -> list[tuple[int, Document]]:
    query_tokens = tokens(query)
    ranked = [(len(query_tokens & tokens(document.text)), document) for document in DOCUMENTS]
    return [(score, document) for score, document in sorted(ranked, reverse=True, key=lambda x: x[0])[:limit] if score]


def answer(query: str) -> str:
    matches = retrieve(query)
    if not matches:
        return "관련 근거를 찾지 못해 답변할 수 없습니다."
    evidence = " ".join(f"[{document.document_id}] {document.text}" for _, document in matches)
    return f"검색된 근거: {evidence}"


if __name__ == "__main__":
    question = "에이전트 도구에는 왜 사용자 승인이 필요한가?"
    print(f"질문: {question}")
    print(answer(question))
