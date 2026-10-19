"""Data models exchanged by the repository reader, agent, and interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


@dataclass(frozen=True)
class RepositorySnapshot:
    name: str
    origin: str
    default_branch: str
    description: str
    file_paths: list[str]
    readme: str = ""


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: str

    @property
    def line_count(self) -> int:
        return len(self.content.splitlines())


@dataclass(frozen=True)
class GuideAnswer:
    question: str
    answer: str
    spoken_answer: str = ""
    files: list[SourceFile] = field(default_factory=list)
    selected_paths: list[str] = field(default_factory=list)
    plan: str = ""


@dataclass(frozen=True)
class TimestampSegment:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class VoiceResult:
    audio: bytes
    segments: list[TimestampSegment] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    segments: list[TimestampSegment] = field(default_factory=list)


class GuideState(TypedDict, total=False):
    question: str
    snapshot: RepositorySnapshot
    plan: str
    selected_paths: list[str]
    files: list[SourceFile]
    answer: str

