"""하위 AI 프로젝트가 최소 실행 문서를 갖췄는지 검사한다.

실행: python 01_project_audit.py <project-directory>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def audit(project_dir: Path) -> list[str]:
    """문서, 의존성, 진입점과 환경 변수 표본을 확인한다."""
    issues: list[str] = []
    if not (project_dir / "README.md").is_file():
        issues.append("README.md가 없습니다.")

    dependency_files = [project_dir / "requirements.txt", project_dir / "pyproject.toml"]
    if not any(path.is_file() for path in dependency_files):
        issues.append("requirements.txt 또는 pyproject.toml이 없습니다.")

    entrypoints = [project_dir / "app.py", project_dir / "main.py"]
    if not any(path.is_file() for path in entrypoints):
        issues.append("일반적인 진입점(app.py 또는 main.py)을 찾지 못했습니다. README를 확인하세요.")

    if not (project_dir / ".env.example").is_file():
        issues.append(".env.example이 없습니다. API 키가 필요 없는 프로젝트인지 확인하세요.")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="AI 프로젝트 기본 구성 감사")
    parser.add_argument("project_dir", type=Path)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()

    if not project_dir.is_dir():
        parser.error(f"디렉터리가 아닙니다: {project_dir}")

    issues = audit(project_dir)
    print(f"검사 대상: {project_dir}")
    if not issues:
        print("기본 구성 검사를 통과했습니다.")
        return 0

    for issue in issues:
        print(f"- {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
