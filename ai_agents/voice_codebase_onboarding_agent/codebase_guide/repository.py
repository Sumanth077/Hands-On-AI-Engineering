"""Read-only GitHub and local repository readers."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from urllib.parse import quote

import httpx

from codebase_guide.models import RepositorySnapshot, SourceFile


GITHUB_URL = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")
MAX_FILE_BYTES = 70_000
MAX_FILES = 1_500
CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt", ".rb",
    ".php", ".cs", ".cpp", ".c", ".h", ".html", ".css", ".scss", ".sql", ".sh",
    ".yaml", ".yml", ".toml", ".json", ".md",
}
IMPORTANT_NAMES = {
    "README.md", "pyproject.toml", "package.json", "requirements.txt", "Dockerfile",
    "docker-compose.yml", "compose.yml", "main.py", "app.py", "server.py", "manage.py",
}
IGNORED_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".next", "dist", "build"}


class RepositoryError(RuntimeError):
    """A repository could not be inspected safely."""


def parse_github_url(url: str) -> tuple[str, str]:
    match = GITHUB_URL.match(url.strip())
    if not match:
        raise RepositoryError("Enter a public GitHub repository URL such as https://github.com/owner/repository.")
    return match.group(1), match.group(2)


def is_candidate_path(path: str) -> bool:
    parts = Path(path).parts
    return not any(part in IGNORED_PARTS for part in parts) and (
        Path(path).suffix.lower() in CODE_EXTENSIONS or Path(path).name in IMPORTANT_NAMES
    )


def rank_paths(paths: list[str], question: str, limit: int = 6) -> list[str]:
    """Choose useful files without requiring a model, used as a safe fallback."""
    terms = {term.lower() for term in re.findall(r"[a-zA-Z_]{3,}", question)}
    broad_question = bool({"main", "entry", "project", "architecture", "overview"} & terms)

    def score(path: str) -> tuple[int, int, str]:
        lowered = path.lower()
        term_hits = sum(term in lowered for term in terms)
        name = Path(path).name
        important = int(name in IMPORTANT_NAMES)
        root_file = int(len(Path(path).parts) == 1)
        broad_context = 5 if broad_question and root_file and name in IMPORTANT_NAMES else 0
        shallow = -len(Path(path).parts)
        return (term_hits + important + root_file + broad_context, shallow, path)

    return sorted(paths, key=score, reverse=True)[:limit]


class GitHubRepository:
    def __init__(self, url: str) -> None:
        self.owner, self.repo = parse_github_url(url)
        self.url = f"https://github.com/{self.owner}/{self.repo}"
        token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _get(self, endpoint: str) -> dict:
        try:
            response = httpx.get(
                f"https://api.github.com{endpoint}", headers=self.headers, timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                raise RepositoryError("GitHub could not find that public repository.") from error
            raise RepositoryError(f"GitHub request failed: {error.response.status_code}.") from error
        except httpx.HTTPError as error:
            raise RepositoryError(f"Could not reach GitHub: {error}") from error

    def snapshot(self) -> RepositorySnapshot:
        metadata = self._get(f"/repos/{self.owner}/{self.repo}")
        branch = str(metadata["default_branch"])
        tree = self._get(f"/repos/{self.owner}/{self.repo}/git/trees/{quote(branch, safe='')}?recursive=1")
        paths = [
            item["path"]
            for item in tree.get("tree", [])
            if item.get("type") == "blob" and is_candidate_path(str(item.get("path", "")))
        ][:MAX_FILES]
        readme = ""
        if "README.md" in paths:
            try:
                readme = self.read_file("README.md", branch).content
            except RepositoryError:
                pass
        return RepositorySnapshot(
            name=str(metadata.get("full_name", f"{self.owner}/{self.repo}")),
            origin=self.url,
            default_branch=branch,
            description=str(metadata.get("description") or "No repository description provided."),
            file_paths=paths,
            readme=readme,
        )

    def read_file(self, path: str, branch: str) -> SourceFile:
        payload = self._get(
            f"/repos/{self.owner}/{self.repo}/contents/{quote(path, safe='/')}?ref={quote(branch, safe='')}"
        )
        if payload.get("encoding") != "base64":
            raise RepositoryError(f"GitHub returned an unsupported format for {path}.")
        raw = base64.b64decode(payload.get("content", ""))
        if len(raw) > MAX_FILE_BYTES:
            raise RepositoryError(f"{path} is too large to inspect safely in this demo.")
        return SourceFile(path=path, content=raw.decode("utf-8", errors="replace"))

    def read_files(self, paths: list[str], branch: str) -> list[SourceFile]:
        files: list[SourceFile] = []
        for path in paths[:6]:
            try:
                files.append(self.read_file(path, branch))
            except RepositoryError:
                continue
        return files


class LocalRepository:
    def __init__(self, raw_path: str) -> None:
        self.root = Path(raw_path).expanduser().resolve()
        if not self.root.is_dir():
            raise RepositoryError("Enter an existing local repository directory.")
        if not (self.root / ".git").exists():
            raise RepositoryError("The selected folder is not a Git repository.")

    def snapshot(self) -> RepositorySnapshot:
        paths: list[str] = []
        for current_root, directories, filenames in os.walk(self.root):
            directories[:] = [directory for directory in directories if directory not in IGNORED_PARTS]
            for filename in filenames:
                relative = (Path(current_root) / filename).relative_to(self.root).as_posix()
                if is_candidate_path(relative):
                    paths.append(relative)
                if len(paths) >= MAX_FILES:
                    break
            if len(paths) >= MAX_FILES:
                break
        readme = ""
        if "README.md" in paths:
            readme = self.read_file("README.md", "local").content
        return RepositorySnapshot(
            name=self.root.name,
            origin=str(self.root),
            default_branch="local working tree",
            description="Local repository inspected read-only.",
            file_paths=paths,
            readme=readme,
        )

    def read_file(self, path: str, branch: str) -> SourceFile:
        target = (self.root / path).resolve()
        if self.root not in target.parents or not target.is_file():
            raise RepositoryError("The requested path is outside the selected repository.")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise RepositoryError(f"{path} is too large to inspect safely in this demo.")
        return SourceFile(path=path, content=target.read_text(encoding="utf-8", errors="replace"))

    def read_files(self, paths: list[str], branch: str) -> list[SourceFile]:
        files: list[SourceFile] = []
        for path in paths[:6]:
            try:
                files.append(self.read_file(path, branch))
            except RepositoryError:
                continue
        return files

