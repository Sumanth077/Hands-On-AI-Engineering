"""GitHub tools the agent can call, plus the OpenAI-style schemas that
describe them to the model.

v1 scope: one fixed repo (GITHUB_REPO, "owner/name") and a personal access
token (GITHUB_TOKEN). Enough to run "check recent changes, flag issues,
summarize" end to end. Letting the spoken instruction name a different repo
is a natural follow-up once this is working.
"""
import os

import requests

API_ROOT = "https://api.github.com"


def _repo() -> str:
    repo = os.environ.get("GITHUB_REPO")
    if not repo:
        raise RuntimeError("GITHUB_REPO is not set (expected 'owner/name'). Check your .env file.")
    return repo


def _headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set. Check your .env file.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _request(method: str, path: str, **kwargs) -> dict:
    response = requests.request(method, f"{API_ROOT}{path}", headers=_headers(), timeout=30, **kwargs)
    if not response.ok:
        raise RuntimeError(f"GitHub API {method} {path} failed ({response.status_code}): {response.text[:300]}")
    return response.json() if response.text else {}


def list_recent_commits(count: int = 5) -> dict:
    """List the most recent commits on the repo's default branch."""
    count = max(1, min(count, 20))
    data = _request("GET", f"/repos/{_repo()}/commits", params={"per_page": count})
    return {
        "commits": [
            {
                "sha": c["sha"][:10],
                "message": c["commit"]["message"].splitlines()[0],
                "author": (c["commit"]["author"] or {}).get("name"),
                "date": (c["commit"]["author"] or {}).get("date"),
                "url": c["html_url"],
            }
            for c in data
        ]
    }


def get_commit_diff(sha: str) -> dict:
    """Get the file-level diff for one commit, to see what actually changed."""
    data = _request("GET", f"/repos/{_repo()}/commits/{sha}")
    files = data.get("files", [])
    return {
        "sha": sha,
        "message": data.get("commit", {}).get("message", ""),
        "files": [
            {
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                # Truncate long patches so a big diff doesn't blow the context budget.
                "patch": (f.get("patch") or "")[:4000],
            }
            for f in files
        ],
    }


def list_open_issues(count: int = 10) -> dict:
    """List open issues (pull requests are excluded)."""
    count = max(1, min(count, 30))
    data = _request("GET", f"/repos/{_repo()}/issues", params={"state": "open", "per_page": count})
    return {
        "issues": [
            {"number": i["number"], "title": i["title"], "url": i["html_url"]}
            for i in data
            if "pull_request" not in i
        ]
    }


def create_issue(title: str, body: str = "") -> dict:
    """Open a new GitHub issue."""
    data = _request("POST", f"/repos/{_repo()}/issues", json={"title": title, "body": body})
    return {"number": data["number"], "url": data["html_url"]}


def add_comment(issue_number: int, body: str) -> dict:
    """Add a comment to an existing issue or pull request."""
    data = _request("POST", f"/repos/{_repo()}/issues/{issue_number}/comments", json={"body": body})
    return {"url": data["html_url"]}


TOOL_FUNCTIONS = {
    "list_recent_commits": list_recent_commits,
    "get_commit_diff": get_commit_diff,
    "list_open_issues": list_open_issues,
    "create_issue": create_issue,
    "add_comment": add_comment,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_recent_commits",
            "description": "List the most recent commits on the repo's default branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "How many commits to return (max 20).",
                        "default": 5,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_commit_diff",
            "description": "Get the file-level diff (patch) for one commit, to inspect what changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sha": {"type": "string", "description": "The commit SHA to inspect."}
                },
                "required": ["sha"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_open_issues",
            "description": "List currently open issues in the repo (pull requests excluded).",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "How many issues to return (max 30).",
                        "default": 10,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_issue",
            "description": (
                "Open a new GitHub issue. Use this for anything the spoken request asks "
                "to flag or track, after you've actually inspected the relevant code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short issue title."},
                    "body": {
                        "type": "string",
                        "description": "Issue description — what was found and why it matters.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_comment",
            "description": "Add a comment to an existing issue or pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_number": {
                        "type": "integer",
                        "description": "The issue or PR number to comment on.",
                    },
                    "body": {"type": "string", "description": "Comment text."},
                },
                "required": ["issue_number", "body"],
            },
        },
    },
]
