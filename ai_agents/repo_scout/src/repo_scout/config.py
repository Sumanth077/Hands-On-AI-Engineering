import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr


def require_live_enabled() -> None:
    if os.getenv("REPOSCOUT_ENABLE_LIVE", "false").lower() != "true":
        raise ValueError(
            "Live research is disabled. REPOSCOUT_ENABLE_LIVE must be explicitly enabled in a later integration stage."
        )


class Settings(BaseModel):
    token: SecretStr
    account_plan: str = "unknown"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
        require_live_enabled()
        token = os.environ.get("APIFY_TOKEN", "").strip()
        if not token or token == "replace_with_local_token":
            raise ValueError("Set APIFY_TOKEN locally before making live requests")
        return cls(token=SecretStr(token), account_plan=os.getenv("APIFY_ACCOUNT_PLAN", "unknown"))
