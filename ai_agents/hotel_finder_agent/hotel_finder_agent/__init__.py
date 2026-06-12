import os

from dotenv import load_dotenv

load_dotenv()

# Map ORQ_API_KEY → OPENAI_API_KEY and set the Orq.ai base URL so that
# LiteLlm can route calls through Orq.ai when using `adk web`.
os.environ.setdefault("OPENAI_API_BASE", "https://my.orq.ai/v3/router")
if "ORQ_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["ORQ_API_KEY"]

from .agent import root_agent  # noqa: E402

__all__ = ["root_agent"]
