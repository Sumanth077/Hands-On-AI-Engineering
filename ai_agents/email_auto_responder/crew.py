import json
import os
import re

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

load_dotenv()

CATEGORY_NEEDS_RESPONSE: dict[str, bool] = {
    "inquiry": True,
    "complaint": True,
    "follow_up": True,
    "marketing": False,
    "notification": False,
    "newsletter": False,
    "spam": False,
}

VALID_CATEGORIES = set(CATEGORY_NEEDS_RESPONSE.keys())

_CATEGORY_ALIASES: dict[str, str] = {
    "follow-up": "follow_up",
    "followup": "follow_up",
    "follow up": "follow_up",
}


def get_llm() -> LLM:
    """Create the NVIDIA NIM LLM client from environment config."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY must be set in the environment.")

    return LLM(
        model="z-ai/glm-5.1",
        provider="openai",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


def normalize_category(raw_category: str) -> str:
    """Map raw classifier output to a valid category name."""
    cleaned = raw_category.strip().lower().replace("-", "_")
    cleaned = _CATEGORY_ALIASES.get(cleaned.replace("_", " "), cleaned)
    cleaned = cleaned.replace(" ", "_")

    if cleaned in VALID_CATEGORIES:
        return cleaned

    if re.search(r"\b(not|non|no)[_\s-]?spam\b", cleaned):
        for category in sorted(VALID_CATEGORIES - {"spam"}, key=len, reverse=True):
            if re.search(rf"\b{category}\b", cleaned):
                return category
        return "inquiry"

    for category in sorted(VALID_CATEGORIES, key=len, reverse=True):
        pattern = category.replace("_", "[_\\s-]?")
        if re.search(rf"\b{pattern}\b", cleaned):
            return category

    return "inquiry"


def parse_classify_output(raw: str) -> tuple[str, bool]:
    """Parse classifier output into category and needs_response values."""
    text = raw.strip()

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            category = normalize_category(str(data.get("category", "inquiry")))
            needs_response = data.get("needs_response")
            if isinstance(needs_response, bool):
                return category, needs_response
            if isinstance(needs_response, str):
                parsed = needs_response.strip().lower() in ("true", "yes", "1")
                return category, parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    category = normalize_category(lines[0] if lines else text)
    needs_response = CATEGORY_NEEDS_RESPONSE[category]

    for line in lines:
        lower_line = line.lower()
        if "needs_response" in lower_line:
            if "false" in lower_line or "no" in lower_line.split(":")[-1]:
                needs_response = False
            elif "true" in lower_line or "yes" in lower_line.split(":")[-1]:
                needs_response = True
            break

    return category, needs_response


class EmailProcessingCrew:
    """CrewAI agents and tasks for email classification and reply drafting."""

    def __init__(self) -> None:
        """Initialize the crew with a configured LLM."""
        self.llm = get_llm()

    def email_classifier(self) -> Agent:
        """Return the agent that classifies email intent."""
        return Agent(
            role="Email Classifier",
            goal=(
                "Classify incoming emails into exactly one category and determine "
                "whether the message needs a human reply."
            ),
            backstory=(
                "You are an expert email triage specialist who quickly identifies "
                "customer intent, promotional content, automated alerts, and malicious "
                "messages. You distinguish legitimate marketing and notifications from "
                "actual spam, and you never miscategorize urgent complaints."
            ),
            llm=self.llm,
            verbose=True,
        )

    def response_writer(self) -> Agent:
        """Return the agent that drafts email replies."""
        return Agent(
            role="Response Writer",
            goal=(
                "Draft professional, concise email replies that address the sender's "
                "needs while matching the tone of the original message."
            ),
            backstory=(
                "You are a skilled customer communications expert with years of "
                "experience writing clear, helpful, and empathetic email responses "
                "for support and sales teams."
            ),
            llm=self.llm,
            verbose=True,
        )

    def classify_task(self, email_data: dict[str, str]) -> Task:
        """Build the classification task for a single email."""
        return Task(
            description=(
                "Classify the following email and decide if it needs a reply.\n\n"
                f"Sender: {email_data['sender']}\n"
                f"Subject: {email_data['subject']}\n"
                f"Body:\n{email_data['body']}\n\n"
                "Choose exactly one category:\n"
                "- inquiry: customer question, project ask, or request for information\n"
                "- complaint: billing issue, service problem, or dissatisfaction\n"
                "- follow_up: checking on a prior thread or awaiting an update\n"
                "- marketing: promotional offers, product announcements, or webinar invites "
                "from known senders\n"
                "- notification: automated alerts the user subscribed to, such as job alerts, "
                "shipping updates, or receipts\n"
                "- newsletter: recurring content digests or blog subscriptions\n"
                "- spam: malicious, phishing, scam, or unsolicited junk ONLY\n\n"
                "Important: Do not use spam for legitimate marketing or notification emails.\n\n"
                "Return ONLY valid JSON with no markdown fences:\n"
                '{"category": "<category>", "needs_response": <true or false>}\n'
                "Use snake_case category values. Set needs_response to true only for "
                "inquiry, complaint, and follow_up."
            ),
            expected_output=(
                'JSON object: {"category": "inquiry", "needs_response": true}'
            ),
            agent=self.email_classifier(),
        )

    def response_task(self, email_data: dict[str, str], classify_task: Task) -> Task:
        """Build the reply drafting task for a single email."""
        return Task(
            description=(
                "Write a professional, concise reply to the email below.\n\n"
                f"Sender: {email_data['sender']}\n"
                f"Subject: {email_data['subject']}\n"
                f"Body:\n{email_data['body']}\n\n"
                "Use the classification JSON from the previous task. "
                "If needs_response is false, or the category is marketing, notification, "
                "newsletter, or spam, respond with exactly: NO_RESPONSE\n\n"
                "Otherwise, write a complete reply body without a subject line prefix. "
                "Match the sender's tone and keep the response concise."
            ),
            expected_output=(
                "A professional email reply body, or NO_RESPONSE when no reply is needed"
            ),
            agent=self.response_writer(),
            context=[classify_task],
        )

    def crew(self) -> Crew:
        """Return a base Crew with classifier and writer agents."""
        return Crew(
            agents=[self.email_classifier(), self.response_writer()],
            tasks=[],
            process=Process.sequential,
            verbose=True,
        )

    def process_email(self, email_data: dict[str, str]) -> dict[str, str | bool | None]:
        """Classify one email and return category, flags, and draft reply."""
        classifier = self.email_classifier()
        writer = self.response_writer()
        classify = self.classify_task(email_data)
        respond = self.response_task(email_data, classify)

        processing_crew = Crew(
            agents=[classifier, writer],
            tasks=[classify, respond],
            process=Process.sequential,
            verbose=True,
        )

        result = processing_crew.kickoff()
        category, needs_response = parse_classify_output(
            str(result.tasks_output[0].raw)
        )
        raw_response = str(result.tasks_output[1].raw).strip()

        draft_response: str | None
        if not needs_response or raw_response.upper() == "NO_RESPONSE":
            draft_response = None
        else:
            draft_response = re.sub(
                r"^(subject:\s*.+\n+)",
                "",
                raw_response,
                flags=re.IGNORECASE,
            ).strip()

        return {
            "sender": email_data["sender"],
            "subject": email_data["subject"],
            "body": email_data["body"],
            "category": category,
            "needs_response": needs_response,
            "draft_response": draft_response,
            "message_id": email_data.get("message_id", ""),
        }
