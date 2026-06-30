import os
from typing import Any

from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crew import EmailProcessingCrew
from email_utils import fetch_unread_emails, get_email_credentials

load_dotenv()


class ProcessedEmail(BaseModel):
    """Structured result for a single processed email."""

    sender: str = ""
    subject: str = ""
    body: str = ""
    category: str = "inquiry"
    needs_response: bool = True
    draft_response: str | None = None
    message_id: str = ""


class EmailFlowState(BaseModel):
    """Shared state for the email auto responder flow."""

    emails: list[dict[str, Any]] = Field(default_factory=list)
    results: list[ProcessedEmail] = Field(default_factory=list)
    error: str | None = None


class EmailAutoResponderFlow(Flow[EmailFlowState]):
    """CrewAI flow that fetches unread emails and processes them."""

    @start()
    def fetch_emails(self) -> list[dict[str, Any]]:
        """Fetch unread emails from Gmail via IMAP."""
        try:
            email_address, app_password = get_email_credentials()
            emails = fetch_unread_emails(
                email_address=email_address,
                app_password=app_password,
                max_emails=int(os.getenv("MAX_EMAILS", "5")),
            )
            self.state.emails = emails
            return emails
        except Exception as exc:
            self.state.error = str(exc)
            self.state.emails = []
            return []

    @listen(fetch_emails)
    def process_emails(self, emails: list[dict[str, Any]]) -> list[ProcessedEmail]:
        """Classify each email and generate draft replies when needed."""
        if self.state.error:
            return []

        if not emails:
            self.state.results = []
            return []

        crew = EmailProcessingCrew()
        processed: list[ProcessedEmail] = []

        for email_data in emails:
            result = crew.process_email(email_data)
            processed.append(ProcessedEmail(**result))

        self.state.results = processed
        return processed


def run_email_flow() -> dict[str, Any]:
    """Run the email flow and return processed results or an error."""
    flow = EmailAutoResponderFlow()
    flow.kickoff()

    return {
        "results": [result.model_dump() for result in flow.state.results],
        "email_count": len(flow.state.emails),
        "error": flow.state.error,
    }
