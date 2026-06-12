# Cal Scheduling Agent

> A conversational scheduling assistant that manages Cal.com appointments through natural language. Check availability, book, reschedule, and cancel with automatic timezone handling. Powered by Gemini 3 Flash and Streamlit.

## Demo

![Demo](assets/demo.gif)

## Overview

Instead of logging into Cal.com and manually navigating availability calendars, this agent lets you ask in plain English: "what slots are free next week?", "book a meeting with Sarah for Tuesday at 3pm GMT", "cancel my appointment on Friday". It handles everything through a conversational interface.

The agent connects to the [Cal.com REST API](https://cal.com/docs/api-reference) to read availability, create bookings, and manage existing appointments in real time.

## Features

- **Availability Check:** Find open slots between any two dates across timezones
- **Smart Booking:** Confirm details before committing, with full meeting info on confirmation
- **Booking Management:** View upcoming, past, or cancelled appointments
- **Reschedule:** Move any appointment to a new time with an optional reason
- **Cancellation:** Cancel bookings with an optional note
- **Timezone Handling:** Automatically converts between timezones on every request

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 3 Flash (`gemini-3-flash-preview`) |
| Agent framework | [Agno](https://github.com/agno-agi/agno) |
| Calendar integration | Agno's built-in `CalComTools` |
| UI | Streamlit |

## Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- A Google API key. Get one free at [aistudio.google.com](https://aistudio.google.com)
- A Cal.com account with API access
- A Cal.com Event Type ID

## Installation

**Clone the repository**

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/cal_scheduling_agent
```

**Set up environment variables**

```bash
cp .env.example .env
```

Open `.env` and add your credentials.

**Install dependencies**

```bash
uv sync
```

## Usage

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser. Enter your credentials in the sidebar and start scheduling.

## Example Queries

```text
What slots are free this week?
Book a meeting for tomorrow at 2pm GMT
Show my upcoming bookings
Reschedule my next appointment to Friday at 10am
Cancel my booking with uid abc123
Find available times in New York timezone next Monday
Check bookings for john@example.com
What's open next Monday morning?
```

## Getting Your Credentials

| Credential | Where to get it |
|---|---|
| Google API Key | [aistudio.google.com](https://aistudio.google.com) |
| Cal.com API Key | [cal.com/settings/developer/api-keys](https://app.cal.com/settings/developer/api-keys) |
| Event Type ID | Your Cal.com dashboard (visible in the URL when editing an event type) |

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Authenticates Gemini 3 Flash requests |
| `CALCOM_API_KEY` | Authenticates Cal.com API requests |
| `CALCOM_EVENT_TYPE_ID` | The event type used for checking availability and creating bookings |

## Project Structure

```text
cal_scheduling_agent/
├── cal_scheduling_agent/      # Agent package
│   ├── __init__.py            # Package entry point
│   ├── agent.py               # root_agent definition (for adk web)
│   └── tools.py               # Cal.com API tools
├── app.py                     # Streamlit UI
├── pyproject.toml             # Project dependencies
├── .env                       # Your credentials (git-ignored)
├── .env.example               # Template for .env
└── assets/
    └── demo.gif
```

## How It Works

```
User message
    │
    ▼
Gemini 3 Flash decides which tool to call
    │
    ├── get_available_slots()   → GET /v1/slots
    ├── create_booking()        → POST /v1/bookings
    ├── get_bookings()          → GET /v1/bookings
    ├── reschedule_booking()    → PATCH /v1/bookings/{uid}/reschedule
    └── cancel_booking()        → DELETE /v1/bookings/{uid}/cancel
    │
    ▼
Cal.com API
    │
    ▼
Structured response in Streamlit chat UI
```
