# Email Auto Responder
> An automated email triage and response agent that reads unread Gmail messages, classifies intent, and drafts professional replies.

## Overview
Email Auto Responder helps you process incoming Gmail messages faster. The app fetches unread emails over IMAP, uses CrewAI agents powered by GLM-5.1 via NVIDIA NIM to classify each message and draft a reply, then presents the original email alongside the suggested response in a clean Streamlit dashboard.

## Demo
<img src="assets/demo.gif" alt="Email Auto Responder demo showing Gmail triage and draft replies in the Streamlit dashboard" width="800"/>

## Features
- Fetch unread Gmail messages via IMAP
- Classify emails into seven categories: inquiry, complaint, follow-up, marketing, notification, newsletter, and spam
- Determine whether each message needs a reply via a separate needs_response flag
- Generate professional draft replies only when a response is needed
- Color-coded category badges and response-needed indicators in the UI
- Friendly empty state when no unread emails exist
- Sidebar with app description and example use cases

## Tech Stack
**Agent Framework:**
- CrewAI Flows - agent orchestration and flow management

**LLM:**
- GLM-5.1 (z-ai/glm-5.1) via NVIDIA NIM - OpenAI-compatible API

**Email:**
- Gmail via IMAP using Python imaplib

**UI:**
- Streamlit

## Prerequisites
- Python 3.10+
- NVIDIA API key - get one at https://build.nvidia.com
- Gmail account with IMAP enabled
- Gmail App Password for IMAP authentication - get one at https://myaccount.google.com/apppasswords

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/email_auto_responder
```

If you already have this folder locally (for example as a standalone checkout or copy), skip to step 2.

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env          # macOS/Linux
copy .env.example .env        # Windows CMD
Copy-Item .env.example .env   # PowerShell
```
Open `.env` and add your credentials:
- `NVIDIA_API_KEY` - your NVIDIA NIM API key
- `EMAIL_ADDRESS` - your Gmail address
- `APP_PASSWORD` - your Gmail App Password

### 5. Run the App
```bash
streamlit run app.py
```

## Usage
Click **Check Emails** in the dashboard to fetch and process your unread Gmail messages. The agent classifies each email and drafts a reply for messages that need a response.

Example email types handled:
- Inquiry from a client asking about pricing
- Complaint about a delayed order
- Follow-up on a previous conversation
- Newsletter or marketing email (no reply needed)
- Spam (no reply needed)

## Project Structure
email_auto_responder/
├── app.py              # Streamlit UI
├── flow.py             # CrewAI Flow orchestration
├── crew.py             # CrewAI agents and tasks
├── email_utils.py      # Gmail IMAP helpers
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── assets/
    └── demo.gif

## How It Works
1. The user clicks **Check Emails** in the Streamlit UI.
2. The CrewAI Flow fetches unread emails from Gmail via IMAP.
3. The Email Classifier agent assigns each email a category and a needs_response flag.
4. The Response Writer agent drafts a professional reply only when needs_response is true.
5. Results are displayed in the UI with the original email, category badge, and draft response when applicable.
