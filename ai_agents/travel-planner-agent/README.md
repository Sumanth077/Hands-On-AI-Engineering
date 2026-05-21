# Travel Planner Agent

**Your AI co-pilot for end-to-end trip planning — from weather and budget to packing lists and day-by-day itineraries.**

## Overview

Travel Planner Agent is a conversational travel assistant that helps you plan trips through natural language. Describe where you want to go, when, your budget, and what you enjoy — the agent gathers any missing details, researches your destination with live tools, and returns a structured travel plan you can use right away.

Built with LangChain and Streamlit, it combines real-time weather, currency conversion, web research, and smart packing recommendations into a single chat experience.

## Demo

![Demo](assets/demo.gif)

## Features

- **Conversational trip planning** — describe trips in plain English; the agent asks follow-up questions when details are missing
- **Live weather data** — current conditions and a 5-day forecast for any destination via OpenWeatherMap
- **Budget and currency tools** — convert amounts between currencies and estimate trip costs via ExchangeRate-API
- **Destination research** — attractions, food, culture, and travel tips sourced from DuckDuckGo web search
- **Smart packing lists** — recommendations tailored to forecast, climate, and trip length
- **Structured travel plans** — overview, weather, budget, highlights, packing, and a suggested day-by-day itinerary
- **Session-based chat history** — multi-turn conversations with context preserved in the Streamlit UI

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent Orchestration | [LangChain](https://www.langchain.com/) + [LangGraph](https://www.langchain.com/langgraph) (`create_agent` tool-calling loop) |
| LLM | [OpenAI SDK](https://github.com/openai/openai-python) → [Orq.ai](https://orq.ai) router (`deepseek-v4-flash`) |
| Weather | [OpenWeatherMap API](https://openweathermap.org/api) |
| Currency | [ExchangeRate-API](https://www.exchangerate-api.com/) |
| Web Search | [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) |
| UI | [Streamlit](https://streamlit.io/) chat interface |

## Prerequisites

- **Python 3.10 or higher**
- API keys for:
  - [Orq.ai](https://orq.ai) — LLM routing
  - [OpenWeatherMap](https://openweathermap.org/api) — weather and forecast data
  - [ExchangeRate-API](https://www.exchangerate-api.com/) — currency conversion

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/travel-planner-agent.git
cd travel-planner-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your API keys. See the [Environment Variables](#environment-variables) section below for details.

## Environment Variables

| Variable | Description | Where to Get It |
|----------|-------------|-----------------|
| `ORQ_API_KEY` | Authenticates requests to the Orq.ai LLM router (`deepseek-v4-flash`) | [Orq.ai Dashboard](https://orq.ai) → API Keys |
| `OPENWEATHER_API_KEY` | Fetches current weather and 5-day forecasts for destinations | [OpenWeatherMap](https://home.openweathermap.org/api_keys) → API keys |
| `EXCHANGERATE_API_KEY` | Converts currencies and returns live exchange rates | [ExchangeRate-API](https://www.exchangerate-api.com/) → Get Free Key |

## Usage

### Run the app

With your virtual environment activated:

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (typically at `http://localhost:8501`). Use the chat input at the bottom of the page to describe your trip.

### Example user inputs

| Scenario | Example prompt |
|----------|----------------|
| Full trip plan | *Plan a 7-day trip to Tokyo in April. My budget is $3,000 USD. I love food, temples, and street photography.* |
| Weather check | *What's the weather like in Barcelona? I'm visiting for 5 days in June.* |
| Currency conversion | *Convert 2,500 USD to EUR and THB for a two-week trip across Europe and Thailand.* |
| Packing help | *I'm going to Iceland for 8 days in February. What should I pack based on the weather?* |

### What the agent returns

Once it has enough context, the agent calls its tools and delivers a structured plan with:

1. **Trip Overview** — destination, dates, duration, and traveler preferences
2. **Weather** — current conditions and a forecast summary from OpenWeatherMap
3. **Budget & Currency** — cost estimates and currency conversions from ExchangeRate-API
4. **Destination Highlights** — attractions, dining, culture, and travel tips from web research
5. **Packing List** — clothing and essentials based on weather and trip length
6. **Suggested Itinerary** — a day-by-day outline aligned with your interests and schedule

The agent uses tools for all factual data (weather, rates, research) rather than guessing. If details are missing, it will ask clarifying questions before building the full plan.

Use **Clear conversation** in the sidebar to start a new trip from scratch.

## Project Structure

```
travel-planner-agent/
├── app.py              # Streamlit UI and chat interface
├── agent.py            # LangChain agent and tools
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
├── README.md
└── assets/
    └── demo.gif        # Application demo
```

## License

MIT
