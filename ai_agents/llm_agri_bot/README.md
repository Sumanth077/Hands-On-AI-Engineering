# LLM Agri Bot
> A multi-tool farming assistant that helps farmers make informed decisions about crop health, weather, pest control, and planting seasons.

## Overview
LLM Agri Bot is a Streamlit chat application powered by LangChain and Mistral. Farmers type questions in plain English and an agent selects the right tool to fetch weather data, search the web for pest and disease guidance, or look up crop calendar information. Mistral then turns that data into practical, actionable advice.

## Demo
![Demo](assets/demo.gif)

## Features
- Streamlit chat interface with conversation history
- LangChain tool-calling agent powered by Mistral (mistral-small-latest)
- OpenWeatherMap integration for real-time local weather queries
- DuckDuckGo search for crop disease and pest control guidance
- Static crop calendar covering 18 common crops with planting and harvest guidance
- Sidebar with app description and example questions

## Tech Stack
**Agent Framework:**
- LangChain - agent orchestration and tool calling

**LLM:**
- Mistral (mistral-small-latest) via langchain-mistralai

**Tools:**
- OpenWeatherMap API - real-time weather data
- DuckDuckGo Search - crop disease and pest control lookup
- Crop Calendar - static knowledge base for 18 common crops

**UI:**
- Streamlit

## Prerequisites
- Python 3.10+
- Mistral API key - get one at https://console.mistral.ai/
- OpenWeatherMap API key - get one at https://openweathermap.org/api

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/llm_agri_bot
```

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
cp .env.example .env
copy .env.example .env  # Windows
```
Open `.env` and add your API keys: `MISTRAL_API_KEY` and `OPENWEATHER_API_KEY`.

### 5. Run the App
```bash
streamlit run app.py
```

## Usage
Type a farming question in the chat box and the agent will select the right tool and return practical advice. Example questions:

- "What is the weather like in Lagos today?"
- "My tomato leaves are turning yellow, what should I do?"
- "When should I plant maize?"
- "How do I control aphids on my pepper farm?"

## Project Structure
llm_agri_bot/

├── app.py              # Streamlit chat UI

├── agent.py            # LangChain agent and Mistral setup

├── tools.py            # Weather, search, and crop calendar tools

├── requirements.txt

├── .env.example

├── .gitignore

├── README.md

└── assets/

└── demo.gif

## How It Works
1. The farmer types a question in the Streamlit chat input.
2. The LangChain agent powered by Mistral decides which tool to call.
3. Tools fetch weather data, web search results, or crop calendar entries.
4. Mistral processes the tool output and generates a concise, actionable response.
5. The response is displayed in the chat UI and stored in session history.
