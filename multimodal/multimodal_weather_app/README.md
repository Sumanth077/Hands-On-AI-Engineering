# Multimodal Weather App

> Upload a map image and get live weather: Mistral Small identifies the city via vision, then fetches real-time conditions through native tool calling.

## Overview

This app combines two AI capabilities in a single pipeline: **vision** and **tool calling**. You upload any map image, Mistral Small 4 reads it to identify the city, and then the same model calls a weather tool to retrieve live conditions — all displayed in a clean Gradio interface.

## Demo

![Demo](assets/demo.png)

## Features

- **Vision-powered city detection:** Mistral Small 4 reads map images and extracts the city name
- **Native tool calling:** Mistral Small 4 decides when and how to invoke the weather tool
- **Live weather data:** temperature, conditions, humidity, and wind speed fetched in real time
- **No weather API key needed:** uses `python_weather`, a free weather client
- **Simple one-click UI:** built with Gradio Blocks

## Tech Stack

| Layer | Technology |
|---|---|
| LLM (vision + tool calling) | Mistral Small 4 (`mistral-small-latest`) via Mistral AI API |
| Weather data | `python_weather` (free, no API key required) |
| LLM client | `langchain-mistralai` |
| UI | Gradio |

## Prerequisites

- Python 3.10 or higher
- A Mistral AI API key; get one at [platform.mistral.ai](https://platform.mistral.ai)

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/multimodal/multimodal_weather_app
```

**2. Create and activate a virtual environment**

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and add your Mistral API key (see [Environment Variables](#environment-variables)).

## Usage

```bash
python app.py
```

Gradio will print a local URL (typically `http://127.0.0.1:7860`). Open it in your browser, upload a map image, and click **Identify City & Get Weather**.

**Example**

| Input | Output |
|---|---|
| Map image of Tokyo | **City:** Tokyo<br>**Temperature:** 18°C<br>**Conditions:** Partly Cloudy<br>**Humidity:** 62%<br>**Wind Speed:** 14 km/h |

## Environment Variables

Create a `.env` file in the project root with the following:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Your Mistral AI API key; get one at [platform.mistral.ai](https://platform.mistral.ai) |

## Project Structure

```text
multimodal-weather-app/
├── app.py              # Main application — vision pipeline, tool calling, Gradio UI
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── .env                # Your local secrets (not committed)
```

---

[Back to Top](#multimodal-weather-app)
