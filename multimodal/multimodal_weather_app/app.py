"""Multimodal weather app that uses Mistral Small vision to identify a city from a map image, then fetches live weather via native tool calling."""

import os
import base64
import asyncio
import logging
import tempfile
import threading
import gradio as gr
from PIL import Image
import python_weather
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
logger = logging.getLogger(__name__)


# ── Image helpers ────────────────────────────────────────────────────────────

def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded JPEG string for use in API requests."""
    # Write to a real temp file first to ensure a complete, valid JPEG on disk
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    image.convert("RGB").save(tmp_path, format="JPEG")
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(tmp_path)
    logger.debug("[base64 preview]: %r  (total length: %d)", b64[:100], len(b64))
    return b64


# ── City identification via Mistral Small (vision) ───────────────────────────

def identify_city(image: Image.Image) -> str:
    """Use Mistral Small vision to read a map image and return the identified city name."""
    b64 = image_to_base64(image)
    llm = ChatMistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY)
    message = HumanMessage(content=[
        {"type": "text", "text": "What city is shown in this map? Reply with the city name only."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])
    response = llm.invoke([message])
    raw = response.content.strip()
    logger.debug("[mistral vision identified city]: %r", raw)
    return raw


# ── Async helper ─────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine in a dedicated thread with its own event loop."""
    result_holder: list = [None]
    error_holder: list = [None]

    def target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder[0] = loop.run_until_complete(coro)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            loop.close()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join()

    if error_holder[0]:
        raise error_holder[0]
    return result_holder[0]


# ── Weather fetcher ───────────────────────────────────────────────────────────

def get_weather(city: str) -> str:
    """Fetch current weather conditions for a city. Input is the city name."""

    async def _fetch():
        async with python_weather.Client(unit=python_weather.METRIC) as client:
            w = await client.get(city)
            kind = w.kind
            if hasattr(kind, "name"):
                description = kind.name.replace("_", " ").title()
            else:
                description = str(kind).replace("WeatherKind.", "").replace("_", " ").title()
            return {
                "temperature": w.temperature,
                "description": description,
                "humidity": w.humidity,
                "wind_speed": w.wind_speed,
            }

    try:
        data = run_async(_fetch())
        return (
            f"City: {city}\n"
            f"Temperature: {data['temperature']}°C\n"
            f"Conditions: {data['description']}\n"
            f"Humidity: {data['humidity']}%\n"
            f"Wind Speed: {data['wind_speed']} km/h"
        )
    except Exception as exc:
        return f"Weather unavailable for '{city}': {exc}"


# ── Weather tool schema + Mistral tool-calling ───────────────────────────────

_WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Fetch current weather conditions for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name to get weather for."},
            },
            "required": ["city"],
        },
    },
}


def call_weather_tool(city: str) -> str:
    """Ask Mistral Small to invoke the weather tool for the given city and return the formatted result."""
    llm = ChatMistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY)
    llm_with_tools = llm.bind_tools([_WEATHER_TOOL_SCHEMA])
    response = llm_with_tools.invoke([HumanMessage(content=f"Get the current weather for {city}.")])
    for tc in response.tool_calls:
        if tc["name"] == "get_weather":
            return get_weather(tc["args"].get("city", city))
    return response.content


# ── Output parser ─────────────────────────────────────────────────────────────

def parse_weather(text: str) -> tuple:
    """Extract temperature, conditions, humidity, and wind speed from a formatted weather string."""
    temp = conditions = humidity = wind = ""
    for line in text.splitlines():
        if "Temperature:" in line:
            temp = line.split("Temperature:", 1)[1].strip()
        elif "Conditions:" in line:
            conditions = line.split("Conditions:", 1)[1].strip()
        elif "Humidity:" in line:
            humidity = line.split("Humidity:", 1)[1].strip()
        elif "Wind Speed:" in line:
            wind = line.split("Wind Speed:", 1)[1].strip()
    return temp, conditions, humidity, wind


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze(image):
    """Run the full pipeline: identify the city from the uploaded map image, then fetch and return live weather data."""
    if image is None:
        return "Please upload a map image.", "", "", "", ""

    try:
        # Step 1: Vision — identify city from map image via Mistral Small
        city = identify_city(image)

        if not city:
            return "Mistral returned an empty response. Try a different image.", "", "", "", ""

        # Step 2: Tool calling — Mistral Small decides to call get_weather
        output = call_weather_tool(city)

        temp, conditions, humidity, wind = parse_weather(output)
        return city, temp, conditions, humidity, wind

    except Exception as exc:
        return f"Error: {exc}", "", "", "", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

css = """
.result-box textarea {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(title="Multimodal Weather App", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        # Multimodal Weather App
        Upload any map image. Mistral Small identifies the city via vision, then uses
        tool calling to fetch live weather using python_weather.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Map Image")
            analyze_btn = gr.Button("Identify City & Get Weather", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Weather Report")
            city_out = gr.Textbox(label="Identified City", interactive=False, elem_classes="result-box")
            with gr.Row():
                temp_out = gr.Textbox(label="Temperature", interactive=False)
                cond_out = gr.Textbox(label="Conditions", interactive=False)
            with gr.Row():
                hum_out = gr.Textbox(label="Humidity", interactive=False)
                wind_out = gr.Textbox(label="Wind Speed", interactive=False)

    analyze_btn.click(
        fn=analyze,
        inputs=[img_input],
        outputs=[city_out, temp_out, cond_out, hum_out, wind_out],
    )

    gr.Markdown(
        "_Powered by Mistral Small (vision + tool calling) + python_weather_",
        elem_id="footer",
    )

if __name__ == "__main__":
    demo.launch()
