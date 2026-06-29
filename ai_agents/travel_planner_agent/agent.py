"""LangChain travel planner agent with weather, currency, research, and packing tools."""

import os
from typing import Optional

import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

SYSTEM_PROMPT = """You are Travel Planner Agent, an expert travel planning assistant.

Help users plan trips end to end. When they describe a trip, gather missing details (destination, dates, budget, interests, home currency) through natural conversation, then use your tools to research and build a comprehensive plan.

Always use tools for factual data - never invent weather, exchange rates, or attraction details.

When you have enough information, return a structured travel plan with these sections:
1. **Trip Overview** - destination, dates, duration, traveler context
2. **Weather** - current conditions and forecast summary (from weather tool)
3. **Budget & Currency** - estimated costs and conversions (from currency tool)
4. **Destination Highlights** - attractions, food, culture, tips (from research tool)
5. **Packing List** - tailored recommendations (from packing tool; pass weather summary)
6. **Suggested Itinerary** - day-by-day outline based on interests and duration

Be friendly, concise, and practical. Ask clarifying questions when key details are missing."""


def _require_key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


@tool
def get_weather(destination: str) -> str:
    """Fetch current weather and a 5-day forecast for a destination city or location.

    Args:
        destination: City name (e.g. 'Tokyo', 'Paris, France').
    """
    api_key = _require_key("OPENWEATHER_API_KEY")
    base = "https://api.openweathermap.org/data/2.5"

    try:
        current_resp = requests.get(
            f"{base}/weather",
            params={"q": destination, "appid": api_key, "units": "metric"},
            timeout=15,
        )
        current_resp.raise_for_status()
        current = current_resp.json()

        forecast_resp = requests.get(
            f"{base}/forecast",
            params={"q": destination, "appid": api_key, "units": "metric"},
            timeout=15,
        )
        forecast_resp.raise_for_status()
        forecast = forecast_resp.json()
    except requests.RequestException as exc:
        return f"Weather lookup failed for '{destination}': {exc}"

    main = current.get("main", {})
    weather = (current.get("weather") or [{}])[0]
    wind = current.get("wind", {})

    lines = [
        f"Weather for {current.get('name', destination)}, {current.get('sys', {}).get('country', '')}",
        "",
        "Current:",
        f"  {weather.get('main', 'N/A')} - {weather.get('description', 'N/A')}",
        f"  Temperature: {main.get('temp', 'N/A')}°C (feels like {main.get('feels_like', 'N/A')}°C)",
        f"  Humidity: {main.get('humidity', 'N/A')}%",
        f"  Wind: {wind.get('speed', 'N/A')} m/s",
        "",
        "5-day forecast (3-hour intervals, sample):",
    ]

    seen_days: set[str] = set()
    for entry in forecast.get("list", [])[:16]:
        dt_txt = entry.get("dt_txt", "")
        day = dt_txt.split(" ")[0] if dt_txt else ""
        if day in seen_days:
            continue
        seen_days.add(day)
        w = (entry.get("weather") or [{}])[0]
        m = entry.get("main", {})
        lines.append(
            f"  {day}: {w.get('main', 'N/A')} - "
            f"{m.get('temp_min', 'N/A')}°C to {m.get('temp_max', 'N/A')}°C, "
            f"{w.get('description', '')}"
        )
        if len(seen_days) >= 5:
            break

    return "\n".join(lines)


@tool
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> str:
    """Convert an amount between currencies and show the exchange rate.

    Args:
        amount: Amount to convert (e.g. 1500).
        from_currency: Source currency code (e.g. 'USD').
        to_currency: Target currency code (e.g. 'JPY').
    """
    api_key = _require_key("EXCHANGERATE_API_KEY")
    from_code = from_currency.strip().upper()
    to_code = to_currency.strip().upper()
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_code}/{to_code}/{amount}"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return f"Currency conversion failed: {exc}"

    if data.get("result") != "success":
        return f"Currency API error: {data.get('error-type', 'unknown')}"

    conversion = data.get("conversion_result", amount)
    rate = data.get("conversion_rate", "N/A")

    return (
        f"{amount:,.2f} {from_code} = {conversion:,.2f} {to_code}\n"
        f"Exchange rate: 1 {from_code} = {rate} {to_code}"
    )


@tool
def research_destination(query: str) -> str:
    """Search the web for destination highlights, attractions, food, and travel tips.

    Args:
        query: Search query (e.g. 'best things to do in Kyoto Japan 2025').
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
    except Exception as exc:
        return f"Web research failed: {exc}"

    if not results:
        return f"No results found for: {query}"

    lines = [f"Research results for: {query}", ""]
    for i, hit in enumerate(results, 1):
        title = hit.get("title", "Untitled")
        body = hit.get("body", "")
        href = hit.get("href", "")
        lines.append(f"{i}. **{title}**")
        if body:
            lines.append(f"   {body[:300]}{'...' if len(body) > 300 else ''}")
        if href:
            lines.append(f"   Source: {href}")
        lines.append("")

    return "\n".join(lines)


@tool
def get_packing_recommendations(
    destination: str,
    trip_days: int,
    weather_summary: str,
) -> str:
    """Generate a packing list based on weather data and trip duration.

    Args:
        destination: Trip destination.
        trip_days: Number of days for the trip.
        weather_summary: Weather text from get_weather (temperatures and conditions).
    """
    summary_lower = weather_summary.lower()
    hot = any(k in summary_lower for k in ("hot", "warm", "clear", "sunny")) or _temp_hint(
        summary_lower, above=24
    )
    cold = any(k in summary_lower for k in ("cold", "snow", "freez", "ice")) or _temp_hint(
        summary_lower, below=10
    )
    rain = any(k in summary_lower for k in ("rain", "drizzle", "storm", "shower", "thunder"))

    essentials = [
        "Passport / ID and travel documents",
        "Phone, charger, and travel adapter",
        "Wallet, cards, and some local cash",
        "Medications and basic first-aid kit",
        "Reusable water bottle",
    ]

    clothing: list[str] = ["Comfortable walking shoes", "Underwear and socks"]
    if hot:
        clothing += [
            "Light breathable shirts and shorts/skirts",
            "Sun hat and sunglasses",
            "Sunscreen (SPF 30+)",
            "Swimwear (if applicable)",
        ]
    elif cold:
        clothing += [
            "Thermal base layers",
            "Warm jacket or coat",
            "Gloves, scarf, and beanie",
            "Closed waterproof boots",
        ]
    else:
        clothing += [
            "Layered tops (t-shirts + light sweater)",
            "Jeans or comfortable pants",
            "Light jacket for evenings",
        ]

    if rain:
        clothing += ["Compact umbrella", "Waterproof jacket", "Quick-dry clothing"]

    extras = []
    if trip_days > 5:
        extras.append("Laundry soap sheets or plan for laundry mid-trip")
    if trip_days > 10:
        extras.append("Extra outfit rotation - pack for 7–10 days and wash")

    lines = [
        f"Packing list for {destination} ({trip_days} days)",
        "",
        "**Essentials**",
        *[f"- {item}" for item in essentials],
        "",
        "**Clothing & weather gear**",
        *[f"- {item}" for item in clothing],
    ]
    if extras:
        lines += ["", "**Trip-length tips**", *[f"- {item}" for item in extras]]

    lines += [
        "",
        "**Toiletries**",
        "- Toothbrush, toothpaste, deodorant",
        "- Skincare and any personal care items",
        "",
        f"_Based on weather context: {weather_summary[:200]}..._"
        if len(weather_summary) > 200
        else f"_Based on weather context provided._",
    ]

    return "\n".join(lines)


def _temp_hint(text: str, above: Optional[float] = None, below: Optional[float] = None) -> bool:
    import re

    temps = [float(t) for t in re.findall(r"(-?\d+(?:\.\d+)?)\s*°C", text)]
    if not temps:
        return False
    avg = sum(temps) / len(temps)
    if above is not None and avg >= above:
        return True
    if below is not None and avg <= below:
        return True
    return False


def create_travel_agent():
    """Build and return the LangChain tool-calling agent graph."""
    api_key = os.getenv("ORQ_API_KEY")
    if not api_key:
        raise ValueError("Missing environment variable: ORQ_API_KEY")

    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        base_url="https://api.orq.ai/v3/router",
        api_key=api_key,
        temperature=0.4,
    )

    tools = [
        get_weather,
        convert_currency,
        research_destination,
        get_packing_recommendations,
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def extract_agent_reply(result: dict) -> str:
    """Return the latest AI message content from an agent invoke result."""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            content = message.content
            if isinstance(content, str):
                return content
            return str(content)
    return "Sorry, I could not generate a response."
