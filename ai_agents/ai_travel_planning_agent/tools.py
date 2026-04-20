import os
from tavily import TavilyClient
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


def tavily_search(query: str) -> str:
    """Search the web for current travel information using Tavily.

    Args:
        query: The search query string for travel-related information.

    Returns:
        Formatted string of search results with titles, URLs, and content snippets.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not configured."
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
        parts = []
        if response.get("answer"):
            parts.append(f"Summary: {response['answer']}\n")
        for r in response.get("results", []):
            snippet = r.get("content", "")[:600].strip()
            parts.append(f"**{r['title']}**\nURL: {r['url']}\n{snippet}")
        return "\n\n---\n\n".join(parts) if parts else "No results found."
    except Exception as e:
        return f"Search error: {e}"


def find_nearby_places(location: str, category: str = "tourist attractions") -> str:
    """Find geographic information and nearby places using OpenStreetMap/Nominatim.

    Args:
        location: City, landmark, or address to look up (e.g., "Paris, France").
        category: Type of places to mention (e.g., "museums", "restaurants", "parks").

    Returns:
        Geographic details including coordinates, full address, and context.
    """
    geolocator = Nominatim(user_agent="ai-travel-planning-agent/1.0", timeout=10)
    try:
        geo = geolocator.geocode(location, addressdetails=True)
        if not geo:
            return f"Could not find geographic data for: {location}"

        address = geo.raw.get("address", {})
        details = {
            "Location": location,
            "Full Address": geo.address,
            "Latitude": f"{geo.latitude:.5f}",
            "Longitude": f"{geo.longitude:.5f}",
            "Country": address.get("country", "N/A"),
            "State/Region": address.get("state", address.get("county", "N/A")),
            "Category Searched": category,
        }
        lines = [f"{k}: {v}" for k, v in details.items()]
        return "\n".join(lines)
    except GeocoderTimedOut:
        return f"Geocoding timed out for: {location}. Try a more specific location name."
    except GeocoderUnavailable:
        return "Nominatim service temporarily unavailable. Please retry."
    except Exception as e:
        return f"Location lookup error: {e}"
