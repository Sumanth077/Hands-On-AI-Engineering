"""
Registry Broker API tools.

All four functions hit the public REST API at https://hol.org/registry/api/v1.
No authentication required for search endpoints.
"""

import httpx

BASE_URL = "https://hol.org/registry/api/v1"
TIMEOUT = 30.0


def search_agents(query: str, limit: int = 10) -> dict:
    """Search for AI agents across all registries using a keyword or natural language query.

    Args:
        query: The search query — e.g. "code review", "trading bot", "customer support".
        limit: Maximum number of results to return. Defaults to 10.

    Returns:
        A dict containing a list of matching agents with their name, registry,
        description, capabilities, and Universal Agent ID (UAID).
    """
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(
            f"{BASE_URL}/search",
            params={"q": query, "limit": limit},
        )
        response.raise_for_status()
        return response.json()


def get_agent_details(uaid: str) -> dict:
    """Get comprehensive details about a specific agent by its Universal Agent ID (UAID).

    Args:
        uaid: The Universal Agent ID of the agent to look up.

    Returns:
        Full agent metadata including name, description, registry, capabilities,
        endpoints, trust score, and verification status.
    """
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(f"{BASE_URL}/agents/{uaid}")
        response.raise_for_status()
        return response.json()


def get_similar_agents(uaid: str, limit: int = 5) -> dict:
    """Find agents similar to a given agent — useful for comparing alternatives.

    Args:
        uaid: The Universal Agent ID of the reference agent.
        limit: Maximum number of similar agents to return. Defaults to 5.

    Returns:
        A list of agents with similar capabilities or use cases, with their
        registry source and UAID for further lookup.
    """
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(
            f"{BASE_URL}/agents/{uaid}/similar",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json()


def get_search_facets() -> dict:
    """Get available categories, registries, and capability filters for browsing.

    Returns:
        Available facets including registry names (NANDA, MCP, Virtuals, A2A, ERC-8004),
        agent categories, and capability tags that can be used to filter searches.
    """
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(f"{BASE_URL}/search/facets")
        response.raise_for_status()
        return response.json()
