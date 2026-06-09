"""
Brand monitoring pipeline.
Runs four platform analyzers sequentially and returns a dict of reports.
"""

from brand_monitor_agent.analyzers import (
    analyze_linkedin,
    analyze_twitter,
    analyze_web,
    analyze_youtube,
)


def run_brand_monitor(brand: str, scrapingdog_key: str, orq_key: str) -> dict:
    """
    Collect and analyse brand intelligence across four platforms.

    Args:
        brand:            Brand or company name to monitor.
        scrapingdog_key:  Scrapingdog API key.
        orq_key:          Orq.ai API key.

    Returns:
        Dict with keys "web", "youtube", "twitter", "linkedin" — each
        mapping to a structured markdown intelligence brief.
    """
    if not brand.strip():
        raise ValueError("Brand name cannot be empty.")
    if not scrapingdog_key.strip():
        raise ValueError("Scrapingdog API key is required.")
    if not orq_key.strip():
        raise ValueError("Orq.ai API key is required.")

    platforms = [
        ("web",      analyze_web),
        ("youtube",  analyze_youtube),
        ("twitter",  analyze_twitter),
        ("linkedin", analyze_linkedin),
    ]

    results = {}
    for platform, fn in platforms:
        try:
            results[platform] = fn(brand, scrapingdog_key, orq_key)
        except Exception as exc:
            results[platform] = f"Error analyzing {platform}: {exc}"

    return results
