"""
Marketing Strategy Agent - Pipeline.

Runs the three agents sequentially and returns all three outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents import run_creative_director, run_market_analyst, run_strategy_officer


@dataclass
class CampaignResult:
    """Holds the outputs of all three agents from a single campaign generation run."""
    research: str
    strategy: str
    creative: str


def run_campaign(api_key: str, product_description: str, target_audience: str) -> CampaignResult:
    """Run all three agents in sequence and return their outputs."""
    research = run_market_analyst(api_key, product_description, target_audience)
    strategy = run_strategy_officer(api_key, product_description, target_audience, research)
    creative = run_creative_director(api_key, product_description, target_audience, strategy)
    return CampaignResult(research=research, strategy=strategy, creative=creative)
