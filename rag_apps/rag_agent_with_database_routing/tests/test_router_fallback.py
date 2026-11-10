from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_router_path = Path(__file__).resolve().parents[1] / "rag_agent" / "router.py"
_router_spec = importlib.util.spec_from_file_location("router_under_test", _router_path)
if _router_spec is None or _router_spec.loader is None:
    raise RuntimeError(f"Could not load router module from {_router_path}")
_router = importlib.util.module_from_spec(_router_spec)
sys.modules[_router_spec.name] = _router
_router_spec.loader.exec_module(_router)
_keyword_fallback = _router._keyword_fallback


class KeywordFallbackTests(unittest.TestCase):
    def test_account_help_intent_precedes_generic_financial_keyword(self) -> None:
        decision = _keyword_fallback("I need a plan to reset my password")

        self.assertEqual(decision.database, "support")

    def test_financial_query_still_routes_to_financial(self) -> None:
        decision = _keyword_fallback("What pricing plans are available?")

        self.assertEqual(decision.database, "financial")

    def test_unmatched_query_defaults_to_products(self) -> None:
        decision = _keyword_fallback("What colors does the device come in?")

        self.assertEqual(decision.database, "products")


if __name__ == "__main__":
    unittest.main()
