from agent import needs_approval


def test_external_execution_requires_approval() -> None:
    assert needs_approval({"name": "execute_action"}) is True


def test_discovery_does_not_require_approval() -> None:
    assert needs_approval({"name": "search_actions"}) is False

