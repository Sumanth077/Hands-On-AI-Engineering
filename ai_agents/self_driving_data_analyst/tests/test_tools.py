import duckdb
import pandas as pd
import pytest

from agent.loop_detector import is_looping
from agent.state import InvestigationState
from tools.charts import create_chart
from tools.dataset import inspect_dataset
from tools.findings import save_finding
from tools.python_tool import run_python
from tools.sql import run_sql


@pytest.fixture
def con():
    connection = duckdb.connect(database=":memory:")
    df = pd.DataFrame({"category": ["A", "B", "A"], "revenue": [100, 200, 150]})
    connection.register("orders", df)
    return connection


def test_inspect_dataset(con):
    result = inspect_dataset(con)
    assert "orders" in result["tables"]
    assert result["tables"]["orders"]["row_count"] == 3
    assert {"name": "category", "type": "VARCHAR"} in [
        {"name": c["name"], "type": c["type"]} for c in result["tables"]["orders"]["columns"]
    ]


def test_run_sql_success(con):
    result = run_sql(con, "SELECT category, SUM(revenue) AS total FROM orders GROUP BY category")
    assert result["success"] is True
    assert result["row_count"] == 2


def test_run_sql_error_is_returned_not_raised(con):
    result = run_sql(con, "SELECT nonexistent_column FROM orders")
    assert result["success"] is False
    assert "error" in result


def test_run_python_sets_result():
    dfs = {"orders": pd.DataFrame({"revenue": [100, 200, 300]})}
    result = run_python("result = orders['revenue'].mean()", dfs)
    assert result["success"] is True
    assert result["result"] == 200.0


def test_run_python_blocks_dangerous_builtins():
    result = run_python("open('/etc/passwd')", {})
    assert result["success"] is False


def test_create_chart_success():
    rows = [{"category": "A", "revenue": 100}, {"category": "B", "revenue": 200}]
    result = create_chart(rows, chart_type="bar", x="category", y="revenue")
    assert result["success"] is True
    assert "figure" in result


def test_create_chart_bad_column():
    rows = [{"category": "A"}]
    result = create_chart(rows, chart_type="bar", x="does_not_exist")
    assert result["success"] is False


def test_save_finding_persists_to_state():
    state = InvestigationState(objective="test")
    save_finding(state, "Revenue dropped 18%", importance="high")
    assert len(state.findings) == 1
    assert state.findings[0].importance == "high"


def test_loop_detector_flags_repeated_queries():
    state = InvestigationState(objective="test")
    for _ in range(4):
        state.add_tool_call("run_sql", {"query": "SELECT * FROM orders"}, "some result")
    assert is_looping(state) is True


def test_loop_detector_ignores_varied_queries():
    state = InvestigationState(objective="test")
    queries = [
        "SELECT * FROM orders",
        "SELECT category, SUM(revenue) FROM orders GROUP BY category",
        "SELECT * FROM customers",
        "SELECT AVG(revenue) FROM orders WHERE category = 'Fashion'",
    ]
    for q in queries:
        state.add_tool_call("run_sql", {"query": q}, "some result")
    assert is_looping(state) is False
