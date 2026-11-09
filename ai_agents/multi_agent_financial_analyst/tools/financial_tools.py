import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json

class StockInput(BaseModel):
    """Input schema for YFinanceStockTool."""
    symbol: str = Field(..., description="The stock symbol to analyze (e.g., 'AAPL', 'GOOGL')")


class MarketDataUnavailable(RuntimeError):
    """Raised when current source data cannot be fetched for a requested symbol."""


class YFinanceStockTool(BaseTool):
    """Fetch Yahoo Finance market data and include the source trading date."""
    name: str = "stock_data_tool"
    description: str = """
    A tool for getting real-time and historical stock market data.
    Use this tool when you need specific stock information like:
    - Latest stock price from most recent trading day
    - Current price and trading volume
    - Historical price data
    - Company financials and metrics
    - Company information and business summary
    """
    args_schema: type[BaseModel] = StockInput

    def fetch_data(self, symbol: str) -> dict:
        """Fetch source data, raising instead of returning an LLM-usable error string."""
        try:
            stock = yf.Ticker(symbol)

            # Get basic info
            info = stock.info or {}

            # Get recent market data
            hist = stock.history(period="1mo")
            if hist.empty:
                raise ValueError("Yahoo Finance returned no recent trading data")

            # Get the latest trading day's data
            latest_data = hist.iloc[-1]
            latest_date = latest_data.name.strftime('%Y-%m-%d')

            # Format 52-week data with dates
            hist_1y = stock.history(period="1y")
            if hist_1y.empty:
                raise ValueError("Yahoo Finance returned no 52-week price history")
            fifty_two_week_high_date = hist_1y['High'].idxmax().strftime('%Y-%m-%d')
            fifty_two_week_low_date = hist_1y['Low'].idxmin().strftime('%Y-%m-%d')

            open_price = float(latest_data['Open'])
            close_price = float(latest_data['Close'])
            # Prepare the response
            response = {
                "company_name": info.get("longName", "N/A"),
                "latest_trading_data": {
                    "date": latest_date,
                    "price": close_price,
                    "volume": int(latest_data['Volume']),
                    "change": f"{((close_price - open_price) / open_price * 100):.2f}%" if open_price else "N/A"
                },
                "52_week_high": {
                    "price": info.get("fiftyTwoWeekHigh", "N/A"),
                    "date": fifty_two_week_high_date
                },
                "52_week_low": {
                    "price": info.get("fiftyTwoWeekLow", "N/A"),
                    "date": fifty_two_week_low_date
                },
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("forwardPE", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "business_summary": info.get("longBusinessSummary", "N/A"),
                "analyst_rating": info.get("recommendationKey", "N/A")
            }

            return response
        except Exception as e:
            raise MarketDataUnavailable(
                f"Unable to fetch current market data for {symbol}; no report was generated."
            ) from e

    def _run(self, symbol: str) -> str:
        """Fetch source data and return it as JSON for CrewAI tool callers."""
        return json.dumps(self.fetch_data(symbol), indent=2)

    def _arun(self, symbol: str) -> str:
        """Async counterpart of _run, not implemented for this tool."""
        # Async implementation if needed
        raise NotImplementedError("Async version not implemented") 
