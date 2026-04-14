import yfinance as yf
from datetime import datetime
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json

class StockInput(BaseModel):
    """Input schema for YFinanceStockTool."""
    symbol: str = Field(..., description="The stock symbol to analyze (e.g., 'AAPL', 'GOOGL')")

class YFinanceStockTool(BaseTool):
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

    def _run(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            
            # Get basic info
            info = stock.info
            
            # Get recent market data
            hist = stock.history(period="1mo")
            
            # Get the latest trading day's data
            latest_data = hist.iloc[-1]
            latest_date = latest_data.name.strftime('%Y-%m-%d')
            
            # Format 52-week data with dates
            hist_1y = stock.history(period="1y")
            fifty_two_week_high_date = hist_1y['High'].idxmax().strftime('%Y-%m-%d')
            fifty_two_week_low_date = hist_1y['Low'].idxmin().strftime('%Y-%m-%d')
            
            # Prepare the response
            response = {
                "company_name": info.get("longName", "N/A"),
                "latest_trading_data": {
                    "date": latest_date,
                    "price": latest_data['Close'],
                    "volume": latest_data['Volume'],
                    "change": f"{((latest_data['Close'] - latest_data['Open']) / latest_data['Open'] * 100):.2f}%"
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
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return f"Error fetching data for {symbol}: {str(e)}"

    def _arun(self, symbol: str) -> str:
        # Async implementation if needed
        raise NotImplementedError("Async version not implemented") 