"""Alpha Vantage API integration for stock data retrieval.

Handles:
- Historical OHLCV data (daily, intraday)
- Company fundamentals (income statement, balance sheet, cash flow)
- Rate limiting and retry logic
- Data caching
"""

import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

from backend.app.config import (
    ALPHA_VANTAGE_API_KEY, AV_BASE_URL,
    AV_RATE_LIMIT_DELAY, AV_MAX_RETRIES, CACHE_DIR, CACHE_TTL_SECONDS
)

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """Client for Alpha Vantage API with rate limiting, caching, and retry logic."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        self.base_url = AV_BASE_URL
        self.last_call_time = 0
        self.session = requests.Session()

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < AV_RATE_LIMIT_DELAY:
            wait = AV_RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate limiting: waiting {wait:.1f}s")
            time.sleep(wait)
        self.last_call_time = time.time()

    def _cache_key(self, params: dict) -> str:
        """Generate a unique cache key for the request."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[dict]:
        """Retrieve cached response if still valid."""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            mtime = cache_file.stat().st_mtime
            if time.time() - mtime < CACHE_TTL_SECONDS:
                with open(cache_file) as f:
                    return json.load(f)
        return None

    def _set_cache(self, cache_key: str, data: dict):
        """Store response in cache."""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(data, f)

    def _request(self, params: dict) -> dict:
        """Make an API request with rate limiting, caching, and retries."""
        params["apikey"] = self.api_key
        cache_key = self._cache_key(params)

        cached = self._get_cached(cache_key)
        if cached:
            logger.debug("Returning cached response")
            return cached

        for attempt in range(AV_MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "Error Message" in data:
                    raise ValueError(f"API error: {data['Error Message']}")
                if "Note" in data:
                    logger.warning(f"API rate limit note: {data['Note']}")
                    if attempt < AV_MAX_RETRIES - 1:
                        time.sleep(60)
                        continue
                    raise ValueError("API rate limit exceeded")

                self._set_cache(cache_key, data)
                return data

            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < AV_MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise

        return {}

    def get_daily_prices(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """Fetch daily OHLCV price data.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            outputsize: 'compact' for last 100 days, 'full' for 20+ years

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, adjusted_close
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
        }

        data = self._request(params)
        time_series = data.get("Time Series (Daily)", {})

        if not time_series:
            logger.warning(f"No daily price data for {symbol}")
            return pd.DataFrame()

        records = []
        for date_str, values in time_series.items():
            records.append({
                "date": datetime.strptime(date_str, "%Y-%m-%d"),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "adjusted_close": float(values["5. adjusted close"]),
                "volume": float(values["6. volume"]),
            })

        df = pd.DataFrame(records)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_intraday_prices(self, symbol: str, interval: str = "60min") -> pd.DataFrame:
        """Fetch intraday OHLCV data.

        Args:
            symbol: Stock ticker
            interval: 1min, 5min, 15min, 30min, 60min
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "full",
        }

        data = self._request(params)
        key = f"Time Series ({interval})"
        time_series = data.get(key, {})

        if not time_series:
            return pd.DataFrame()

        records = []
        for dt_str, values in time_series.items():
            records.append({
                "date": datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S"),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": float(values["5. volume"]),
            })

        df = pd.DataFrame(records)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_company_overview(self, symbol: str) -> dict:
        """Fetch company overview / profile data."""
        params = {"function": "OVERVIEW", "symbol": symbol}
        return self._request(params)

    def get_income_statement(self, symbol: str) -> dict:
        """Fetch quarterly and annual income statements."""
        params = {"function": "INCOME_STATEMENT", "symbol": symbol}
        return self._request(params)

    def get_balance_sheet(self, symbol: str) -> dict:
        """Fetch quarterly and annual balance sheets."""
        params = {"function": "BALANCE_SHEET", "symbol": symbol}
        return self._request(params)

    def get_cash_flow(self, symbol: str) -> dict:
        """Fetch quarterly and annual cash flow statements."""
        params = {"function": "CASH_FLOW", "symbol": symbol}
        return self._request(params)

    def get_earnings(self, symbol: str) -> dict:
        """Fetch quarterly and annual earnings data."""
        params = {"function": "EARNINGS", "symbol": symbol}
        return self._request(params)

    def search_symbol(self, keywords: str) -> list[dict]:
        """Search for stock symbols by keyword."""
        params = {"function": "SYMBOL_SEARCH", "keywords": keywords}
        data = self._request(params)
        matches = data.get("bestMatches", [])
        return [
            {
                "symbol": m.get("1. symbol"),
                "name": m.get("2. name"),
                "type": m.get("3. type"),
                "region": m.get("4. region"),
                "currency": m.get("8. currency"),
            }
            for m in matches
        ]


# Module-level singleton
av_client = AlphaVantageClient()
