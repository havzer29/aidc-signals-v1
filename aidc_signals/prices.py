"""Price utilities using yfinance."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pylint: disable=broad-except
    yf = None


def fetch_price_history(ticker: str, start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required for price fetching")
    data = yf.download(ticker, start=start, end=end or datetime.utcnow())
    if data.empty:
        raise ValueError(f"No price data for {ticker}")
    return data


__all__ = ["fetch_price_history"]
