"""Fetch daily log-return series for VaR/Monte Carlo (Task 5).

Routes everything through yfinance — it has both equity and crypto coverage,
and avoids CoinGecko's market_chart endpoint which now requires a paid plan
for daily-interval data. Cash-equivalents and unknown names return an empty
array, which var.py treats as a zero-return column.

A small alias map turns friendly names ('BTC', 'NIFTY50', 'GOLD') into the
yfinance tickers that resolve in INR contexts. Users can still pass real
tickers directly — anything not in the alias map is forwarded verbatim.

Failures are isolated: if BTC fetch flakes, NIFTY history still flows through.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

# Names treated as zero-volatility cash (no fetch attempted).
_CASH_NAMES = {"CASH", "FD", "DEPOSIT", "SAVINGS", "LIQUID"}

# Friendly-name → yfinance ticker map. Keep INR-relevant where possible:
#   - GOLDBEES.NS = Nippon India ETF Gold BeES (INR-priced gold ETF on NSE).
#   - ^NSEI = Nifty 50 index.
#   - BTC-USD / ETH-USD = yfinance crypto tickers (USD-priced).
#   - SP500 / NASDAQ aliases included for convenience.
_TICKER_ALIASES: dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "MATIC": "MATIC-USD",
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "GOLD": "GOLDBEES.NS",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
}


def _resolve_ticker(name: str) -> str:
    return _TICKER_ALIASES.get(name.upper(), name)


def _to_log_returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size < 2:
        return np.array([])
    return np.diff(np.log(p))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
def _fetch_yahoo_returns(ticker: str, years: int) -> np.ndarray:
    import yfinance as yf  # heavy import — defer

    hist = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"no history for {ticker}")
    return _to_log_returns(hist["Close"].to_numpy())


def fetch_returns(asset_names: Iterable[str], years: int = 5) -> dict[str, np.ndarray]:
    """Map each asset name to its daily log-return array.

    Cash and fetch failures map to an empty array; var.py interprets that as
    a zero-return column and flags the asset in `assets_synthetic`.
    """
    out: dict[str, np.ndarray] = {}
    for name in asset_names:
        if name.upper() in _CASH_NAMES:
            out[name] = np.array([])
            continue
        ticker = _resolve_ticker(name)
        try:
            out[name] = _fetch_yahoo_returns(ticker, years)
        except Exception as exc:  # noqa: BLE001 — graceful failure is the whole point
            log.warning("history fetch failed for %s (%s): %s", name, ticker, exc)
            out[name] = np.array([])
    return out
