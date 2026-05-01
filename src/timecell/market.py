"""Task 2 — Live market data fetcher.

- Routes stocks/indices/ETFs to Yahoo Finance (yfinance), crypto to CoinGecko.
- Fetches in parallel via asyncio.gather.
- Each provider is wrapped with tenacity exponential backoff (3 attempts).
- Per-symbol failure is isolated: one provider going down does not crash the rest.
- IST timestamp printed on the table.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DEFAULT_SYMBOLS = ["^NSEI", "RELIANCE.NS", "BTC"]

# Map common crypto tickers → CoinGecko IDs.
_COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "MATIC": "matic-network",
    "DOGE": "dogecoin",
}


@dataclass
class Quote:
    asset: str
    price: float | None
    currency: str
    provider: str
    fetched_at: datetime
    ok: bool
    error: str | None = None


def _classify(symbol: str) -> Literal["crypto", "yahoo"]:
    return "crypto" if symbol.upper() in _COINGECKO_IDS else "yahoo"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
async def _fetch_yahoo(client: httpx.AsyncClient, symbol: str) -> tuple[float, str]:
    """yfinance is sync; run it in a thread to keep the gather non-blocking."""
    import yfinance as yf  # heavy import — defer

    def _sync() -> tuple[float, str]:
        ticker = yf.Ticker(symbol)
        # `fast_info` is faster than `info` and returns last_price + currency.
        fi = ticker.fast_info
        price = fi.get("last_price") or fi.get("lastPrice")
        if price is None:
            hist = ticker.history(period="1d")
            if hist.empty:
                raise ValueError(f"no price data for {symbol}")
            price = float(hist["Close"].iloc[-1])
        currency = fi.get("currency") or "INR"
        return float(price), str(currency).upper()

    return await asyncio.to_thread(_sync)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
async def _fetch_coingecko(client: httpx.AsyncClient, symbol: str) -> tuple[float, str]:
    cg_id = _COINGECKO_IDS[symbol.upper()]
    r = await client.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": cg_id, "vs_currencies": "usd"},
    )
    r.raise_for_status()
    payload = r.json()
    if cg_id not in payload or "usd" not in payload[cg_id]:
        raise ValueError(f"coingecko returned no usd price for {cg_id}")
    return float(payload[cg_id]["usd"]), "USD"


async def _fetch_one(client: httpx.AsyncClient, symbol: str) -> Quote:
    """Dispatch + isolate failures so one bad symbol does not kill the run."""
    now = datetime.now(IST)
    provider = _classify(symbol)
    try:
        if provider == "crypto":
            price, currency = await _fetch_coingecko(client, symbol)
            return Quote(symbol, price, currency, "coingecko", now, ok=True)
        price, currency = await _fetch_yahoo(client, symbol)
        return Quote(symbol, price, currency, "yfinance", now, ok=True)
    except Exception as exc:  # noqa: BLE001 — graceful failure is the whole point
        log.warning("fetch failed for %s: %s", symbol, exc)
        return Quote(
            asset=symbol,
            price=None,
            currency="—",
            provider=provider,
            fetched_at=now,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def fetch_quotes(symbols: list[str], timeout: float = 10.0) -> list[Quote]:
    """Fetch all symbols in parallel. Per-symbol errors do not propagate."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await asyncio.gather(*(_fetch_one(client, s) for s in symbols))


def render_quotes_table(quotes: list[Quote]) -> str:
    """Rich-free fallback renderer for log output / non-Rich consumers."""
    when = quotes[0].fetched_at.strftime("%Y-%m-%d %H:%M:%S IST") if quotes else "—"
    lines = [f"Asset Prices — fetched at {when}"]
    lines.append(f"{'Asset':<14} {'Price':>14} {'Currency':<10} {'Provider':<10} {'Status':<6}")
    lines.append("─" * 60)
    for q in quotes:
        price_str = f"{q.price:,.2f}" if q.ok and q.price is not None else "—"
        status = "OK" if q.ok else (q.error or "FAIL")[:30]
        lines.append(f"{q.asset:<14} {price_str:>14} {q.currency:<10} {q.provider:<10} {status:<6}")
    return "\n".join(lines)
