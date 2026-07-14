"""Thread-safe live quote snapshots derived from the forming K-line."""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class LiveQuote:
    symbol: str
    timeframe: str
    last_price: float
    received_at_ms: int
    source: str = "forming_bar_close"


class LiveQuoteStore:
    """Keep the newest forming-bar price without sharing Qt objects."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quotes: dict[tuple[str, str], LiveQuote] = {}

    @staticmethod
    def _key(symbol: str, timeframe: str) -> tuple[str, str]:
        return symbol.strip().upper(), timeframe.strip().lower()

    def update_from_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: Iterable[Any],
        *,
        received_at_ms: int | None = None,
    ) -> LiveQuote | None:
        rows = list(bars)
        if not rows:
            return None
        newest = rows[0]
        if bool(getattr(newest, "closed", True)):
            return None
        try:
            price = float(getattr(newest, "close"))
        except (TypeError, ValueError, AttributeError):
            return None
        if not math.isfinite(price) or price <= 0:
            return None
        quote = LiveQuote(
            symbol=symbol.strip(),
            timeframe=timeframe.strip(),
            last_price=price,
            received_at_ms=int(
                time.time() * 1000 if received_at_ms is None else received_at_ms
            ),
        )
        with self._lock:
            self._quotes[self._key(symbol, timeframe)] = quote
        return quote

    def get(self, symbol: str, timeframe: str) -> LiveQuote | None:
        with self._lock:
            return self._quotes.get(self._key(symbol, timeframe))

    def clear(self, symbol: str | None = None, timeframe: str | None = None) -> None:
        if (symbol is None) != (timeframe is None):
            raise ValueError("symbol and timeframe must be provided together")
        with self._lock:
            if symbol is None:
                self._quotes.clear()
            else:
                assert timeframe is not None
                self._quotes.pop(self._key(symbol, timeframe), None)
