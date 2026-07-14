from __future__ import annotations

import math

import pytest

from pa_agent.data.base import KlineBar
from pa_agent.data.live_quote import LiveQuoteStore


def _bar(*, close: float, closed: bool) -> KlineBar:
    return KlineBar(
        seq=0 if not closed else 1,
        ts_open=1_000,
        open=100.0,
        high=101.0,
        low=99.0,
        close=close,
        volume=1.0,
        closed=closed,
    )


def test_store_accepts_only_newest_forming_bar_close() -> None:
    store = LiveQuoteStore()
    quote = store.update_from_bars(
        "btcusdt",
        "15M",
        [_bar(close=100.25, closed=False), _bar(close=99.0, closed=True)],
        received_at_ms=2_000,
    )

    assert quote is not None
    assert quote.last_price == 100.25
    assert store.get("BTCUSDT", "15m") == quote


@pytest.mark.parametrize("price", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_store_rejects_invalid_forming_prices(price: float) -> None:
    store = LiveQuoteStore()
    assert (
        store.update_from_bars(
            "BTCUSDT",
            "15m",
            [_bar(close=price, closed=False)],
            received_at_ms=2_000,
        )
        is None
    )
    assert store.get("BTCUSDT", "15m") is None


def test_closed_bar_does_not_replace_live_quote() -> None:
    store = LiveQuoteStore()
    first = store.update_from_bars(
        "BTCUSDT",
        "15m",
        [_bar(close=100.25, closed=False)],
        received_at_ms=2_000,
    )

    assert store.update_from_bars(
        "BTCUSDT",
        "15m",
        [_bar(close=101.0, closed=True)],
        received_at_ms=3_000,
    ) is None
    assert store.get("BTCUSDT", "15m") == first


def test_clear_requires_complete_identity() -> None:
    store = LiveQuoteStore()
    with pytest.raises(ValueError, match="provided together"):
        store.clear(symbol="BTCUSDT")

