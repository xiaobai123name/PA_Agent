"""Lifecycle tests for the isolated TradingView fetch supervisor."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pa_agent.data.base import DataSourceInvalidSymbolError
from pa_agent.data.tradingview import TradingViewSource
from pa_agent.data.tradingview_process import TradingViewFetchRequest


def _make_source_with_mock_supervisor() -> tuple[TradingViewSource, MagicMock]:
    source = TradingViewSource()
    supervisor = MagicMock()
    source._supervisor = supervisor
    source._connected = True
    source.set_exchange("OANDA")
    source.subscribe("XAUUSD", "15m")
    supervisor.reset_mock()
    return source, supervisor


def test_cancel_pending_terminates_inflight_worker() -> None:
    source, supervisor = _make_source_with_mock_supervisor()

    source.cancel_pending()

    supervisor.cancel_inflight.assert_called_once_with()


def test_disconnect_stops_supervisor_and_clears_connection() -> None:
    source, supervisor = _make_source_with_mock_supervisor()

    source.disconnect()

    supervisor.stop.assert_called_once_with()
    assert source._supervisor is None
    assert source._connected is False


def test_subscribe_cancels_previous_request_and_keeps_clean_symbol() -> None:
    source, supervisor = _make_source_with_mock_supervisor()

    source.set_exchange("BINANCE")
    source.subscribe("BTCUSDT.P", "1h")

    supervisor.cancel_inflight.assert_called_once_with()
    assert source._symbol == "BTCUSDT"
    assert source._timeframe == "1h"


def test_subscribe_rejects_unknown_binance_symbol_before_network() -> None:
    source, supervisor = _make_source_with_mock_supervisor()
    source.set_exchange("BINANCE")

    with pytest.raises(DataSourceInvalidSymbolError, match="仅支持"):
        source.subscribe("SOLUSDT", "15m")

    supervisor.cancel_inflight.assert_not_called()


def test_fetch_delegates_one_request_to_supervisor() -> None:
    source, supervisor = _make_source_with_mock_supervisor()
    rows = [{"datetime": None, "open": 1, "high": 2, "low": 0, "close": 1, "volume": 3}]
    supervisor.fetch.return_value = rows

    out = source._fetch_hist_with_retry(
        symbol="XAUUSD",
        exchange="OANDA",
        interval_name="in_15_minute",
        n_bars=10,
        cancel_token=None,
        timeout_s=12.0,
    )

    assert out is rows
    request = supervisor.fetch.call_args.args[0]
    assert request == TradingViewFetchRequest(
        exchange="OANDA",
        symbol="XAUUSD",
        interval_name="in_15_minute",
        n_bars=10,
    )
    assert supervisor.fetch.call_args.kwargs == {
        "cancel_token": None,
        "timeout_s": 12.0,
    }


def test_subscribe_rejects_unknown_timeframe() -> None:
    source, _supervisor = _make_source_with_mock_supervisor()

    with pytest.raises(ValueError):
        source.subscribe("XAUUSD", "7m")
