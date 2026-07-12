"""TradingView timestamp conversion regressions."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from pa_agent.data.tradingview import _row_ts_ms


def test_tvdatafeed_naive_datetime_is_host_local_epoch() -> None:
    """tvDatafeed converts raw epoch seconds via ``datetime.fromtimestamp``."""
    epoch_ms = 1_718_454_600_000
    row = SimpleNamespace(datetime=datetime.fromtimestamp(epoch_ms / 1000))

    assert _row_ts_ms(row) == epoch_ms


def test_tvdatafeed_pandas_timestamp_is_host_local_epoch() -> None:
    import pytest

    pd = pytest.importorskip("pandas")
    epoch_ms = 1_718_454_600_000
    row = SimpleNamespace(
        datetime=pd.Timestamp(datetime.fromtimestamp(epoch_ms / 1000))
    )

    assert _row_ts_ms(row) == epoch_ms
