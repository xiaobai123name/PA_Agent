from __future__ import annotations

from pathlib import Path

import pytest

from pa_agent.backtest.historical_data import HistoricalDataError, HistoricalDataRepository


def _rows(start: int, count: int, step: int, base: float = 100.0):
    rows = []
    for i in range(count):
        ts = start + i * step
        price = base + i * 0.1
        rows.append([ts, str(price), str(price + 1), str(price - 1), str(price + 0.2), "10"])
    return rows


class _FakeBinance:
    def __init__(self, analysis_rows, execution_rows, *, onboard=1_000_000):
        self.analysis_rows = analysis_rows
        self.execution_rows = execution_rows
        self.onboard = onboard

    def get_json(self, path, params=None):
        if path.endswith("/time"):
            return {"serverTime": 99_999_999_999}
        if path.endswith("exchangeInfo"):
            return {
                "serverTime": 99_999_999_999,
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "onboardDate": self.onboard,
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                            {
                                "filterType": "LOT_SIZE",
                                "stepSize": "0.001",
                                "minQty": "0.001",
                            },
                            {"filterType": "MIN_NOTIONAL", "notional": "5"},
                        ],
                    }
                ],
            }
        interval = params["interval"]
        source = self.execution_rows if interval == "1m" else self.analysis_rows
        end = int(params.get("endTime", 2**63 - 1))
        start = params.get("startTime")
        limit = int(params["limit"])
        selected = [row for row in source if row[0] <= end]
        if start is not None:
            selected = [row for row in selected if row[0] >= int(start)]
            return selected[:limit]
        return selected[-limit:]


def test_download_freezes_and_reloads_dataset(tmp_path: Path):
    tf_ms = 15 * 60_000
    minute = 60_000
    history_start = 1_800_000
    analysis = _rows(history_start, 90, tf_ms)
    target_start = analysis[75][0]
    target_end = target_start + 5 * tf_ms
    execution = _rows(target_start, 5 * 15, minute, 200.0)
    repo = HistoricalDataRepository(
        tmp_path,
        client=_FakeBinance(analysis, execution),
    )

    dataset = repo.download_and_freeze(
        "BTCUSDT",
        "15m",
        target_start,
        target_end,
        analysis_bar_count=20,
    )

    assert dataset.path.is_file()
    assert len(dataset.dataset_hash) == 64
    loaded_analysis = repo.load_bars(dataset, "15m")
    loaded_execution = repo.load_bars(dataset, "1m")
    assert loaded_analysis[-1].ts_open < target_end
    assert len(loaded_execution) == 75

    same = repo.download_and_freeze(
        "BTCUSDT",
        "15m",
        target_start,
        target_end,
        analysis_bar_count=20,
    )
    assert same.dataset_id == dataset.dataset_id


def test_request_before_listing_fails_instead_of_truncating(tmp_path: Path):
    repo = HistoricalDataRepository(
        tmp_path,
        client=_FakeBinance([], [], onboard=5_000_000),
    )
    with pytest.raises(HistoricalDataError, match="早于"):
        repo.download_and_freeze(
            "BTCUSDT",
            "15m",
            3_600_000,
            5_400_000,
            analysis_bar_count=20,
        )


def test_perpetual_minute_gap_is_not_hidden():
    rows = _rows(1_000_000, 3, 60_000)
    rows[2][0] += 60_000
    with pytest.raises(HistoricalDataError, match="缺口"):
        HistoricalDataRepository._validate_rows(rows, "1m", allow_gaps=False)


def test_unaligned_range_is_rejected(tmp_path: Path):
    repo = HistoricalDataRepository(
        tmp_path,
        client=_FakeBinance([], []),
    )
    with pytest.raises(HistoricalDataError, match="对齐"):
        repo.download_and_freeze(
            "BTCUSDT",
            "15m",
            1_800_001,
            2_700_000,
            analysis_bar_count=20,
        )
