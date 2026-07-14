from __future__ import annotations

from pathlib import Path

import pytest

from pa_agent.backtest.models import (
    BacktestRunConfig,
    ContractMetadata,
    FrozenDataset,
)
from pa_agent.backtest.simulator import AmbiguousExecutionError, ExecutionSimulator
from pa_agent.data.base import KlineBar


def _config(*, leverage=5.0):
    metadata = ContractMetadata(
        symbol="BTCUSDT",
        status="TRADING",
        contract_type="PERPETUAL",
        onboard_date_ms=1,
        tick_size=0.1,
        step_size=0.001,
        min_qty=0.001,
        min_notional=5.0,
    )
    dataset = FrozenDataset(
        dataset_id="d",
        dataset_hash="h",
        path=Path("unused"),
        symbol="BTCUSDT",
        analysis_timeframe="15m",
        requested_start_ms=0,
        requested_end_ms=10_000_000,
        target_start_ms=0,
        target_end_ms=10_000_000,
        analysis_bar_count=20,
        metadata=metadata,
    )
    return BacktestRunConfig(dataset=dataset, max_leverage=leverage)


def _decision(intent="pullback"):
    return {
        "order_direction": "做多",
        "entry_intent": intent,
        "order_type": "限价单" if intent == "pullback" else "市价单",
        "entry_price": 100.0,
        "stop_loss_price": 95.0,
        "take_profit_price": 110.0,
        "take_profit_price_2": 115.0,
    }


def _bar(ts, o, h, l, c):
    return KlineBar(1, ts, o, h, l, c, 1.0, closed=True)


def test_limit_fill_then_tp1_full_exit():
    sim = ExecutionSimulator(_config())
    sim.place_from_decision(
        _decision(),
        decision_id="decision",
        created_at_ms=0,
        active_from_ms=60_000,
        valid_bars=3,
    )
    sim.process_analysis_interval(
        [
            _bar(60_000, 101, 102, 99, 101),
            _bar(120_000, 101, 111, 100, 110),
        ],
        close_ms=900_000,
    )
    assert sim.position is None
    assert len(sim.trades) == 1
    assert sim.trades[0].exit_reason == "tp1"
    assert sim.trades[0].r_multiple == pytest.approx(2.0)


def test_same_minute_intrabar_entry_and_exit_is_ambiguous():
    sim = ExecutionSimulator(_config())
    sim.place_from_decision(
        _decision(),
        decision_id="decision",
        created_at_ms=0,
        active_from_ms=60_000,
        valid_bars=3,
    )
    with pytest.raises(AmbiguousExecutionError):
        sim.process_analysis_interval(
            [_bar(60_000, 105, 106, 94, 100)],
            close_ms=900_000,
        )


def test_quantity_is_limited_by_max_leverage_and_exposed():
    sim = ExecutionSimulator(_config(leverage=1.0))
    decision = _decision("immediate")
    decision["stop_loss_price"] = 99.9
    sim.place_from_decision(
        decision,
        decision_id="decision",
        created_at_ms=0,
        active_from_ms=60_000,
        valid_bars=3,
    )
    sim.process_analysis_interval(
        [_bar(60_000, 100, 100.05, 99.95, 100)],
        close_ms=900_000,
    )
    assert sim.position is not None
    assert sim.position.size_limited_by_leverage is True


def test_limit_gap_beyond_stop_exits_at_open_without_inverting_stop():
    sim = ExecutionSimulator(_config())
    sim.place_from_decision(
        _decision(),
        decision_id="decision",
        created_at_ms=0,
        active_from_ms=60_000,
        valid_bars=3,
    )
    sim.process_analysis_interval(
        [_bar(60_000, 94, 95, 93, 94)],
        close_ms=900_000,
    )
    assert sim.position is None
    assert sim.trades[0].exit_reason == "stop_gap"
    assert sim.trades[0].exit_price == 94
