from __future__ import annotations

from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame
from pa_agent.util.price_tick import (
    breakout_entry_target,
    format_breakout_tick_hint,
    infer_price_tick_from_frame,
)


def _frame() -> KlineFrame:
    bars = (
        KlineBar(1, 1, 10.10, 10.25, 10.05, 10.20, 1),
        KlineBar(2, 0, 10.00, 10.15, 9.95, 10.10, 1),
    )
    return KlineFrame(
        symbol="XAUUSDT",
        timeframe="15m",
        bars=bars,
        indicators=IndicatorBundle(ema20=(10.0, 10.0), atr14=(0.2, 0.2)),
        snapshot_ts_local_ms=1,
    )


def test_breakout_target_is_exactly_one_tick_outside_extreme() -> None:
    assert breakout_entry_target(
        direction="做多",
        extreme="high",
        basis_high=10.25,
        basis_low=10.05,
        tick=0.01,
    ) == 10.26
    assert breakout_entry_target(
        direction="做空",
        extreme="low",
        basis_high=10.25,
        basis_low=10.05,
        tick=0.01,
    ) == 10.04


def test_tick_hint_states_that_program_rejects_instead_of_rewriting() -> None:
    frame = _frame()
    assert infer_price_tick_from_frame(frame) == 0.01
    hint = format_breakout_tick_hint(frame)
    assert "程序只校验" in hint
    assert "不会修改 entry_price" in hint
