from __future__ import annotations

import pytest

from pa_agent.ai.execution_resolver import ExecutionPolicy, resolve_stage2_execution
from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame, VolumeMeta
from pa_agent.data.live_quote import LiveQuote

NOW_MS = 10_000


def _frame() -> KlineFrame:
    bars = (
        KlineBar(0, 2_000, 100.00, 100.10, 99.90, 100.00, 1, closed=False),
        KlineBar(1, 1_000, 99.80, 100.20, 99.70, 100.00, 1, closed=True),
    )
    return KlineFrame(
        volume_meta=VolumeMeta(kind="traded", source="test", unit="test"),
        symbol="BTCUSDT",
        timeframe="15m",
        bars=bars,
        indicators=IndicatorBundle(ema20=(99.9, 99.8), atr14=(2.0, 2.0)),
        snapshot_ts_local_ms=NOW_MS,
    )


def _quote(price: float = 100.0, *, received_at_ms: int = 9_900) -> LiveQuote:
    return LiveQuote("BTCUSDT", "15m", price, received_at_ms)


def _stage2(intent: str, order_type: str, direction: str | None, entry: float | None) -> dict:
    return {
        "decision": {
            "entry_intent": intent,
            "order_type": order_type,
            "order_direction": direction,
            "entry_price": entry,
            "take_profit_price": 110.0 if direction == "做多" else 90.0,
            "take_profit_price_2": 115.0 if direction == "做多" else 85.0,
            "stop_loss_price": 95.0 if direction == "做多" else 105.0,
            "estimated_win_rate": 55,
            "estimated_win_rate_reasoning": "结构成立",
            "high_rr_review": None,
            "reasoning": "结构方案",
            "entry_basis_bar": "K1" if intent == "breakout" else None,
            "entry_basis_extreme": (
                "high" if intent == "breakout" and direction == "做多"
                else "low" if intent == "breakout"
                else None
            ),
            "entry_rule": "依据极值外一跳" if intent == "breakout" else None,
        },
        "decision_trace": [
            {
                "node_id": "10.3",
                "answer": "是",
                "reason": "entry/stop/target 已验证",
                "bar_range": "K1",
            }
        ],
        "terminal": {"node_id": "10.3", "outcome": "trade", "label": "待解析"},
    }


@pytest.mark.parametrize(
    ("intent", "order_type", "direction", "entry", "expected_node"),
    [
        ("pullback", "限价单", "做多", 99.0, "11.3"),
        ("pullback", "限价单", "做空", 101.0, "11.3"),
        ("breakout", "突破单", "做多", 101.0, "11.2"),
        ("breakout", "突破单", "做空", 99.0, "11.2"),
        ("immediate", "市价单", "做多", 100.1, "11.1"),
        ("immediate", "市价单", "做空", 99.9, "11.1"),
    ],
)
def test_resolves_each_trade_intent_without_switching_type(
    intent: str,
    order_type: str,
    direction: str,
    entry: float,
    expected_node: str,
) -> None:
    out = resolve_stage2_execution(
        _stage2(intent, order_type, direction, entry),
        frame=_frame(),
        quote=_quote(),
        now_ms=NOW_MS,
    )

    decision = out["decision"]
    assert decision["order_type"] == order_type
    assert decision["execution_review"]["status"] == "resolved"
    assert out["terminal"]["node_id"] == expected_node


def test_none_intent_needs_no_quote_and_adds_no_execution_node() -> None:
    payload = _stage2("none", "不下单", None, None)
    payload["decision_trace"].append({"node_id": "11.3"})
    out = resolve_stage2_execution(payload, frame=_frame(), quote=None, now_ms=NOW_MS)

    assert out["decision"]["execution_review"]["status"] == "not_applicable"
    assert all(
        not str(item.get("node_id", "")).startswith("11.")
        for item in out["decision_trace"]
    )


@pytest.mark.parametrize(
    ("intent", "order_type", "direction", "entry", "reason_code"),
    [
        ("pullback", "限价单", "做多", 101.0, "pullback_entry_not_pending"),
        ("pullback", "限价单", "做空", 99.0, "pullback_entry_not_pending"),
        ("breakout", "突破单", "做多", 99.0, "breakout_trigger_already_crossed"),
        ("breakout", "突破单", "做空", 101.0, "breakout_trigger_already_crossed"),
    ],
)
def test_invalid_price_relation_rejects_instead_of_switching_type(
    intent: str,
    order_type: str,
    direction: str,
    entry: float,
    reason_code: str,
) -> None:
    out = resolve_stage2_execution(
        _stage2(intent, order_type, direction, entry),
        frame=_frame(),
        quote=_quote(),
        now_ms=NOW_MS,
    )

    review = out["decision"]["execution_review"]
    assert out["decision"]["order_type"] == "不下单"
    assert review["reason_code"] == reason_code
    assert review["proposed_order_type"] == order_type
    assert review["proposed_structure"]["entry_price"] == entry


def test_declared_type_mismatch_is_rejected_without_quote() -> None:
    out = resolve_stage2_execution(
        _stage2("breakout", "限价单", "做多", 101.0),
        frame=_frame(),
        quote=None,
        now_ms=NOW_MS,
    )
    assert out["decision"]["execution_review"]["reason_code"] == (
        "declared_order_type_mismatch"
    )


@pytest.mark.parametrize(
    ("quote", "reason_code"),
    [
        (None, "live_quote_unavailable"),
        (_quote(received_at_ms=6_000), "stale_live_quote"),
        (LiveQuote("ETHUSDT", "15m", 100.0, 9_900), "live_quote_identity_mismatch"),
    ],
)
def test_missing_stale_or_wrong_quote_is_rejected(
    quote: LiveQuote | None,
    reason_code: str,
) -> None:
    out = resolve_stage2_execution(
        _stage2("pullback", "限价单", "做多", 99.0),
        frame=_frame(),
        quote=quote,
        now_ms=NOW_MS,
    )
    assert out["decision"]["execution_review"]["reason_code"] == reason_code


def test_immediate_slippage_limit_rejects_without_converting_to_pending_order() -> None:
    out = resolve_stage2_execution(
        _stage2("immediate", "市价单", "做多", 99.0),
        frame=_frame(),
        quote=_quote(100.0),
        policy=ExecutionPolicy(
            quote_max_age_ms=3_000,
            immediate_max_slippage_atr=0.1,
            immediate_max_slippage_ticks=3,
        ),
        now_ms=NOW_MS,
    )
    review = out["decision"]["execution_review"]
    assert review["reason_code"] == "immediate_entry_missed"
    assert review["proposed_order_type"] == "市价单"
    assert review["resolved_order_type"] == "不下单"


@pytest.mark.parametrize("entry", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_entry_number_is_explicitly_rejected(entry: float) -> None:
    out = resolve_stage2_execution(
        _stage2("immediate", "市价单", "做多", entry),
        frame=_frame(),
        quote=_quote(),
        now_ms=NOW_MS,
    )
    assert out["decision"]["execution_review"]["reason_code"] == "invalid_entry_price"


def test_non_numeric_live_quote_is_explicitly_rejected() -> None:
    quote = LiveQuote("BTCUSDT", "15m", "bad", 9_900)  # type: ignore[arg-type]
    out = resolve_stage2_execution(
        _stage2("pullback", "限价单", "做多", 99.0),
        frame=_frame(),
        quote=quote,
        now_ms=NOW_MS,
    )
    review = out["decision"]["execution_review"]
    assert review["reason_code"] == "invalid_live_quote"
    assert review["market_price"] is None
