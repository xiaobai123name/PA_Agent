"""Tests for decision continuity (flip cooldown, neutral+AIS, guard)."""
from __future__ import annotations

from pa_agent.ai.decision_continuity import (
    apply_continuity_guard,
    assess_plan_invalidation,
    audit_relation_fields,
    build_continuity_context,
    continuity_violation_reason,
    entries_same_structure,
    order_direction_sign,
    render_continuity_prompt_block,
)
from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame, VolumeMeta


def _frame(*, close: float = 4193.0, high: float = 4194.0, low: float = 4190.0) -> KlineFrame:
    bars = (
        KlineBar(
            seq=1,
            ts_open=1.0,
            open=4192.0,
            high=high,
            low=low,
            close=close,
            volume=1.0,
            closed=True,
        ),
    )
    return KlineFrame(
        volume_meta=VolumeMeta(kind="traded", source="test", unit="test"),
        symbol="XAUUSDm",
        timeframe="5m",
        bars=bars,
        indicators=IndicatorBundle(ema20=(4195.0,), atr14=(2.0,)),
        snapshot_ts_local_ms=1_700_000_300_000,
    )


def test_assess_plan_invalidation_short_stop_hit():
    dec = {"order_direction": "做空", "order_type": "限价单", "stop_loss_price": 4194.0}
    inv, reason = assess_plan_invalidation(dec, _frame(close=4194.5, high=4195.0))
    assert inv is True
    assert "止损" in reason


def test_entries_same_structure_within_ticks():
    assert entries_same_structure(4196.79, 4196.791, tick=0.001) is True
    assert entries_same_structure(4196.79, 4190.0, tick=0.001) is False


def test_continuity_blocks_neutral_ais_long():
    ctx = {
        "direction": "neutral",
        "always_in_branch": "AIS",
        "has_previous_plan": False,
        "cooldown_bars": 3,
    }
    decision = {
        "order_type": "限价单",
        "order_direction": "做多",
        "entry_price": 4190.0,
        "stop_loss_price": 4185.0,
        "take_profit_price": 4196.0,
    }
    reason = continuity_violation_reason(ctx, decision)
    assert reason is not None
    assert "AIS" in reason


def test_continuity_blocks_flip_same_structure():
    ctx = {
        "direction": "bearish",
        "always_in_branch": "AIS",
        "has_previous_plan": True,
        "previous_decision": {
            "order_direction": "\u505a\u7a7a",
            "order_type": "\u9650\u4ef7\u5355",
            "entry_price": 4196.79,
            "stop_loss_price": 4200.0,
        },
        "previous_entry": 4196.79,
        "tick": 0.001,
        "bars_since": 1,
        "cooldown_bars": 3,
        "invalidated": False,
    }
    decision = {
        "order_type": "\u9650\u4ef7\u5355",
        "order_direction": "\u505a\u591a",
        "entry_price": 4196.791,
        "stop_loss_price": 4190.0,
        "take_profit_price": 4204.0,
    }
    reason = continuity_violation_reason(ctx, decision)
    assert reason is not None
    assert "反手" in reason


def test_apply_continuity_guard_forces_no_order():
    ctx = {
        "direction": "neutral",
        "always_in_branch": "AIS",
        "has_previous_plan": False,
        "cooldown_bars": 3,
    }
    stage2 = {
        "decision": {
            "order_type": "限价单",
            "order_direction": "做多",
            "entry_price": 4190.0,
            "stop_loss_price": 4185.0,
            "take_profit_price": 4196.0,
            "reasoning": "test",
        },
        "terminal": {"outcome": "trade", "node_id": "11.2"},
    }
    out = apply_continuity_guard(stage2, ctx)
    assert out["decision"]["order_type"] == "不下单"
    assert out["terminal"]["outcome"] == "wait"


def test_render_prompt_mentions_neutral_ais():
    ctx = build_continuity_context(
        frame=_frame(),
        stage1_json={
            "direction": "neutral",
            "gate_trace": [
                {"node_id": "2.4", "answer": "是", "branch": "AIS"},
            ],
        },
        cooldown_bars=3,
    )
    block = render_continuity_prompt_block(ctx)
    assert "AIS" in block
    assert "做空" in block


def test_build_continuity_context_ignore_previous_skips_record_and_csv():
    previous = {
        "meta": {"timestamp_local_iso": "2026-07-05T10:00:00.000"},
        "stage2_decision": {
            "decision": {
                "order_direction": "\u505a\u7a7a",
                "order_type": "\u9650\u4ef7\u5355",
                "entry_price": 4196.79,
                "stop_loss_price": 4200.0,
            }
        },
    }
    ctx = build_continuity_context(
        frame=_frame(),
        stage1_json={"direction": "bearish"},
        previous_record=previous,
        ignore_previous=True,
    )
    assert ctx["has_previous_plan"] is False
    assert ctx["previous_decision"] == {}
    assert ctx["previous_source"] == "ignored"


def test_normalize_stage2_ignore_previous_context_skips_continuity_guard(monkeypatch):
    from pa_agent.ai.stage2_normalizer import normalize_stage2

    monkeypatch.setattr(
        "pa_agent.ai.decision_nodes.DecisionNodeEngine.apply_stage2",
        staticmethod(lambda *args, **kwargs: None),
    )

    payload = {
        "decision": {
            "order_type": "\u9650\u4ef7\u5355",
            "order_direction": "\u505a\u591a",
            "entry_price": 4190.0,
            "stop_loss_price": 4185.0,
            "take_profit_price": 4196.0,
            "reasoning": "test",
        },
        "terminal": {"outcome": "trade", "node_id": "11.2"},
    }
    stage1 = {
        "direction": "neutral",
        "gate_trace": [
            {"node_id": "2.4", "answer": "\u662f", "branch": "AIS"},
        ],
    }

    guarded = normalize_stage2(payload, kline_frame=_frame(), stage1_json=stage1)
    independent = normalize_stage2(
        payload,
        kline_frame=_frame(),
        stage1_json=stage1,
        ignore_previous_context=True,
    )

    assert guarded["decision"]["order_type"] == "\u4e0d\u4e0b\u5355"
    assert independent["decision"]["order_type"] == "\u9650\u4ef7\u5355"


def test_audit_relation_flip_label():
    prev = {
        "record_time": "2026-06-22 22:49:07",
        "order_direction": "做空",
        "order_type": "限价单",
        "entry_price": "4196.79",
        "stop_loss_price": "4200.657",
    }
    curr = {
        "order_direction": "做多",
        "order_type": "限价单",
        "entry_price": 4196.55,
    }
    audit = audit_relation_fields(prev, curr, frame=_frame(), cooldown_bars=3)
    assert audit["prev_plan_relation"] == "反手"
    assert order_direction_sign("做空") == -1
