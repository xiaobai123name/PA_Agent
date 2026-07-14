from __future__ import annotations

import copy
import json

import pytest

from pa_agent.ai.execution_resolver import resolve_stage2_execution
from pa_agent.ai.json_validator import JsonValidator, Ok
from pa_agent.backtest.lifecycle import (
    BacktestDecisionError,
    render_lifecycle_prompt,
    validate_lifecycle_decision,
)
from pa_agent.backtest.models import OrderAction
from pa_agent.config.settings import Settings
from pa_agent.data.live_quote import LiveQuote
from tests.integration.conftest import VALID_STAGE1, VALID_STAGE2, make_frame


def _stage2(action: str, *, valid_bars=None, status="not_applicable", trade=False):
    decision = {
        "order_action": action,
        "order_valid_bars": valid_bars,
        "entry_intent": "immediate" if trade else "none",
        "order_type": "市价单" if trade else "不下单",
        "entry_price": 100.0 if trade else None,
        "stop_loss_price": 99.0 if trade else None,
        "take_profit_price": 102.0 if trade else None,
        "take_profit_price_2": 103.0 if trade else None,
        "order_direction": "做多" if trade else None,
        "execution_review": {"status": status},
    }
    return {"decision": decision}


_PENDING = {
    "entry_intent": "immediate",
    "order_type": "市价单",
    "direction": "做多",
    "entry_price": 100.0,
    "stop_price": 99.0,
    "tp1_price": 102.0,
    "tp2_price": 103.0,
}


@pytest.mark.parametrize(
    ("action", "has_pending", "trade", "status", "valid"),
    [
        ("place", False, True, "resolved", 3),
        ("replace", True, True, "resolved", 5),
        ("keep", True, True, "resolved", None),
        ("cancel", True, False, "not_applicable", None),
        ("none", False, False, "not_applicable", None),
    ],
)
def test_valid_lifecycle_actions(action, has_pending, trade, status, valid):
    result = validate_lifecycle_decision(
        _stage2(action, valid_bars=valid, status=status, trade=trade),
        pending_order=_PENDING if has_pending else None,
    )
    assert result.action == OrderAction(action)
    assert result.valid_bars == valid


def test_keep_without_pending_is_exposed():
    with pytest.raises(BacktestDecisionError, match="没有挂单"):
        validate_lifecycle_decision(
            _stage2("keep", trade=True, status="resolved"),
            pending_order=None,
        )


def test_place_requires_explicit_validity():
    with pytest.raises(BacktestDecisionError, match="1..100"):
        validate_lifecycle_decision(
            _stage2("place", trade=True, status="resolved"),
            pending_order=None,
        )


def test_none_does_not_silently_cancel_pending():
    with pytest.raises(BacktestDecisionError, match="已有挂单"):
        validate_lifecycle_decision(
            _stage2("none"),
            pending_order=_PENDING,
        )


def test_keep_must_repeat_pending_order_exactly():
    stage2 = _stage2("keep", trade=True, status="resolved")
    stage2["decision"]["entry_price"] = 100.1
    with pytest.raises(BacktestDecisionError, match="原样重复"):
        validate_lifecycle_decision(stage2, pending_order=_PENDING)


def test_keep_with_repeated_pending_prices_passes_standard_schema_and_lifecycle():
    payload = copy.deepcopy(VALID_STAGE2)
    payload["decision"]["order_action"] = "keep"
    payload["decision"]["order_valid_bars"] = None
    frame = make_frame()
    validated = JsonValidator(Settings()).validate(
        "stage2",
        json.dumps(payload, ensure_ascii=False),
        kline_frame=frame,
        stage1_json=VALID_STAGE1,
        skip_next_bar=True,
        ignore_previous_context=True,
    )
    assert isinstance(validated, Ok)
    resolved = resolve_stage2_execution(
        validated.obj,
        frame=frame,
        quote=LiveQuote(
            symbol=frame.symbol,
            timeframe=frame.timeframe,
            last_price=2040.0,
            received_at_ms=10_000,
        ),
        now_ms=10_000,
    )
    pending = {
        "entry_intent": "breakout",
        "order_type": "突破单",
        "direction": "做多",
        "entry_price": 2047.0,
        "stop_price": 2030.0,
        "tp1_price": 2064.0,
        "tp2_price": 2081.0,
    }
    lifecycle = validate_lifecycle_decision(resolved, pending_order=pending)
    assert lifecycle.action == OrderAction.KEEP
    assert lifecycle.execution_status == "resolved"


def test_pending_prompt_requires_keep_to_repeat_existing_order():
    prompt = render_lifecycle_prompt(_PENDING)
    assert "必须把当前挂单" in prompt
    assert "原样重复" in prompt
