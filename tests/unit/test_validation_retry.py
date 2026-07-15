"""Tests for validation retry policy and feedback."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from pa_agent.ai.json_validator import Ok, ValidationError
from pa_agent.ai.retry_feedback import build_retry_feedback
from pa_agent.ai.retry_policy import (
    detect_cheat,
    extract_feedback_targets,
    should_retry,
)
from pa_agent.ai.stage2_normalizer import ensure_stage2_predictions
from pa_agent.data.base import VolumeMeta
from pa_agent.gui.stage2_payload import prepare_stage2_for_ui
from pa_agent.orchestrator.validation_retry import validate_with_retry


@dataclass
class _FakeErr:
    category: str
    message: str
    missing_fields: list[str]
    invalid_fields: list[str]
    parse_position: str | None = None
    raw_text: str = ""


class _Settings:
    retry_enabled = True
    retry_max = 2
    retry_max_semantic = 1
    retry_stage2 = True
    retry_mode = "standard"


def test_should_retry_format_errors():
    assert should_retry("b", [], ["gate_trace"], attempt=0, settings=_Settings())
    assert not should_retry("c", ["metrics:bad"], [], attempt=0, settings=_Settings())


def test_detect_cheat_immutable_direction():
    before = {"direction": "bullish", "cycle_position": "spike", "gate_result": "proceed"}
    after = {"direction": "bearish", "cycle_position": "spike", "gate_result": "proceed"}
    flags = detect_cheat("stage1", before, after)
    assert any("direction" in f for f in flags)


def test_detect_cheat_no_false_positive_when_program_normalizes_direction():
    """Raw AI direction may differ from post-normalize value; compare normalized copies."""
    from pa_agent.ai.json_validator import JsonValidator
    from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame

    bars = tuple(
        KlineBar(
            seq=i + 1,
            ts_open=float(1_000_000 - (i + 1) * 60_000),
            open=2000.0,
            high=2010.0,
            low=1990.0,
            close=2005.0,
            volume=1.0,
            closed=True,
        )
        for i in range(25)
    )
    frame = KlineFrame(
        volume_meta=VolumeMeta(kind="traded", source="test", unit="test"),
        symbol="TEST",
        timeframe="1h",
        bars=bars,
        snapshot_ts_local_ms=1,
        indicators=IndicatorBundle(
            ema20=tuple([2000.0] * 25),
            atr14=tuple([10.0] * 25),
        ),
    )
    validator = JsonValidator()
    raw = {
        "direction": "bearish",
        "cycle_position": "broad_channel",
        "gate_result": "proceed",
        "gate_trace": [],
    }
    before_norm = validator.normalize_parsed("stage1", raw, kline_frame=frame)
    after_norm = validator.normalize_parsed("stage1", dict(raw), kline_frame=frame)
    flags = detect_cheat("stage1", before_norm, after_norm)
    assert not flags


def test_build_retry_feedback_contains_stage():
    err = _FakeErr("b", "missing", ["next_bar_prediction"], [], None, "{}")
    text = build_retry_feedback(err, stage="stage2", attempt=1, max_attempts=2)
    assert "next_bar_prediction" in text
    assert "阶段二" in text


def test_retry_feedback_preserves_exact_missing_array_path():
    path = "bar_by_bar_summary[4].trapped_side"
    err = _FakeErr("b", "missing", [path], [], None, "{}")

    text = build_retry_feedback(err, stage="stage1", attempt=1, max_attempts=1)

    assert f"[缺少] {path}" in text
    assert path in extract_feedback_targets([], [path])


def test_ensure_stage2_predictions_for_old_record():
    s2 = {
        "decision": {"order_type": "不下单", "reasoning": "等待"},
        "diagnosis_summary": {"cycle_position": "broad_channel", "direction": "neutral"},
        "decision_trace": [],
        "terminal": {"node_id": "9.0", "outcome": "wait", "label": "x"},
    }
    assert ensure_stage2_predictions(s2) is True
    assert isinstance(s2.get("next_bar_prediction"), dict)
    assert isinstance(s2.get("next_cycle_prediction"), dict)


def test_prepare_stage2_for_ui_merges_predictions():
    s2 = {
        "decision": {"order_type": "不下单"},
        "diagnosis_summary": {"cycle_position": "broad_channel", "direction": "neutral"},
    }
    payload = prepare_stage2_for_ui(s2)
    assert "next_bar_prediction" in payload
    assert "next_cycle_prediction" in payload


class _FormatOnlySettings:
    retry_enabled = True
    retry_max = 1
    retry_max_semantic = 1
    retry_stage2 = True
    retry_mode = "format_only"


class _SequenceValidator:
    def __init__(self, results):
        self._results = iter(results)

    def validate(self, _stage, _content, **_kwargs):
        return next(self._results)

    @staticmethod
    def normalize_parsed(_stage, value, **_kwargs):
        return value


def _reply(content: str):
    return SimpleNamespace(content=content, reasoning_content="")


def test_format_only_retry_succeeds_once_and_preserves_failure_audit():
    first = ValidationError(
        category="c",
        stage="stage2",
        raw_text='{"decision": {}}',
        invalid_fields=["decision_trace[0].node_id='1.2' 超出 Stage 2 范围"],
        message="非法 Stage 2 节点",
        retryable_format=True,
    )
    validator = _SequenceValidator([first, Ok(obj={"decision": {}})])
    calls = []

    result = validate_with_retry(
        stage="stage2",
        messages=[{"role": "user", "content": "prompt"}],
        reply=_reply(first.raw_text),
        validator=validator,
        validation_settings=_FormatOnlySettings(),
        validate_kwargs={},
        call_api=lambda messages: calls.append(messages) or _reply('{"decision": {}}'),
    )

    assert isinstance(result.result, Ok)
    assert result.attempts == 2
    assert len(result.failures) == 1
    assert result.failures[0].raw_text == first.raw_text
    assert "Stage 2 决策路径范围" in (result.failures[0].feedback or "")
    assert len(calls) == 1


def test_format_only_retry_stops_after_second_format_failure():
    failures = [
        ValidationError(
            category="a",
            stage="stage2",
            raw_text="{bad",
            message="JSON syntax error",
            retryable_format=True,
        ),
        ValidationError(
            category="c",
            stage="stage2",
            raw_text='{"answer":"极速"}',
            invalid_fields=["decision_trace.0.answer"],
            message="invalid enum",
            retryable_format=True,
        ),
    ]
    result = validate_with_retry(
        stage="stage2",
        messages=[],
        reply=_reply("{bad"),
        validator=_SequenceValidator(failures),
        validation_settings=_FormatOnlySettings(),
        validate_kwargs={},
        call_api=lambda _messages: _reply('{"answer":"极速"}'),
    )

    assert isinstance(result.result, ValidationError)
    assert result.attempts == 2
    assert [item.attempt for item in result.failures] == [1, 2]
    assert result.failures[1].feedback is None


def test_format_only_mode_does_not_retry_semantic_failure():
    semantic = ValidationError(
        category="c",
        stage="stage2",
        raw_text="{}",
        invalid_fields=["metrics:stop loss lacks structure basis"],
        message="semantic failure",
        retryable_format=False,
    )
    calls = []
    result = validate_with_retry(
        stage="stage2",
        messages=[],
        reply=_reply("{}"),
        validator=_SequenceValidator([semantic]),
        validation_settings=_FormatOnlySettings(),
        validate_kwargs={},
        call_api=lambda messages: calls.append(messages),
    )

    assert isinstance(result.result, ValidationError)
    assert result.attempts == 1
    assert calls == []


def test_stage2_retry_rejects_unrequested_trade_plan_changes():
    before = {
        "decision": {
            "order_action": "place",
            "order_type": "限价单",
            "entry_price": 100.0,
            "stop_loss_price": 95.0,
            "take_profit_price": 110.0,
        }
    }
    after = {
        "decision": {
            "order_action": "none",
            "order_type": "不下单",
            "entry_price": None,
            "stop_loss_price": None,
            "take_profit_price": None,
        }
    }

    flags = detect_cheat("stage2", before, after, feedback_mentioned={"decision_trace"})

    assert any("decision.order_action" in flag for flag in flags)
    assert any("decision.entry_price" in flag for flag in flags)
