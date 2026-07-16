"""Regression tests for the immutable K1 timing anchor in follow-up chat."""
from __future__ import annotations

from unittest.mock import MagicMock

from pa_agent.orchestrator.free_chat import FreeChatSession
from pa_agent.records.schema import AnalysisRecord, RecordMeta
from pa_agent.util.threading import CancelToken


def _make_record() -> AnalysisRecord:
    return AnalysisRecord(
        meta=RecordMeta(
            timestamp_local_iso="2026-07-16T07:31:16.826",
            timestamp_local_ms=1_784_158_276_826,
            symbol="BTCUSDT",
            timeframe="15m",
            bar_count=2,
            ai_provider={"model": "test-model"},
        ),
        kline_data=[
            {
                "seq": 1,
                "ts_open": 1_784_157_300_000,
                "open": 64880.5,
                "high": 64902.4,
                "low": 64810.3,
                "close": 64819.2,
                "volume": 239.64,
                "closed": True,
            }
        ],
        htf_text="",
        stage1_messages=[],
        stage1_response=None,
        stage1_diagnosis=None,
        stage2_messages=[],
        stage2_response=None,
        stage2_decision={
            "decision": {
                "order_type": "限价单",
                "order_direction": "做空",
                "entry_price": 64900.0,
                "take_profit_price": 64825.0,
                "take_profit_price_2": 64739.0,
                "stop_loss_price": 64955.0,
            },
            "bar_analysis": {
                "entry_bar": {
                    "freshness": "pending",
                    "strength": "not_triggered",
                }
            },
        },
        strategy_files_used=[],
        experience_loaded=[],
        exception=None,
        usage_total={},
    )


def _make_reply() -> MagicMock:
    reply = MagicMock()
    reply.content = "仍待确认"
    reply.reasoning_content = ""
    reply.usage.prompt_tokens = 100
    reply.usage.cached_prompt_tokens = 0
    reply.usage.completion_tokens = 10
    reply.usage.total_tokens = 110
    return reply


def test_followup_prompt_injects_original_k1_timing_anchor() -> None:
    client = MagicMock()
    client.stream_chat.return_value = _make_reply()
    session = FreeChatSession(
        base_record=_make_record(),
        client=client,
        assembler=MagicMock(),
        pending_writer=MagicMock(),
        ledger=MagicMock(),
        kline_snapshot_fn=lambda: "1 | 2026-07-16 07:30 | 64800 | 65000",
    )

    session.send("现在成交了吗？", CancelToken())

    messages: list[dict] = client.stream_chat.call_args[0][0]
    prompt = messages[-1]["content"]
    assert "订单时序硬锚点（程序生成，必须严格遵守）" in prompt
    assert "ts_open=1784157300000" in prompt
    assert "H=64902.4" in prompt
    assert "订单生成时点：上述原始分析K1收盘之后" in prompt
    assert "限价单 / 做空 / entry=64900.0" in prompt
    assert "freshness=pending, strength=not_triggered" in prompt
    assert "ts_open > 1784157300000" in prompt
    assert "不得用于判定本订单已成交、已止盈或已止损" in prompt
    assert "当前图表K线数据" in prompt
    assert prompt.endswith("现在成交了吗？")


def test_system_prompt_marks_timing_anchor_as_highest_priority() -> None:
    session = FreeChatSession(
        base_record=_make_record(),
        client=MagicMock(),
        assembler=MagicMock(),
        pending_writer=MagicMock(),
        ledger=MagicMock(),
    )

    system_prompt = session._cached_prefix[0]["content"]
    assert "订单时序硬锚点" in system_prompt
    assert "禁止使用原始分析K1或更早K线" in system_prompt


def test_record_without_kline_data_keeps_followup_text_unchanged() -> None:
    client = MagicMock()
    client.stream_chat.return_value = _make_reply()
    session = FreeChatSession(
        base_record=_make_record().model_copy(update={"kline_data": []}),
        client=client,
        assembler=MagicMock(),
        pending_writer=MagicMock(),
        ledger=MagicMock(),
    )

    session.send("普通追问", CancelToken())

    messages: list[dict] = client.stream_chat.call_args[0][0]
    assert messages[-1]["content"] == "普通追问"
