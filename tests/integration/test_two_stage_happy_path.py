"""Integration test: happy path — both stages succeed.

Task 11.4
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from tests.fixtures.validators import schema_test_validator
from pa_agent.ai.router import route_strategy_files
from pa_agent.config.settings import Settings
from pa_agent.data.live_quote import LiveQuote
from pa_agent.orchestrator.two_stage import TwoStageOrchestrator
from pa_agent.util.threading import CancelToken, OrchestratorEvent

from .conftest import VALID_STAGE1, VALID_STAGE2, make_reply


def _quote_provider(symbol: str, timeframe: str) -> LiveQuote:
    return LiveQuote(symbol, timeframe, 2043.0, int(time.time() * 1000))


def test_happy_path(frame, pending_writer, assembler, exp_reader):
    """Both stages return valid JSON → full record saved, counter stays at 0."""
    client = MagicMock()
    client.stream_chat.side_effect = [
        make_reply(VALID_STAGE1),
        make_reply(VALID_STAGE2),
    ]

    validator = schema_test_validator()
    orchestrator = TwoStageOrchestrator(
        client=client,
        assembler=assembler,
        router=route_strategy_files,
        validator=validator,
        pending_writer=pending_writer,
        exp_reader=exp_reader,
        quote_provider=_quote_provider,
    )

    events: list[OrchestratorEvent] = []
    cancel_token = CancelToken()

    record = orchestrator.submit(
        frame=frame,
        cancel_token=cancel_token,
        on_event=events.append,
    )

    # Event sequence
    assert events == [
        OrchestratorEvent.Stage1Started,
        OrchestratorEvent.Stage1Done,
        OrchestratorEvent.Stage2Started,
        OrchestratorEvent.Stage2Done,
        OrchestratorEvent.RecordSaved,
    ]

    # Record has both stages populated
    assert record.stage1_diagnosis is not None
    assert record.stage2_decision is not None

    decision = record.stage2_decision["decision"]
    assert decision["order_type"] == "突破单"
    assert decision["execution_review"]["status"] == "resolved"

    # save_full was called (not save_partial)
    pending_writer.save_full.assert_called_once_with(record)
    pending_writer.save_partial.assert_not_called()


def test_independent_mode_ignores_previous_record(
    frame,
    pending_writer,
    assembler,
    exp_reader,
):
    """Independent mode must force a full Stage 1 and omit Stage 2 history."""
    client = MagicMock()
    client.stream_chat.side_effect = [
        make_reply(VALID_STAGE1),
        make_reply(VALID_STAGE2),
    ]
    settings = Settings()
    settings.general.independent_analysis_mode = True

    orchestrator = TwoStageOrchestrator(
        client=client,
        assembler=assembler,
        router=route_strategy_files,
        validator=schema_test_validator(),
        pending_writer=pending_writer,
        exp_reader=exp_reader,
        settings=settings,
        quote_provider=_quote_provider,
    )

    previous_record = MagicMock()
    record = orchestrator.submit(
        frame=frame,
        cancel_token=CancelToken(),
        on_event=lambda e: None,
        previous_record=previous_record,
        incremental_new_bar_count=1,
    )

    assert record.stage1_diagnosis is not None
    assembler.build_stage1.assert_called_once()
    assert not assembler.build_incremental_stage1.called
    kwargs = assembler.build_stage2_continuation.call_args.kwargs
    assert kwargs["previous_record"] is None
    assert kwargs["ignore_previous_context"] is True


def test_execution_rejection_is_saved_as_explicit_full_record(
    frame,
    pending_writer,
    assembler,
    exp_reader,
):
    client = MagicMock()
    client.stream_chat.side_effect = [
        make_reply(VALID_STAGE1),
        make_reply(VALID_STAGE2),
    ]

    def crossed_quote(symbol: str, timeframe: str) -> LiveQuote:
        return LiveQuote(symbol, timeframe, 2050.0, int(time.time() * 1000))

    orchestrator = TwoStageOrchestrator(
        client=client,
        assembler=assembler,
        router=route_strategy_files,
        validator=schema_test_validator(),
        pending_writer=pending_writer,
        exp_reader=exp_reader,
        quote_provider=crossed_quote,
    )

    record = orchestrator.submit(
        frame=frame,
        cancel_token=CancelToken(),
        on_event=lambda event: None,
    )

    decision = record.stage2_decision["decision"]
    assert decision["order_type"] == "不下单"
    assert decision["execution_review"]["status"] == "rejected"
    assert decision["execution_review"]["reason_code"] == (
        "breakout_trigger_already_crossed"
    )
    assert decision["execution_review"]["proposed_structure"]["entry_price"] == 2047.0
    pending_writer.save_full.assert_called_once_with(record)
    pending_writer.save_partial.assert_not_called()
