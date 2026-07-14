from __future__ import annotations

import sqlite3

import pytest

from pa_agent.backtest.decision_runner import AIDecisionRunner, BacktestAIError
from pa_agent.backtest.models import SimulationClock
from pa_agent.backtest.storage import DecisionCache, build_cache_key
from pa_agent.config.settings import Settings
from pa_agent.data.base import KlineBar
from pa_agent.data.live_quote import LiveQuote
from pa_agent.data.snapshot import build_analysis_frame
from pa_agent.records.schema import AnalysisRecord, RecordMeta
from pa_agent.util.threading import CancelToken


def _record():
    return AnalysisRecord(
        meta=RecordMeta(
            timestamp_local_iso="2026-01-01T00:00:00.000",
            timestamp_local_ms=1,
            symbol="BTCUSDT",
            timeframe="15m",
            bar_count=20,
            ai_provider={},
        ),
        kline_data=[],
        htf_text="",
        stage1_messages=[],
        stage1_response={},
        stage1_diagnosis={},
        stage2_messages=[],
        stage2_response={},
        stage2_decision={"decision": {"order_action": "none"}},
        strategy_files_used=[],
        experience_loaded=[],
        exception=None,
        usage_total={},
    )


def test_decision_cache_only_matches_exact_key(tmp_path):
    cache = DecisionCache(tmp_path / "decisions.sqlite")
    key = build_cache_key({"dataset": "a", "prompt": "p", "frame": [1, 2]})
    other = build_cache_key({"dataset": "a", "prompt": "p", "frame": [1, 3]})
    cache.put(key, _record())

    assert cache.get(key) is not None
    assert cache.get(other) is None


def test_analysis_record_without_validation_attempts_remains_compatible():
    payload = _record().model_dump()
    payload.pop("validation_attempts")

    restored = AnalysisRecord.model_validate(payload)

    assert restored.validation_attempts == []


def test_ai_audit_snapshot_excludes_unrelated_credentials():
    settings = Settings()
    settings.provider.api_key = "provider-secret"
    settings.provider.api_key_encrypted = "encrypted-secret"
    settings.feishu.webhook_url = "https://secret-webhook"
    settings.feishu.app_secret = "feishu-secret"
    settings.pushplus.token = "push-secret"
    settings.tushare.token = "tushare-secret"
    runner = object.__new__(AIDecisionRunner)
    runner._settings = settings

    snapshot = runner.settings_snapshot
    text = str(snapshot)
    assert "provider-secret" not in text
    assert "encrypted-secret" not in text
    assert "secret-webhook" not in text
    assert "feishu-secret" not in text
    assert "push-secret" not in text
    assert "tushare-secret" not in text


def test_backtest_runner_uses_single_format_only_retry():
    settings = Settings()
    runner = object.__new__(AIDecisionRunner)
    isolated = settings.model_copy(deep=True)
    isolated.validation.normalization_mode = "strict"
    isolated.validation.disable_truncation_repair = True
    isolated.validation.retry_enabled = True
    isolated.validation.retry_mode = "format_only"
    isolated.validation.retry_max = 1
    isolated.validation.retry_stage2 = True
    runner._settings = isolated

    snapshot = runner.settings_snapshot["validation"]
    assert snapshot["normalization_mode"] == "strict"
    assert snapshot["disable_truncation_repair"] is True
    assert snapshot["retry_mode"] == "format_only"
    assert snapshot["retry_max"] == 1
    assert snapshot["retry_stage2"] is True


class _RecordOrchestrator:
    def __init__(self, record: AnalysisRecord):
        self.record = record

    def submit(self, *_args, **_kwargs):
        return self.record


def _manual_runner(tmp_path, record: AnalysisRecord) -> AIDecisionRunner:
    runner = object.__new__(AIDecisionRunner)
    runner._dataset_hash = "dataset"
    runner._cache = DecisionCache(tmp_path / "cache.sqlite")
    runner._prompt_hash = "prompt"
    runner._quote = None
    runner._clock = SimulationClock()
    runner._api_calls = 0
    runner._settings = Settings()
    runner._orchestrator = _RecordOrchestrator(record)
    return runner


def _frame_and_quote():
    bars = [
        KlineBar(1, 0, 100, 101, 99, 100, 1, closed=True),
        KlineBar(1, 60_000, 100, 101, 99, 100, 1, closed=True),
    ]
    frame = build_analysis_frame(
        list(reversed(bars)),
        2,
        "BTCUSDT",
        "1m",
        now_ms=120_000,
        price_tick=0.1,
    )
    assert frame is not None
    quote = LiveQuote(
        symbol="BTCUSDT",
        timeframe="1m",
        last_price=100,
        received_at_ms=120_000,
        source="test",
    )
    return frame, quote


def test_runner_classifies_lifecycle_error_as_skippable_and_does_not_cache(tmp_path):
    runner = _manual_runner(tmp_path, _record())
    frame, quote = _frame_and_quote()

    with pytest.raises(BacktestAIError) as raised:
        runner.decide(
            frame,
            quote=quote,
            pending=None,
            cancel_token=CancelToken(),
            reuse_cache=True,
        )

    assert raised.value.failure_type == "lifecycle_error"
    assert raised.value.stage == "lifecycle"
    assert raised.value.skippable is True
    with sqlite3.connect(runner._cache.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_runner_classifies_network_record_as_fatal(tmp_path):
    record = _record().model_copy(
        update={
            "stage2_decision": None,
            "exception": {
                "type": "network_error",
                "stage": "stage1",
                "message": "connection reset",
            },
        }
    )
    runner = _manual_runner(tmp_path, record)
    frame, quote = _frame_and_quote()

    with pytest.raises(BacktestAIError) as raised:
        runner.decide(
            frame,
            quote=quote,
            pending=None,
            cancel_token=CancelToken(),
            reuse_cache=False,
        )

    assert raised.value.failure_type == "network_error"
    assert raised.value.skippable is False


def test_runner_classifies_missing_stage2_as_skippable(tmp_path):
    runner = _manual_runner(
        tmp_path,
        _record().model_copy(update={"stage2_decision": None}),
    )
    frame, quote = _frame_and_quote()

    with pytest.raises(BacktestAIError) as raised:
        runner.decide(
            frame,
            quote=quote,
            pending=None,
            cancel_token=CancelToken(),
            reuse_cache=False,
        )

    assert raised.value.failure_type == "missing_stage2_decision"
    assert raised.value.stage == "stage2"
    assert raised.value.skippable is True
