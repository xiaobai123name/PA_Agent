from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pa_agent.backtest.decision_runner import BacktestAIError
from pa_agent.backtest.engine import BacktestEngine
from pa_agent.backtest.models import (
    BacktestRunConfig,
    BacktestRunStatus,
    ContractMetadata,
    FrozenDataset,
)
from pa_agent.backtest.storage import BacktestRunStore
from pa_agent.data.base import KlineBar
from pa_agent.data.snapshot import build_analysis_frame
from pa_agent.records.schema import AnalysisRecord, RecordMeta
from pa_agent.util.threading import CancelToken


def _bar(ts, o=100.0, h=101.0, l=99.0, c=100.0):
    return KlineBar(1, ts, o, h, l, c, 1.0, closed=True)


def _record(frame, decision):
    return AnalysisRecord(
        meta=RecordMeta(
            timestamp_local_iso="1970-01-01T00:00:00.000",
            timestamp_local_ms=frame.snapshot_ts_local_ms,
            symbol=frame.symbol,
            timeframe=frame.timeframe,
            bar_count=len(frame.bars),
            ai_provider={},
        ),
        kline_data=[],
        htf_text="",
        stage1_messages=[],
        stage1_response={},
        stage1_diagnosis={},
        stage2_messages=[],
        stage2_response={},
        stage2_decision={"decision": decision},
        strategy_files_used=[],
        experience_loaded=[],
        exception=None,
        usage_total={},
    )


def _failed_record(frame, *, failure_type="validation_error", stage="stage1"):
    return _record(frame, None).model_copy(
        update={
            "stage2_decision": None,
            "exception": {
                "type": failure_type,
                "stage": stage,
                "message": "invalid historical decision",
            },
        }
    )


def _none_decision():
    return {
        "order_action": "none",
        "order_valid_bars": None,
        "entry_intent": "none",
        "order_type": "不下单",
        "order_direction": None,
        "entry_price": None,
        "stop_loss_price": None,
        "take_profit_price": None,
        "take_profit_price_2": None,
        "execution_review": {"status": "not_applicable"},
    }


class _FakeRunner:
    def __init__(self):
        self.calls = 0
        self.price_ticks = []

    def decide(self, frame, **kwargs):
        self.calls += 1
        self.price_ticks.append(frame.price_tick)
        if self.calls == 1:
            decision = {
                "order_action": "place",
                "order_valid_bars": 3,
                "entry_intent": "immediate",
                "order_type": "市价单",
                "order_direction": "做多",
                "entry_price": 100.0,
                "stop_loss_price": 95.0,
                "take_profit_price": 110.0,
                "take_profit_price_2": 115.0,
                "execution_review": {"status": "resolved"},
            }
        else:
            decision = {
                "order_action": "none",
                "order_valid_bars": None,
                "entry_intent": "none",
                "order_type": "不下单",
                "order_direction": None,
                "entry_price": None,
                "stop_loss_price": None,
                "take_profit_price": None,
                "take_profit_price_2": None,
                "execution_review": {"status": "not_applicable"},
            }
        return _record(frame, decision), f"key-{self.calls}", False


class _FakeRepo:
    def __init__(self, analysis, execution):
        self.analysis = analysis
        self.execution = execution

    def load_bars(self, dataset, timeframe):
        return self.execution if timeframe == "1m" else self.analysis


class _SkipOnceRunner:
    api_calls = 0

    def __init__(self):
        self.calls = 0

    def decide(self, frame, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise BacktestAIError(
                "历史决策失败 stage=stage1: missing trapped_side",
                record=_failed_record(frame),
                cache_key="failed-key",
                failure_type="validation_error",
                stage="stage1",
                skippable=True,
            )
        return _record(frame, _none_decision()), f"key-{self.calls}", False


class _AlwaysSkipRunner:
    api_calls = 0

    def __init__(self):
        self.calls = 0

    def decide(self, frame, **_kwargs):
        self.calls += 1
        raise BacktestAIError(
            "历史决策失败 stage=stage2: invalid enum",
            record=_failed_record(frame, stage="stage2"),
            cache_key=f"failed-{self.calls}",
            failure_type="validation_error",
            stage="stage2",
            skippable=True,
        )


class _FatalRunner:
    api_calls = 0

    def __init__(self, *, failure_type: str | None):
        self.calls = 0
        self.failure_type = failure_type

    def decide(self, frame, **_kwargs):
        self.calls += 1
        if self.failure_type is None:
            raise RuntimeError("program defect")
        raise BacktestAIError(
            f"历史决策失败 stage=stage1: {self.failure_type}",
            record=_failed_record(frame, failure_type=self.failure_type),
            cache_key="fatal-key",
            failure_type=self.failure_type,
            stage="stage1",
            skippable=False,
        )


class _PendingThenSkipRunner:
    api_calls = 0

    def __init__(self):
        self.calls = 0

    def decide(self, frame, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            decision = {
                "order_action": "place",
                "order_valid_bars": 3,
                "entry_intent": "pullback",
                "order_type": "限价单",
                "order_direction": "做多",
                "entry_price": 90.0,
                "stop_loss_price": 80.0,
                "take_profit_price": 110.0,
                "take_profit_price_2": 115.0,
                "execution_review": {"status": "resolved"},
            }
            return _record(frame, decision), "place-key", False
        if self.calls == 2:
            raise BacktestAIError(
                "历史决策失败 stage=stage1: missing trapped_side",
                record=_failed_record(frame),
                cache_key="failed-key",
                failure_type="validation_error",
                stage="stage1",
                skippable=True,
            )
        decision = {
            "order_action": "keep",
            "order_valid_bars": None,
            "entry_intent": "pullback",
            "order_type": "限价单",
            "order_direction": "做多",
            "entry_price": 90.0,
            "stop_loss_price": 80.0,
            "take_profit_price": 110.0,
            "take_profit_price_2": 115.0,
            "execution_review": {"status": "resolved"},
        }
        return _record(frame, decision), "keep-key", False


class _InvalidLifecycleRunner:
    api_calls = 0

    def __init__(self):
        self.calls = 0

    def decide(self, frame, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            invalid = _none_decision()
            invalid["order_action"] = "place"
            return _record(frame, invalid), "invalid-lifecycle", False
        return _record(frame, _none_decision()), f"valid-{self.calls}", False


def _setup(tmp_path: Path, *, ai_limit=500):
    tf_ms = 15 * 60_000
    analysis = [_bar(-tf_ms), *[_bar(i * tf_ms) for i in range(4)]]
    execution = []
    for i in range(60):
        ts = i * 60_000
        if ts == 30 * 60_000:
            execution.append(_bar(ts, 100, 111, 99, 110))
        else:
            execution.append(_bar(ts, 100, 101, 99, 100))
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
        dataset_id="dataset",
        dataset_hash="hash",
        path=tmp_path / "unused.sqlite",
        symbol="BTCUSDT",
        analysis_timeframe="15m",
        requested_start_ms=0,
        requested_end_ms=4 * tf_ms,
        target_start_ms=0,
        target_end_ms=4 * tf_ms,
        analysis_bar_count=2,
        metadata=metadata,
    )
    return (
        _FakeRepo(analysis, execution),
        _FakeRunner(),
        BacktestRunConfig(dataset=dataset, analysis_bar_count=2, ai_call_limit=ai_limit),
    )


def test_engine_pauses_ai_while_position_is_open(tmp_path: Path):
    repo, runner, config = _setup(tmp_path)
    engine = BacktestEngine(
        repo,
        runner,
        run_store_factory=lambda cfg: BacktestRunStore(cfg, root=tmp_path),
    )
    summary = engine.run(config, CancelToken())

    assert summary.status == BacktestRunStatus.COMPLETED
    assert runner.calls == 2
    assert runner.price_ticks == [0.1, 0.1]
    assert summary.api_calls == 0
    assert len(summary.trades) == 1
    assert summary.trades[0].exit_reason == "tp1"


def test_engine_rejects_excess_ai_points_without_truncating(tmp_path: Path):
    repo, runner, config = _setup(tmp_path, ai_limit=1)
    engine = BacktestEngine(
        repo,
        runner,
        run_store_factory=lambda cfg: BacktestRunStore(cfg, root=tmp_path),
    )
    summary = engine.run(config, CancelToken())

    assert summary.status == BacktestRunStatus.FAILED
    assert "超过上限" in (summary.error or "")
    assert runner.calls == 0


def test_historical_frame_uses_simulation_timestamp():
    frame = build_analysis_frame(
        list(reversed([_bar(0), _bar(900_000)])),
        2,
        "BTCUSDT",
        "15m",
        now_ms=1_800_000,
    )
    assert frame is not None
    assert frame.snapshot_ts_local_ms == 1_800_000


def test_engine_cancellation_is_not_reported_as_failure(tmp_path: Path):
    repo, runner, config = _setup(tmp_path)
    engine = BacktestEngine(
        repo,
        runner,
        run_store_factory=lambda cfg: BacktestRunStore(cfg, root=tmp_path),
    )
    token = CancelToken()
    token.set()
    summary = engine.run(config, token)

    assert summary.status == BacktestRunStatus.CANCELLED
    assert runner.calls == 0


def test_engine_skips_decision_failure_and_completes_with_gaps(tmp_path: Path):
    repo, _runner, config = _setup(tmp_path)
    runner = _SkipOnceRunner()
    events = []
    engine = BacktestEngine(
        repo,
        runner,
        run_store_factory=lambda cfg: BacktestRunStore(cfg, root=tmp_path),
    )

    summary = engine.run(config, CancelToken(), events.append)

    assert summary.status == BacktestRunStatus.COMPLETED_WITH_ERRORS
    assert summary.decisions == 3
    assert summary.successful_decisions == 2
    assert summary.skipped_decisions == 1
    assert summary.decision_coverage_pct == pytest.approx(200 / 3)
    assert summary.decision_failure_counts == {"validation_error": 1}
    assert any(event.kind == "ai_decision_skipped" for event in events)
    manifest = json.loads((summary.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_errors"
    assert manifest["summary"]["skipped_decisions"] == 1
    with sqlite3.connect(summary.run_dir / "run.sqlite") as conn:
        payload = json.loads(
            conn.execute(
                "SELECT payload_json FROM events WHERE kind='ai_decision_skipped'"
            ).fetchone()[0]
        )
    assert payload["failure_type"] == "validation_error"
    assert payload["stage"] == "stage1"


def test_all_decisions_can_skip_without_becoming_successful(tmp_path: Path):
    repo, _runner, config = _setup(tmp_path)
    runner = _AlwaysSkipRunner()
    engine = BacktestEngine(
        repo,
        runner,
        run_store_factory=lambda cfg: BacktestRunStore(cfg, root=tmp_path),
    )

    summary = engine.run(config, CancelToken())

    assert summary.status == BacktestRunStatus.COMPLETED_WITH_ERRORS
    assert summary.decisions == 3
    assert summary.successful_decisions == 0
    assert summary.skipped_decisions == 3
    assert summary.decision_coverage_pct == 0.0


def test_pending_order_survives_skipped_decision_and_expires_normally(tmp_path: Path):
    repo, _runner, config = _setup(tmp_path)
    runner = _PendingThenSkipRunner()
    engine = BacktestEngine(
        repo,
        runner,
        run_store_factory=lambda cfg: BacktestRunStore(cfg, root=tmp_path),
    )

    summary = engine.run(config, CancelToken())

    assert summary.status == BacktestRunStatus.COMPLETED_WITH_ERRORS
    assert summary.skipped_decisions == 1
    assert summary.expired_orders == 1
    with sqlite3.connect(summary.run_dir / "run.sqlite") as conn:
        rows = conn.execute(
            "SELECT kind, payload_json FROM events "
            "WHERE kind IN ('order_placed', 'ai_decision_skipped', 'order_expired') "
            "ORDER BY id"
        ).fetchall()
    assert [row[0] for row in rows] == [
        "order_placed",
        "ai_decision_skipped",
        "order_expired",
    ]
    placed = json.loads(rows[0][1])
    skipped = json.loads(rows[1][1])
    assert skipped["pending_order_id"] == placed["order_id"]
    assert skipped["pending_remaining_bars"] == 2


def test_lifecycle_error_is_recorded_and_skipped(tmp_path: Path):
    repo, _runner, config = _setup(tmp_path)
    runner = _InvalidLifecycleRunner()
    engine = BacktestEngine(
        repo,
        runner,
        run_store_factory=lambda cfg: BacktestRunStore(cfg, root=tmp_path),
    )

    summary = engine.run(config, CancelToken())

    assert summary.status == BacktestRunStatus.COMPLETED_WITH_ERRORS
    assert summary.successful_decisions == 2
    assert summary.skipped_decisions == 1
    assert summary.decision_failure_counts == {"lifecycle_error": 1}
    with sqlite3.connect(summary.run_dir / "run.sqlite") as conn:
        record = json.loads(
            conn.execute(
                "SELECT record_json FROM decisions ORDER BY decision_time_ms LIMIT 1"
            ).fetchone()[0]
        )
    assert record["exception"]["type"] == "lifecycle_error"


@pytest.mark.parametrize("failure_type", ["network_error", "provider_error", None])
def test_infrastructure_and_program_errors_remain_fatal(
    tmp_path: Path,
    failure_type: str | None,
):
    repo, _runner, config = _setup(tmp_path)
    runner = _FatalRunner(failure_type=failure_type)
    engine = BacktestEngine(
        repo,
        runner,
        run_store_factory=lambda cfg: BacktestRunStore(cfg, root=tmp_path),
    )

    summary = engine.run(config, CancelToken())

    assert summary.status == BacktestRunStatus.FAILED
    assert summary.skipped_decisions == 0
    expected = "program defect" if failure_type is None else failure_type
    assert expected in (summary.error or "")
