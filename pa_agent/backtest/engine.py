"""Historical walk-forward engine connecting AI decisions to deterministic fills."""
from __future__ import annotations

import bisect
import math
import uuid
from dataclasses import asdict
from typing import Any, Callable

from pa_agent.backtest.lifecycle import BacktestDecisionError, validate_lifecycle_decision
from pa_agent.backtest.decision_runner import BacktestAIError
from pa_agent.backtest.models import (
    BacktestEvent,
    BacktestRunConfig,
    BacktestRunStatus,
    BacktestSummary,
    OrderAction,
)
from pa_agent.backtest.simulator import AmbiguousExecutionError, ExecutionSimulator
from pa_agent.backtest.storage import BacktestRunStore
from pa_agent.data.live_quote import LiveQuote
from pa_agent.data.snapshot import build_analysis_frame


class BacktestEngine:
    """Run one symbol/timeframe through historical AI analysis and execution."""

    def __init__(
        self,
        repository: Any,
        decision_runner: Any,
        *,
        run_store_factory: Callable[[BacktestRunConfig], BacktestRunStore] | None = None,
    ) -> None:
        self._repository = repository
        self._decision_runner = decision_runner
        self._run_store_factory = run_store_factory or BacktestRunStore

    def run(
        self,
        config: BacktestRunConfig,
        cancel_token: object,
        on_event: Callable[[BacktestEvent], None] | None = None,
    ) -> BacktestSummary:
        store = self._run_store_factory(config)
        status = BacktestRunStatus.PREPARING
        error: str | None = None
        decisions = 0
        successful_decisions = 0
        skipped_decisions = 0
        decision_failure_counts: dict[str, int] = {}
        api_calls = 0
        cache_hits = 0
        ambiguous_events = 0
        last_price = 0.0
        simulator = ExecutionSimulator(
            config,
            on_event=lambda kind, payload: self._persist_simulator_event(
                store, on_event, kind, payload
            ),
        )
        try:
            self._emit(on_event, "preparing", "加载冻结数据集")
            analysis = self._repository.load_bars(
                config.dataset, config.dataset.analysis_timeframe
            )
            execution = self._repository.load_bars(config.dataset, "1m")
            target = [
                bar
                for bar in analysis
                if config.dataset.target_start_ms
                <= int(bar.ts_open)
                < config.dataset.target_end_ms
            ]
            if not target:
                raise RuntimeError("冻结数据集没有目标分析 K线")
            execution_times = [int(bar.ts_open) for bar in execution]
            decision_points = [
                bar
                for bar in target
                if self._next_execution_bar(
                    execution,
                    execution_times,
                    self._analysis_close_ms(bar, config.dataset.analysis_timeframe),
                    config.dataset.requested_end_ms,
                )
                is not None
            ]
            if len(decision_points) > config.ai_call_limit:
                raise RuntimeError(
                    f"最坏情况需要 {len(decision_points)} 次 AI 决策，"
                    f"超过上限 {config.ai_call_limit}；禁止静默截断"
                )

            status = BacktestRunStatus.RUNNING
            manifest_update: dict[str, Any] = {
                "status": status.value,
                "estimated_decisions": len(decision_points),
            }
            audit_snapshot = getattr(self._decision_runner, "audit_snapshot", None)
            if isinstance(audit_snapshot, dict):
                manifest_update["ai"] = audit_snapshot
            store.update_manifest(**manifest_update)
            analysis_times = [int(bar.ts_open) for bar in analysis]
            for index, bar in enumerate(target, start=1):
                self._check_cancel(cancel_token)
                open_ms = int(bar.ts_open)
                close_ms = self._analysis_close_ms(bar, config.dataset.analysis_timeframe)
                interval_exec = self._bars_between(
                    execution,
                    execution_times,
                    open_ms,
                    close_ms,
                )
                if interval_exec:
                    last_price = float(interval_exec[-1].close)
                simulator.process_analysis_interval(interval_exec, close_ms=close_ms)

                self._emit(
                    on_event,
                    "progress",
                    f"推进至 {close_ms}",
                    index,
                    len(target),
                    {"simulation_time_ms": close_ms},
                )
                if simulator.position is not None:
                    self._emit(
                        on_event,
                        "analysis_skipped",
                        "持仓期间暂停 AI",
                        index,
                        len(target),
                    )
                    continue

                quote_bar = self._next_execution_bar(
                    execution,
                    execution_times,
                    close_ms,
                    config.dataset.requested_end_ms,
                )
                if quote_bar is None:
                    continue
                frame_index = bisect.bisect_right(analysis_times, open_ms)
                raw_desc = list(reversed(analysis[:frame_index]))
                frame = build_analysis_frame(
                    raw_desc,
                    config.analysis_bar_count,
                    config.dataset.symbol,
                    config.dataset.analysis_timeframe,
                    now_ms=close_ms,
                    price_tick=config.dataset.metadata.tick_size,
                )
                if frame is None:
                    raise RuntimeError(
                        f"{close_ms} 无法构造 {config.analysis_bar_count} 根闭合 K线窗口"
                    )
                quote = LiveQuote(
                    symbol=config.dataset.symbol,
                    timeframe=config.dataset.analysis_timeframe,
                    last_price=float(quote_bar.open),
                    received_at_ms=int(quote_bar.ts_open),
                    source="backtest_next_1m_open",
                )
                decisions += 1
                self._emit(
                    on_event,
                    "ai_started",
                    f"AI 决策 {decisions}/{len(decision_points)}",
                    decisions,
                    len(decision_points),
                )
                try:
                    record, cache_key, cache_hit = self._decision_runner.decide(
                        frame,
                        quote=quote,
                        pending=simulator.pending_context(),
                        cancel_token=cancel_token,
                        reuse_cache=config.reuse_decision_cache,
                    )
                except BacktestAIError as exc:
                    api_calls = int(
                        getattr(self._decision_runner, "api_calls", api_calls)
                    )
                    decision_id = uuid.uuid4().hex
                    if exc.record is not None:
                        store.add_decision(
                            decision_id,
                            close_ms,
                            exc.cache_key,
                            False,
                            exc.record,
                        )
                    if not exc.skippable:
                        raise
                    if exc.record is None:
                        raise RuntimeError(
                            "可跳过的 AI 决策错误缺少 AnalysisRecord 审计记录"
                        ) from exc
                    skipped_decisions += 1
                    decision_failure_counts[exc.failure_type] = (
                        decision_failure_counts.get(exc.failure_type, 0) + 1
                    )
                    self._record_skipped_decision(
                        store,
                        on_event,
                        simulator,
                        decision_id=decision_id,
                        timestamp_ms=close_ms,
                        failure_type=exc.failure_type,
                        stage=exc.stage,
                        message=str(exc),
                        current=decisions,
                        total=len(decision_points),
                    )
                    continue
                self._check_cancel(cancel_token)
                api_calls = int(getattr(self._decision_runner, "api_calls", api_calls))
                decision_id = uuid.uuid4().hex
                stage2 = record.stage2_decision
                if not isinstance(stage2, dict):
                    raise RuntimeError("AI 决策记录缺少 stage2_decision")
                try:
                    lifecycle = validate_lifecycle_decision(
                        stage2,
                        pending_order=simulator.pending_context(),
                    )
                except BacktestDecisionError as exc:
                    failed_record = record.model_copy(
                        update={
                            "exception": {
                                "type": "lifecycle_error",
                                "stage": "lifecycle",
                                "message": str(exc),
                            }
                        }
                    )
                    store.add_decision(
                        decision_id,
                        close_ms,
                        cache_key,
                        False,
                        failed_record,
                    )
                    skipped_decisions += 1
                    decision_failure_counts["lifecycle_error"] = (
                        decision_failure_counts.get("lifecycle_error", 0) + 1
                    )
                    self._record_skipped_decision(
                        store,
                        on_event,
                        simulator,
                        decision_id=decision_id,
                        timestamp_ms=close_ms,
                        failure_type="lifecycle_error",
                        stage="lifecycle",
                        message=str(exc),
                        current=decisions,
                        total=len(decision_points),
                    )
                    continue
                store.add_decision(
                    decision_id,
                    close_ms,
                    cache_key,
                    cache_hit,
                    record,
                )
                successful_decisions += 1
                cache_hits += int(cache_hit)
                if lifecycle.action == OrderAction.KEEP:
                    store.add_event(
                        "order_kept",
                        {
                            "decision_id": decision_id,
                            "timestamp_ms": close_ms,
                            "order_id": simulator.pending_order.order_id,
                        },
                    )
                    continue
                if lifecycle.action in (OrderAction.CANCEL, OrderAction.REPLACE):
                    simulator.cancel_pending(
                        timestamp_ms=close_ms,
                        reason=f"ai_{lifecycle.action.value}",
                    )
                if lifecycle.action in (OrderAction.CANCEL, OrderAction.NONE):
                    continue
                if lifecycle.execution_status == "rejected":
                    review = lifecycle.decision.get("execution_review") or {}
                    simulator.reject_execution(
                        timestamp_ms=close_ms,
                        reason=str(review.get("reason") or "执行解析拒绝"),
                        code=str(review.get("reason_code") or "execution_rejected"),
                    )
                    continue
                assert lifecycle.valid_bars is not None
                simulator.place_from_decision(
                    lifecycle.decision,
                    decision_id=decision_id,
                    created_at_ms=close_ms,
                    active_from_ms=int(quote_bar.ts_open),
                    valid_bars=lifecycle.valid_bars,
                )

            status = (
                BacktestRunStatus.COMPLETED_WITH_ERRORS
                if skipped_decisions
                else BacktestRunStatus.COMPLETED
            )
        except AmbiguousExecutionError as exc:
            ambiguous_events = 1
            status = BacktestRunStatus.INDETERMINATE
            error = str(exc)
            self._emit(on_event, "ambiguous", error)
            store.add_event("ambiguous", {"message": error})
        except InterruptedError as exc:
            status = BacktestRunStatus.CANCELLED
            error = str(exc)
            self._emit(on_event, "cancelled", error)
        except Exception as exc:
            status = BacktestRunStatus.FAILED
            error = str(exc)
            self._emit(on_event, "failed", error)
            store.add_event("failed", {"type": type(exc).__name__, "message": error})

        if last_price <= 0:
            execution = self._repository.load_bars(config.dataset, "1m")
            last_price = float(execution[-1].close) if execution else 0.0
        summary = self._build_summary(
            status,
            store,
            simulator,
            decisions=decisions,
            successful_decisions=successful_decisions,
            skipped_decisions=skipped_decisions,
            decision_failure_counts=decision_failure_counts,
            api_calls=int(getattr(self._decision_runner, "api_calls", api_calls)),
            cache_hits=cache_hits,
            ambiguous_events=ambiguous_events,
            last_price=last_price,
            error=error,
        )
        store.replace_equity(simulator.equity_points)
        store.replace_trades(simulator.trades)
        store.update_manifest(status=status.value, summary=asdict(summary), error=error)
        store.close()
        self._emit(
            on_event,
            "finished",
            f"回测状态：{status.value}",
            payload={"summary": summary},
        )
        return summary

    @staticmethod
    def _build_summary(
        status: BacktestRunStatus,
        store: BacktestRunStore,
        simulator: ExecutionSimulator,
        *,
        decisions: int,
        successful_decisions: int,
        skipped_decisions: int,
        decision_failure_counts: dict[str, int],
        api_calls: int,
        cache_hits: int,
        ambiguous_events: int,
        last_price: float,
        error: str | None,
    ) -> BacktestSummary:
        final_equity = simulator.mark_to_market(last_price) if last_price > 0 else simulator.cash_equity
        unrealized = final_equity - simulator.cash_equity
        realized = simulator.cash_equity - simulator.config.initial_equity
        trades = tuple(simulator.trades)
        positive = sum(trade.net_pnl for trade in trades if trade.net_pnl > 0)
        negative = abs(sum(trade.net_pnl for trade in trades if trade.net_pnl < 0))
        profit_factor = positive / negative if negative > 0 else (None if positive == 0 else math.inf)
        win_rate = (
            sum(1 for trade in trades if trade.net_pnl > 0) / len(trades) * 100
            if trades
            else None
        )
        expectancy = (
            sum(trade.r_multiple for trade in trades) / len(trades) if trades else None
        )
        values = [simulator.config.initial_equity] + [value for _, value in simulator.equity_points]
        peak = values[0]
        max_dd = 0.0
        for value in values:
            peak = max(peak, value)
            if peak > 0:
                max_dd = max(max_dd, (peak - value) / peak * 100)
        return BacktestSummary(
            status=status,
            run_id=store.run_id,
            run_dir=store.run_dir,
            initial_equity=simulator.config.initial_equity,
            final_equity=final_equity,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            total_r=sum(trade.r_multiple for trade in trades),
            net_return_pct=(final_equity / simulator.config.initial_equity - 1) * 100,
            max_drawdown_pct=max_dd,
            profit_factor=profit_factor,
            win_rate_pct=win_rate,
            expectancy_r=expectancy,
            fees=simulator.total_fees,
            trades=trades,
            decisions=decisions,
            successful_decisions=successful_decisions,
            skipped_decisions=skipped_decisions,
            decision_coverage_pct=(
                successful_decisions / decisions * 100 if decisions else 100.0
            ),
            decision_failure_counts=dict(decision_failure_counts),
            api_calls=api_calls,
            cache_hits=cache_hits,
            execution_rejections=simulator.execution_rejections,
            expired_orders=simulator.expired_orders,
            ambiguous_events=ambiguous_events,
            open_position=simulator.position is not None,
            error=error,
        )

    @staticmethod
    def _analysis_close_ms(bar: Any, timeframe: str) -> int:
        from pa_agent.backtest.historical_data import timeframe_ms

        return int(bar.ts_open) + timeframe_ms(timeframe)

    @staticmethod
    def _bars_between(
        bars: list[Any],
        times: list[int],
        start_ms: int,
        end_ms: int,
    ) -> list[Any]:
        left = bisect.bisect_left(times, int(start_ms))
        right = bisect.bisect_left(times, int(end_ms))
        return bars[left:right]

    @staticmethod
    def _next_execution_bar(
        bars: list[Any],
        times: list[int],
        timestamp_ms: int,
        end_ms: int,
    ) -> Any | None:
        index = bisect.bisect_left(times, int(timestamp_ms))
        if index >= len(bars) or int(bars[index].ts_open) >= int(end_ms):
            return None
        return bars[index]

    @staticmethod
    def _check_cancel(cancel_token: object) -> None:
        checker = getattr(cancel_token, "is_set", None)
        if callable(checker) and checker():
            raise InterruptedError("回测已取消")

    @staticmethod
    def _record_skipped_decision(
        store: BacktestRunStore,
        on_event: Callable[[BacktestEvent], None] | None,
        simulator: ExecutionSimulator,
        *,
        decision_id: str,
        timestamp_ms: int,
        failure_type: str,
        stage: str,
        message: str,
        current: int,
        total: int,
    ) -> None:
        pending = simulator.pending_order
        payload = {
            "decision_id": decision_id,
            "timestamp_ms": int(timestamp_ms),
            "failure_type": failure_type,
            "stage": stage,
            "message": message,
            "pending_order_id": pending.order_id if pending is not None else None,
            "pending_remaining_bars": (
                pending.remaining_bars if pending is not None else None
            ),
        }
        store.add_event("ai_decision_skipped", payload)
        BacktestEngine._emit(
            on_event,
            "ai_decision_skipped",
            f"AI 决策失败，已跳过本轮：{message}",
            current,
            total,
            payload,
        )

    @staticmethod
    def _persist_simulator_event(
        store: BacktestRunStore,
        on_event: Callable[[BacktestEvent], None] | None,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        store.add_event(kind, payload)
        BacktestEngine._emit(on_event, kind, kind, payload=payload)

    @staticmethod
    def _emit(
        on_event: Callable[[BacktestEvent], None] | None,
        kind: str,
        message: str,
        current: int = 0,
        total: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if on_event is not None:
            on_event(
                BacktestEvent(
                    kind=kind,
                    message=message,
                    progress_current=current,
                    progress_total=total,
                    payload=payload or {},
                )
            )
