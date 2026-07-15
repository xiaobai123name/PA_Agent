"""AI decision generation with isolated settings and exact-input caching."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from pa_agent.ai.json_validator import JsonValidator
from pa_agent.ai.smc_features import SMC_FEATURE_VERSION
from pa_agent.ai.volume_price_features import VOLUME_FEATURE_VERSION
from pa_agent.backtest.lifecycle import (
    BacktestDecisionError,
    render_lifecycle_prompt,
    validate_lifecycle_decision,
)
from pa_agent.backtest.models import SimulationClock
from pa_agent.backtest.storage import (
    DecisionCache,
    MemoryPendingWriter,
    build_cache_key,
    prompt_source_hash,
)
from pa_agent.data.base import KlineFrame
from pa_agent.data.live_quote import LiveQuote
from pa_agent.orchestrator.two_stage import TwoStageOrchestrator
from pa_agent.records.schema import AnalysisRecord
from pa_agent.util.threading import OrchestratorEvent


class BacktestAIError(RuntimeError):
    """The isolated historical AI decision failed."""

    def __init__(
        self,
        message: str,
        *,
        record: AnalysisRecord | None = None,
        cache_key: str = "",
        failure_type: str = "unknown",
        stage: str = "unknown",
        skippable: bool = False,
    ) -> None:
        super().__init__(message)
        self.record = record
        self.cache_key = cache_key
        self.failure_type = str(failure_type)
        self.stage = str(stage)
        self.skippable = bool(skippable)


class AIDecisionRunner:
    """Run the existing two-stage pipeline without reading or writing live state."""

    def __init__(
        self,
        app_context: Any,
        *,
        dataset_hash: str,
        cache: DecisionCache | None = None,
        on_orchestrator_event: Callable[[Any], None] | None = None,
    ) -> None:
        self._ctx = app_context
        self._dataset_hash = dataset_hash
        self._cache = cache or DecisionCache()
        self._prompt_hash = prompt_source_hash()
        self._quote: LiveQuote | None = None
        self._clock = SimulationClock()
        self._on_orchestrator_event = on_orchestrator_event or (lambda _event: None)
        self._api_calls = 0

        settings = app_context.settings.model_copy(deep=True)
        settings.general.independent_analysis_mode = True
        settings.prompt.experience_max_entries = 0
        settings.validation.normalization_mode = "strict"
        settings.validation.disable_truncation_repair = True
        settings.validation.retry_enabled = True
        settings.validation.retry_mode = "format_only"
        settings.validation.retry_max = 1
        settings.validation.retry_stage2 = True
        self._settings = settings
        self._writer = MemoryPendingWriter()
        self._orchestrator = TwoStageOrchestrator(
            client=app_context.client,
            assembler=app_context.assembler,
            router=app_context.router,
            validator=JsonValidator(settings),
            pending_writer=self._writer,
            exp_reader=app_context.exp_reader,
            settings=settings,
            quote_provider=lambda _symbol, _timeframe: self._quote,
            execution_now_ms_provider=lambda _frame: self._clock.now_ms(),
        )

    @property
    def settings_snapshot(self) -> dict[str, Any]:
        return {
            "provider": self._settings.provider.model_dump(
                exclude={"api_key", "api_key_encrypted"}
            ),
            "general": self._settings.general.model_dump(),
            "prompt": self._settings.prompt.model_dump(),
            "validation": self._settings.validation.model_dump(),
        }

    @property
    def audit_snapshot(self) -> dict[str, Any]:
        return {
            "prompt_source_hash": self._prompt_hash,
            "settings": self.settings_snapshot,
        }

    @property
    def api_calls(self) -> int:
        return self._api_calls

    def _handle_orchestrator_event(self, event: Any) -> None:
        if event in {
            OrchestratorEvent.Stage1Started,
            OrchestratorEvent.Stage1Retry,
            OrchestratorEvent.Stage2Started,
            OrchestratorEvent.Stage2Retry,
        }:
            self._api_calls += 1
        self._on_orchestrator_event(event)

    def decide(
        self,
        frame: KlineFrame,
        *,
        quote: LiveQuote,
        pending: dict[str, Any] | None,
        cancel_token: Any,
        reuse_cache: bool,
    ) -> tuple[AnalysisRecord, str, bool]:
        lifecycle_prompt = render_lifecycle_prompt(pending)
        provider_snapshot = self._settings.provider.model_dump(
            exclude={"api_key", "api_key_encrypted"}
        )
        cache_key = build_cache_key(
            {
                "dataset_hash": self._dataset_hash,
                "prompt_source_hash": self._prompt_hash,
                "feature_versions": {
                    "smc": SMC_FEATURE_VERSION,
                    "volume_price": VOLUME_FEATURE_VERSION,
                },
                "provider": provider_snapshot,
                "general": {
                    "analysis_mode": getattr(
                        self._settings.general, "analysis_mode", "original"
                    ),
                    "decision_stance": self._settings.general.decision_stance,
                    "structure_flip_cooldown_bars": (
                        self._settings.general.structure_flip_cooldown_bars
                    ),
                    "execution_quote_max_age_ms": (
                        self._settings.general.execution_quote_max_age_ms
                    ),
                    "execution_max_slippage_atr": (
                        self._settings.general.execution_max_slippage_atr
                    ),
                    "execution_max_slippage_ticks": (
                        self._settings.general.execution_max_slippage_ticks
                    ),
                },
                "validation": self._settings.validation.model_dump(),
                "frame": {
                    "symbol": frame.symbol,
                    "timeframe": frame.timeframe,
                    "snapshot_ts_local_ms": frame.snapshot_ts_local_ms,
                    "price_tick": frame.price_tick,
                    "volume_meta": dataclasses.asdict(frame.volume_meta),
                    "bars": [dataclasses.asdict(bar) for bar in frame.bars],
                    "indicators": dataclasses.asdict(frame.indicators),
                },
                "quote": dataclasses.asdict(quote),
                "pending": pending,
                "lifecycle_prompt": lifecycle_prompt,
            }
        )
        if reuse_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                try:
                    validate_lifecycle_decision(
                        cached.stage2_decision or {},
                        pending_order=pending,
                    )
                except BacktestDecisionError as exc:
                    self._cache.delete(cache_key)
                    failed = cached.model_copy(
                        update={
                            "exception": {
                                "type": "lifecycle_error",
                                "stage": "lifecycle",
                                "message": str(exc),
                            }
                        }
                    )
                    raise BacktestAIError(
                        f"历史决策失败 stage=lifecycle: {exc}",
                        record=failed,
                        cache_key=cache_key,
                        failure_type="lifecycle_error",
                        stage="lifecycle",
                        skippable=True,
                    ) from exc
                return cached, cache_key, True

        self._quote = quote
        self._clock.set(quote.received_at_ms)
        record = self._orchestrator.submit(
            frame,
            cancel_token,
            self._handle_orchestrator_event,
            previous_record=None,
            incremental_new_bar_count=None,
            stage2_extra_task_context=lifecycle_prompt,
            force_stage2_on_gate_wait=True,
        )
        checker = getattr(cancel_token, "is_set", None)
        if callable(checker) and checker():
            raise InterruptedError("回测已取消")
        if record.exception is not None:
            stage = record.exception.get("stage") or "unknown"
            failure_type = str(record.exception.get("type") or "unknown")
            message = record.exception.get("message") or record.exception.get("type")
            raise BacktestAIError(
                f"历史决策失败 stage={stage}: {message}",
                record=record,
                cache_key=cache_key,
                failure_type=failure_type,
                stage=str(stage),
                skippable=failure_type == "validation_error",
            )
        if record.stage2_decision is None:
            raise BacktestAIError(
                "历史决策没有生成 stage2_decision",
                record=record,
                cache_key=cache_key,
                failure_type="missing_stage2_decision",
                stage="stage2",
                skippable=True,
            )
        try:
            validate_lifecycle_decision(
                record.stage2_decision,
                pending_order=pending,
            )
        except BacktestDecisionError as exc:
            failed = record.model_copy(
                update={
                    "exception": {
                        "type": "lifecycle_error",
                        "stage": "lifecycle",
                        "message": str(exc),
                    }
                }
            )
            raise BacktestAIError(
                f"历史决策失败 stage=lifecycle: {exc}",
                record=failed,
                cache_key=cache_key,
                failure_type="lifecycle_error",
                stage="lifecycle",
                skippable=True,
            ) from exc
        self._cache.put(cache_key, record)
        return record, cache_key, False
