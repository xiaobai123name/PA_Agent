"""Shared backtest models and serialization helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class BacktestRunStatus(StrEnum):
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INDETERMINATE = "indeterminate"


class OrderAction(StrEnum):
    PLACE = "place"
    KEEP = "keep"
    REPLACE = "replace"
    CANCEL = "cancel"
    NONE = "none"


@dataclass(slots=True)
class SimulationClock:
    current_ms: int = 0

    def set(self, timestamp_ms: int) -> None:
        self.current_ms = int(timestamp_ms)

    def now_ms(self) -> int:
        return self.current_ms


@dataclass(frozen=True, slots=True)
class ContractMetadata:
    symbol: str
    status: str
    contract_type: str
    onboard_date_ms: int
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float


@dataclass(frozen=True, slots=True)
class FrozenDataset:
    dataset_id: str
    dataset_hash: str
    path: Path
    symbol: str
    analysis_timeframe: str
    requested_start_ms: int
    requested_end_ms: int
    target_start_ms: int
    target_end_ms: int
    analysis_bar_count: int
    metadata: ContractMetadata


@dataclass(frozen=True, slots=True)
class BacktestRunConfig:
    dataset: FrozenDataset
    analysis_bar_count: int = 100
    initial_equity: float = 100_000.0
    risk_fraction: float = 0.01
    max_leverage: float = 5.0
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0
    slippage_ticks: int = 0
    ai_call_limit: int = 500
    reuse_decision_cache: bool = True

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset"]["path"] = str(self.dataset.path)
        return data


@dataclass(frozen=True, slots=True)
class BacktestEvent:
    kind: str
    message: str
    progress_current: int = 0
    progress_total: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PendingOrder:
    order_id: str
    created_at_ms: int
    active_from_ms: int
    remaining_bars: int
    direction: str
    entry_intent: str
    order_type: str
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    source_decision_id: str


@dataclass(slots=True)
class Position:
    position_id: str
    opened_at_ms: int
    direction: str
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    quantity: float
    planned_risk: float
    entry_fee: float
    source_order_id: str
    size_limited_by_leverage: bool = False


@dataclass(frozen=True, slots=True)
class TradeResult:
    trade_id: str
    opened_at_ms: int
    closed_at_ms: int
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    exit_reason: str
    gross_pnl: float
    fees: float
    net_pnl: float
    r_multiple: float
    size_limited_by_leverage: bool


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    status: BacktestRunStatus
    run_id: str
    run_dir: Path
    initial_equity: float
    final_equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_r: float
    net_return_pct: float
    max_drawdown_pct: float
    profit_factor: float | None
    win_rate_pct: float | None
    expectancy_r: float | None
    fees: float
    trades: tuple[TradeResult, ...]
    decisions: int
    successful_decisions: int
    skipped_decisions: int
    decision_coverage_pct: float
    decision_failure_counts: dict[str, int]
    api_calls: int
    cache_hits: int
    execution_rejections: int
    expired_orders: int
    ambiguous_events: int
    open_position: bool
    error: str | None = None
