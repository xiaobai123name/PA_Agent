"""Deterministic 1-minute order and position simulation."""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict
from typing import Any, Callable

from pa_agent.backtest.models import (
    BacktestRunConfig,
    PendingOrder,
    Position,
    TradeResult,
)
from pa_agent.data.base import KlineBar


class AmbiguousExecutionError(RuntimeError):
    """OHLC data cannot determine the event ordering inside one 1m bar."""


class ExecutionSizingRejected(RuntimeError):
    """A resolved order cannot satisfy exchange quantity constraints."""


class ExecutionSimulator:
    """Single-pending-order, single-position simulation state machine."""

    def __init__(
        self,
        config: BacktestRunConfig,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.cash_equity = float(config.initial_equity)
        self.pending_order: PendingOrder | None = None
        self.position: Position | None = None
        self.trades: list[TradeResult] = []
        self.total_fees = 0.0
        self.expired_orders = 0
        self.execution_rejections = 0
        self._on_event = on_event
        self.equity_points: list[tuple[int, float]] = []

    def pending_context(self) -> dict[str, Any] | None:
        order = self.pending_order
        if order is None:
            return None
        return {
            "created_at_ms": order.created_at_ms,
            "active_from_ms": order.active_from_ms,
            "remaining_bars": order.remaining_bars,
            "direction": order.direction,
            "entry_intent": order.entry_intent,
            "order_type": order.order_type,
            "entry_price": order.entry_price,
            "stop_price": order.stop_price,
            "tp1_price": order.tp1_price,
            "tp2_price": order.tp2_price,
        }

    def cancel_pending(self, *, timestamp_ms: int, reason: str) -> None:
        if self.pending_order is None:
            return
        order = self.pending_order
        self.pending_order = None
        self._emit(
            "order_cancelled",
            {
                "timestamp_ms": timestamp_ms,
                "order_id": order.order_id,
                "reason": reason,
            },
        )

    def place_from_decision(
        self,
        decision: dict[str, Any],
        *,
        decision_id: str,
        created_at_ms: int,
        active_from_ms: int,
        valid_bars: int,
    ) -> PendingOrder:
        if self.pending_order is not None or self.position is not None:
            raise RuntimeError("创建挂单时交易状态不是空闲")
        order = PendingOrder(
            order_id=uuid.uuid4().hex,
            created_at_ms=int(created_at_ms),
            active_from_ms=int(active_from_ms),
            remaining_bars=int(valid_bars),
            direction=self._required_text(decision, "order_direction"),
            entry_intent=self._required_text(decision, "entry_intent"),
            order_type=self._required_text(decision, "order_type"),
            entry_price=self._required_price(decision, "entry_price"),
            stop_price=self._required_price(decision, "stop_loss_price"),
            tp1_price=self._required_price(decision, "take_profit_price"),
            tp2_price=self._required_price(decision, "take_profit_price_2"),
            source_decision_id=decision_id,
        )
        self.pending_order = order
        self._emit("order_placed", asdict(order))
        return order

    def process_analysis_interval(self, bars: list[KlineBar], *, close_ms: int) -> None:
        for bar in bars:
            if self.position is not None:
                self._process_position_bar(bar)
            elif self.pending_order is not None:
                self._process_pending_bar(bar)
            self._mark_equity(int(bar.ts_open) + 60_000, float(bar.close))
        if self.pending_order is not None:
            self.pending_order.remaining_bars -= 1
            if self.pending_order.remaining_bars <= 0:
                expired = self.pending_order
                self.pending_order = None
                self.expired_orders += 1
                self._emit(
                    "order_expired",
                    {
                        "timestamp_ms": close_ms,
                        "order_id": expired.order_id,
                    },
                )

    def reject_execution(self, *, timestamp_ms: int, reason: str, code: str) -> None:
        self.execution_rejections += 1
        self._emit(
            "execution_rejected",
            {"timestamp_ms": timestamp_ms, "reason": reason, "code": code},
        )

    def mark_to_market(self, price: float) -> float:
        if self.position is None:
            return self.cash_equity
        sign = 1.0 if self.position.direction == "做多" else -1.0
        unrealized = sign * (float(price) - self.position.entry_price) * self.position.quantity
        return self.cash_equity + unrealized

    def _process_pending_bar(self, bar: KlineBar) -> None:
        order = self.pending_order
        if order is None or int(bar.ts_open) < order.active_from_ms:
            return
        fill = self._entry_fill(order, bar)
        if fill is None:
            return
        fill_price, fill_at_open = fill
        self.pending_order = None
        try:
            position = self._open_position(order, fill_price, int(bar.ts_open))
        except ExecutionSizingRejected as exc:
            self.reject_execution(
                timestamp_ms=int(bar.ts_open),
                reason=str(exc),
                code="sizing_rejected",
            )
            return
        self.position = position
        self._emit("position_opened", asdict(position))

        gap_exit = self._gap_exit(position, bar)
        if gap_exit is not None:
            price, reason = gap_exit
            self._close_position(price, int(bar.ts_open), reason)
            return

        stop_hit, tp_hit = self._exit_touches(position, bar)
        if not stop_hit and not tp_hit:
            return
        if not fill_at_open:
            raise AmbiguousExecutionError(
                f"1m K线 {int(bar.ts_open)} 内入场与退出触发顺序不确定"
            )
        if stop_hit and tp_hit:
            raise AmbiguousExecutionError(
                f"1m K线 {int(bar.ts_open)} 同时触及止损与TP1"
            )
        self._close_position_from_bar(bar, stop_hit=stop_hit, tp_hit=tp_hit)

    def _process_position_bar(self, bar: KlineBar) -> None:
        position = self.position
        if position is None:
            return
        gap_exit = self._gap_exit(position, bar)
        if gap_exit is not None:
            price, reason = gap_exit
            self._close_position(price, int(bar.ts_open), reason)
            return
        stop_hit, tp_hit = self._exit_touches(position, bar)
        if stop_hit and tp_hit:
            raise AmbiguousExecutionError(
                f"1m K线 {int(bar.ts_open)} 同时触及止损与TP1"
            )
        if stop_hit or tp_hit:
            self._close_position_from_bar(bar, stop_hit=stop_hit, tp_hit=tp_hit)

    def _entry_fill(
        self,
        order: PendingOrder,
        bar: KlineBar,
    ) -> tuple[float, bool] | None:
        slip = self.config.slippage_ticks * self.config.dataset.metadata.tick_size
        is_long = order.direction == "做多"
        if order.entry_intent == "immediate":
            price = float(bar.open) + (slip if is_long else -slip)
            return price, True
        if order.entry_intent == "pullback":
            touched = bar.low <= order.entry_price if is_long else bar.high >= order.entry_price
            if not touched:
                return None
            at_open = bar.open <= order.entry_price if is_long else bar.open >= order.entry_price
            if at_open:
                price = min(order.entry_price, bar.open) if is_long else max(order.entry_price, bar.open)
            else:
                price = order.entry_price
            return float(price), bool(at_open)
        if order.entry_intent == "breakout":
            touched = bar.high >= order.entry_price if is_long else bar.low <= order.entry_price
            if not touched:
                return None
            at_open = bar.open >= order.entry_price if is_long else bar.open <= order.entry_price
            base = (
                max(order.entry_price, bar.open)
                if is_long
                else min(order.entry_price, bar.open)
            )
            return float(base + (slip if is_long else -slip)), bool(at_open)
        raise RuntimeError(f"未知 entry_intent：{order.entry_intent}")

    def _open_position(
        self,
        order: PendingOrder,
        fill_price: float,
        opened_at_ms: int,
    ) -> Position:
        stop_distance = abs(fill_price - order.stop_price)
        if stop_distance <= 0:
            raise ExecutionSizingRejected("成交价与结构止损距离为零")
        risk_budget = self.cash_equity * self.config.risk_fraction
        risk_qty = risk_budget / stop_distance
        leverage_qty = self.cash_equity * self.config.max_leverage / fill_price
        raw_qty = min(risk_qty, leverage_qty)
        limited = leverage_qty < risk_qty
        step = self.config.dataset.metadata.step_size
        qty = math.floor(raw_qty / step) * step
        qty = round(qty, self._step_decimals(step))
        metadata = self.config.dataset.metadata
        if qty < metadata.min_qty:
            raise ExecutionSizingRejected(
                f"计算数量 {qty:g} 低于最小数量 {metadata.min_qty:g}"
            )
        if qty * fill_price < metadata.min_notional:
            raise ExecutionSizingRejected(
                f"计算名义价值 {qty * fill_price:g} 低于最小名义价值 {metadata.min_notional:g}"
            )
        fee_rate = (
            self.config.maker_fee_rate
            if order.entry_intent == "pullback"
            else self.config.taker_fee_rate
        )
        entry_fee = qty * fill_price * fee_rate
        self.cash_equity -= entry_fee
        self.total_fees += entry_fee
        return Position(
            position_id=uuid.uuid4().hex,
            opened_at_ms=opened_at_ms,
            direction=order.direction,
            entry_price=fill_price,
            stop_price=order.stop_price,
            tp1_price=order.tp1_price,
            tp2_price=order.tp2_price,
            quantity=qty,
            planned_risk=qty * stop_distance,
            entry_fee=entry_fee,
            source_order_id=order.order_id,
            size_limited_by_leverage=limited,
        )

    @staticmethod
    def _gap_exit(position: Position, bar: KlineBar) -> tuple[float, str] | None:
        if position.direction == "做多":
            if bar.open <= position.stop_price:
                return float(bar.open), "stop_gap"
            if bar.open >= position.tp1_price:
                return float(bar.open), "tp1_gap"
        else:
            if bar.open >= position.stop_price:
                return float(bar.open), "stop_gap"
            if bar.open <= position.tp1_price:
                return float(bar.open), "tp1_gap"
        return None

    @staticmethod
    def _exit_touches(position: Position, bar: KlineBar) -> tuple[bool, bool]:
        if position.direction == "做多":
            return bar.low <= position.stop_price, bar.high >= position.tp1_price
        return bar.high >= position.stop_price, bar.low <= position.tp1_price

    def _close_position_from_bar(
        self,
        bar: KlineBar,
        *,
        stop_hit: bool,
        tp_hit: bool,
    ) -> None:
        position = self.position
        if position is None:
            return
        if stop_hit:
            slip = self.config.slippage_ticks * self.config.dataset.metadata.tick_size
            price = position.stop_price + (slip if position.direction == "做空" else -slip)
            self._close_position(price, int(bar.ts_open), "stop")
        elif tp_hit:
            self._close_position(position.tp1_price, int(bar.ts_open), "tp1")

    def _close_position(self, price: float, closed_at_ms: int, reason: str) -> None:
        position = self.position
        if position is None:
            raise RuntimeError("没有持仓可关闭")
        sign = 1.0 if position.direction == "做多" else -1.0
        gross = sign * (price - position.entry_price) * position.quantity
        exit_fee_rate = (
            self.config.maker_fee_rate if reason.startswith("tp1") else self.config.taker_fee_rate
        )
        exit_fee = position.quantity * price * exit_fee_rate
        self.cash_equity += gross - exit_fee
        self.total_fees += exit_fee
        net = gross - position.entry_fee - exit_fee
        r_multiple = net / position.planned_risk if position.planned_risk > 0 else 0.0
        trade = TradeResult(
            trade_id=uuid.uuid4().hex,
            opened_at_ms=position.opened_at_ms,
            closed_at_ms=closed_at_ms,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=float(price),
            quantity=position.quantity,
            exit_reason=reason,
            gross_pnl=gross,
            fees=position.entry_fee + exit_fee,
            net_pnl=net,
            r_multiple=r_multiple,
            size_limited_by_leverage=position.size_limited_by_leverage,
        )
        self.position = None
        self.trades.append(trade)
        self._emit("position_closed", asdict(trade))

    def _mark_equity(self, timestamp_ms: int, price: float) -> None:
        self.equity_points.append((timestamp_ms, self.mark_to_market(price)))

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(kind, payload)

    @staticmethod
    def _required_price(decision: dict[str, Any], key: str) -> float:
        value = decision.get(key)
        if isinstance(value, bool):
            raise ValueError(f"decision.{key} 不是价格")
        try:
            price = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decision.{key} 缺少有效价格") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"decision.{key} 不是有效正数")
        return price

    @staticmethod
    def _required_text(decision: dict[str, Any], key: str) -> str:
        value = str(decision.get(key) or "").strip()
        if not value:
            raise ValueError(f"decision.{key} 缺失")
        return value

    @staticmethod
    def _step_decimals(step: float) -> int:
        text = f"{step:.16f}".rstrip("0")
        return len(text.split(".", 1)[1]) if "." in text else 0
