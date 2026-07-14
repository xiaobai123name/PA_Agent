"""Independent Binance AI walk-forward backtest window."""
from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import QDateTime, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pa_agent.backtest.decision_runner import AIDecisionRunner
from pa_agent.backtest.engine import BacktestEngine
from pa_agent.backtest.historical_data import (
    BINANCE_BACKTEST_SYMBOLS,
    BINANCE_TIMEFRAMES,
    HistoricalDataRepository,
)
from pa_agent.backtest.models import (
    BacktestEvent,
    BacktestRunConfig,
    BacktestRunStatus,
    BacktestSummary,
)
from pa_agent.backtest.storage import record_preparation_failure
from pa_agent.gui.backtest_audit import (
    FILTER_LABELS,
    DecisionAuditEntry,
    ai_basis_text,
    build_audit_entry,
    execution_audit_text,
    format_local_time,
    matches_filter,
    summary_fields,
    validation_attempts_text,
)
from pa_agent.gui.theme import tokens as T
from pa_agent.util.threading import CancelToken


def _apply_dark_table_palette(widget: QTableWidget) -> None:
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor(T.BG_BASE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(T.BG_PANEL))
    palette.setColor(QPalette.ColorRole.Text, QColor(T.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Window, QColor(T.BG_BASE))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#264f78"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(T.TEXT_PRIMARY))
    widget.setPalette(palette)


class _BacktestWorker(QThread):
    event_ready = pyqtSignal(object)
    summary_ready = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(self, app_context: Any, request: dict[str, Any], parent: Any = None) -> None:
        super().__init__(parent)
        self._ctx = app_context
        self._request = request
        self.cancel_token = CancelToken()

    def run(self) -> None:
        try:
            repository = HistoricalDataRepository()
            dataset = repository.download_and_freeze(
                self._request["symbol"],
                self._request["timeframe"],
                self._request["start_ms"],
                self._request["end_ms"],
                analysis_bar_count=self._request["analysis_bar_count"],
                cancel_token=self.cancel_token,
                on_progress=lambda message, current, total: self.event_ready.emit(
                    BacktestEvent(
                        kind="download",
                        message=message,
                        progress_current=current,
                        progress_total=total,
                    )
                ),
            )
            config = BacktestRunConfig(
                dataset=dataset,
                analysis_bar_count=self._request["analysis_bar_count"],
                initial_equity=self._request["initial_equity"],
                risk_fraction=self._request["risk_pct"] / 100.0,
                max_leverage=self._request["max_leverage"],
                maker_fee_rate=self._request["maker_fee_pct"] / 100.0,
                taker_fee_rate=self._request["taker_fee_pct"] / 100.0,
                slippage_ticks=self._request["slippage_ticks"],
                ai_call_limit=self._request["ai_call_limit"],
                reuse_decision_cache=self._request["reuse_cache"],
            )
            def on_orchestrator_event(event: Any) -> None:
                name = str(getattr(event, "name", event))
                labels = {
                    "Stage1Started": "Stage 1 开始",
                    "Stage1Retry": "Stage 1 格式校验失败，定向重试 1/1",
                    "Stage1Done": "Stage 1 完成",
                    "Stage1Failed": "Stage 1 失败",
                    "Stage2Started": "Stage 2 开始",
                    "Stage2Retry": "Stage 2 格式校验失败，定向重试 1/1",
                    "Stage2Done": "Stage 2 完成",
                    "Stage2Failed": "Stage 2 失败",
                    "RecordSaved": "决策记录已保存",
                }
                self.event_ready.emit(
                    BacktestEvent(
                        kind="ai_stage",
                        message=f"{labels.get(name, name)}；API 调用 {runner.api_calls}",
                        payload={"api_calls": runner.api_calls},
                    )
                )

            runner = AIDecisionRunner(
                self._ctx,
                dataset_hash=dataset.dataset_hash,
                on_orchestrator_event=on_orchestrator_event,
            )
            engine = BacktestEngine(repository, runner)
            summary = engine.run(config, self.cancel_token, self.event_ready.emit)
            self.summary_ready.emit(summary)
        except InterruptedError as exc:
            try:
                record_preparation_failure(self._request, exc, status="cancelled")
            except Exception as persist_exc:  # noqa: BLE001
                self.fatal_error.emit(f"{exc}；取消现场写入失败：{persist_exc}")
            else:
                self.event_ready.emit(
                    BacktestEvent(kind="cancelled", message=str(exc))
                )
        except Exception as exc:  # noqa: BLE001
            try:
                record_preparation_failure(self._request, exc)
            except Exception as persist_exc:  # noqa: BLE001
                self.fatal_error.emit(f"{exc}；失败现场写入失败：{persist_exc}")
            else:
                self.fatal_error.emit(str(exc))


class BacktestWindow(QMainWindow):
    """Dense operator UI for configuring and auditing one historical run."""

    def __init__(self, app_context: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctx = app_context
        self._worker: _BacktestWorker | None = None
        self._close_pending = False
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("PA Agent 回测")
        self.resize(1500, 900)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        config_group = QGroupBox("回测配置")
        grid = QGridLayout(config_group)
        grid.setHorizontalSpacing(14)
        left = QFormLayout()
        right = QFormLayout()

        self._symbol = QComboBox()
        self._symbol.addItems(BINANCE_BACKTEST_SYMBOLS)
        self._timeframe = QComboBox()
        self._timeframe.addItems(BINANCE_TIMEFRAMES)
        self._timeframe.setCurrentText("15m")
        self._bar_count = QSpinBox()
        self._bar_count.setRange(20, 5000)
        self._bar_count.setValue(
            int(getattr(self._ctx.settings.general, "analysis_bar_count", 100))
        )
        self._model_label = QLabel(
            str(getattr(self._ctx.settings.provider, "model", "") or "未配置")
        )
        self._model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        now_ms = self._aligned_timestamp_ms(int(now.timestamp() * 1000), "15m")
        now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        start = now - timedelta(days=3)
        start_qt = QDateTime.fromMSecsSinceEpoch(int(start.timestamp() * 1000))
        end_qt = QDateTime.fromMSecsSinceEpoch(int(now.timestamp() * 1000))
        start_qt.setTimeSpec(Qt.TimeSpec.UTC)
        end_qt.setTimeSpec(Qt.TimeSpec.UTC)
        self._start = QDateTimeEdit(start_qt)
        self._end = QDateTimeEdit(end_qt)
        for widget in (self._start, self._end):
            widget.setDisplayFormat("yyyy-MM-dd HH:mm")
            widget.setCalendarPopup(True)
        self._timeframe.currentTextChanged.connect(self._align_date_inputs)

        left.addRow("品种", self._symbol)
        left.addRow("分析周期", self._timeframe)
        left.addRow("分析K线数", self._bar_count)
        left.addRow("AI模型", self._model_label)
        left.addRow("开始时间 UTC", self._start)
        left.addRow("结束时间 UTC", self._end)

        self._initial_equity = self._double_spin(100_000.0, 100.0, 1_000_000_000.0, 2)
        self._risk_pct = self._double_spin(1.0, 0.01, 100.0, 2)
        self._max_leverage = self._double_spin(5.0, 1.0, 125.0, 1)
        self._maker_fee = self._double_spin(0.0, 0.0, 10.0, 4)
        self._taker_fee = self._double_spin(0.0, 0.0, 10.0, 4)
        self._slippage_ticks = QSpinBox()
        self._slippage_ticks.setRange(0, 1000)
        self._ai_limit = QSpinBox()
        self._ai_limit.setRange(1, 100_000)
        self._ai_limit.setValue(500)
        self._reuse_cache = QCheckBox("复用完全相同输入的成功决策")
        self._reuse_cache.setChecked(True)

        right.addRow("初始资金 USDT", self._initial_equity)
        right.addRow("每笔风险 %", self._risk_pct)
        right.addRow("最大杠杆", self._max_leverage)
        right.addRow("Maker费率 %", self._maker_fee)
        right.addRow("Taker费率 %", self._taker_fee)
        right.addRow("滑点 ticks", self._slippage_ticks)
        right.addRow("AI决策上限", self._ai_limit)
        right.addRow("决策缓存", self._reuse_cache)
        grid.addLayout(left, 0, 0)
        grid.addLayout(right, 0, 1)
        layout.addWidget(config_group)

        action_row = QHBoxLayout()
        self._start_btn = QPushButton("开始回测")
        self._start_btn.setObjectName("primaryButton")
        self._start_btn.clicked.connect(self._start_run)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_run)
        self._status = QLabel("就绪")
        self._status.setWordWrap(True)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        action_row.addWidget(self._start_btn)
        action_row.addWidget(self._stop_btn)
        action_row.addWidget(self._status, 1)
        action_row.addWidget(self._progress, 2)
        layout.addLayout(action_row)

        self._zero_cost = QLabel("Maker/Taker/滑点当前均为 0，结果将明确按零交易成本计算")
        self._zero_cost.setStyleSheet("color: #d29922; font-weight: 600;")
        layout.addWidget(self._zero_cost)
        for widget in (self._maker_fee, self._taker_fee, self._slippage_ticks):
            widget.valueChanged.connect(self._sync_cost_label)

        splitter = QSplitter()
        result_panel = QWidget()
        result_panel.setMinimumWidth(500)
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        summary_grid = QGridLayout()
        self._summary_labels: dict[str, QLabel] = {}
        metrics = (
            ("状态", "status"),
            ("最终权益", "equity"),
            ("净收益率", "return"),
            ("累计R", "r"),
            ("最大回撤", "drawdown"),
            ("胜率", "win_rate"),
            ("Profit Factor", "pf"),
            ("成功/尝试/跳过/API/缓存", "decisions"),
        )
        for index, (title, key) in enumerate(metrics):
            value = QLabel("—")
            value.setStyleSheet("font-weight: 600;")
            summary_grid.addWidget(QLabel(title), index // 4 * 2, index % 4)
            summary_grid.addWidget(value, index // 4 * 2 + 1, index % 4)
            self._summary_labels[key] = value
        result_layout.addLayout(summary_grid)

        self._result_quality = QLabel()
        self._result_quality.setWordWrap(True)
        self._result_quality.setStyleSheet(
            f"color: {T.ACCENT_WARNING}; font-weight: 600;"
        )
        self._result_quality.hide()
        result_layout.addWidget(self._result_quality)

        self._equity_plot = pg.PlotWidget()
        self._equity_plot.setBackground("#0d1117")
        self._equity_plot.showGrid(x=True, y=True, alpha=0.2)
        self._equity_plot.setLabel("left", "Equity")
        self._equity_plot.setLabel("bottom", "Event")
        result_layout.addWidget(self._equity_plot, 2)

        self._trades = QTableWidget(0, 8)
        self._trades.setHorizontalHeaderLabels(
            ["开仓时间（本地）", "平仓时间（本地）", "方向", "入场", "退出", "原因", "净盈亏", "R"]
        )
        self._trades.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._trades.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._trades.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._trades.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._trades.setAlternatingRowColors(True)
        self._trades.verticalHeader().setVisible(False)
        _apply_dark_table_palette(self._trades)
        self._trades.cellDoubleClicked.connect(self._jump_trade_to_decision)
        self._trade_decision_ids: list[str | None] = []
        result_layout.addWidget(self._trades, 2)
        splitter.addWidget(result_panel)

        self._audit_tabs = QTabWidget()
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(3000)
        self._audit_tabs.addTab(self._log, "运行日志")

        self._decision_tab = QWidget()
        decision_layout = QVBoxLayout(self._decision_tab)
        decision_layout.setContentsMargins(0, 0, 0, 0)

        decision_splitter = QSplitter(Qt.Orientation.Horizontal)
        decision_master = QWidget()
        master_layout = QVBoxLayout(decision_master)
        master_layout.setContentsMargins(0, 0, 4, 0)
        master_layout.setSpacing(6)
        decision_master.setMinimumWidth(520)

        filter_row = QHBoxLayout()
        self._decision_filter = QComboBox()
        for label, key in FILTER_LABELS:
            self._decision_filter.addItem(label, key)
        self._decision_search = QLineEdit()
        self._decision_search.setPlaceholderText("搜索动作、方向、理由、关键因素…")
        self._decision_search.setClearButtonEnabled(True)
        filter_row.addWidget(self._decision_filter)
        filter_row.addWidget(self._decision_search, 1)
        master_layout.addLayout(filter_row)

        nav_row = QHBoxLayout()
        self._decision_count = QLabel("显示 0/0")
        self._decision_count.setObjectName("mutedLabel")
        self._prev_decision = QPushButton("上一条")
        self._prev_decision.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        )
        self._prev_decision.setToolTip("上一条关键决策")
        self._next_decision = QPushButton("下一条")
        self._next_decision.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward)
        )
        self._next_decision.setToolTip("下一条关键决策")
        nav_row.addWidget(self._decision_count, 1)
        nav_row.addWidget(self._prev_decision)
        nav_row.addWidget(self._next_decision)
        master_layout.addLayout(nav_row)

        self._decisions = QTableWidget(0, 6)
        self._decisions.setHorizontalHeaderLabels(
            ["本地时间", "决策动作", "方向/方式", "入场价", "执行状态", "置信度"]
        )
        self._decisions.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._decisions.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._decisions.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._decisions.setAlternatingRowColors(True)
        self._decisions.setShowGrid(False)
        self._decisions.setWordWrap(False)
        self._decisions.verticalHeader().setVisible(False)
        self._decisions.verticalHeader().setDefaultSectionSize(28)
        decision_header = self._decisions.horizontalHeader()
        decision_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        decision_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        decision_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        decision_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        decision_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        decision_header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        _apply_dark_table_palette(self._decisions)
        master_layout.addWidget(self._decisions, 1)
        decision_splitter.addWidget(decision_master)

        self._decision_detail_tabs = QTabWidget()
        self._decision_detail_tabs.setMinimumWidth(380)
        self._decision_summary = QTableWidget(0, 2)
        self._decision_summary.setHorizontalHeaderLabels(["项目", "内容"])
        self._decision_summary.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._decision_summary.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._decision_summary.setAlternatingRowColors(True)
        self._decision_summary.setShowGrid(False)
        self._decision_summary.verticalHeader().setVisible(False)
        self._decision_summary.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._decision_summary.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        _apply_dark_table_palette(self._decision_summary)
        self._decision_detail_tabs.addTab(self._decision_summary, "决策摘要")

        self._decision_basis = QPlainTextEdit()
        self._decision_basis.setReadOnly(True)
        self._decision_detail_tabs.addTab(self._decision_basis, "AI依据")

        self._execution_audit = QPlainTextEdit()
        self._execution_audit.setReadOnly(True)
        self._decision_detail_tabs.addTab(self._execution_audit, "执行审计")

        self._validation_attempts = QPlainTextEdit()
        self._validation_attempts.setReadOnly(True)
        self._decision_detail_tabs.addTab(self._validation_attempts, "校验重试")

        self._decision_raw = QPlainTextEdit()
        self._decision_raw.setReadOnly(True)
        self._decision_raw.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._decision_detail_tabs.addTab(self._decision_raw, "原始JSON")
        decision_splitter.addWidget(self._decision_detail_tabs)
        decision_splitter.setStretchFactor(0, 3)
        decision_splitter.setStretchFactor(1, 2)
        decision_splitter.setSizes([560, 420])
        decision_layout.addWidget(decision_splitter)

        self._decision_entries: list[DecisionAuditEntry] = []
        self._visible_decision_indices: list[int] = []
        self._decisions.currentCellChanged.connect(self._show_decision_details)
        self._decision_filter.currentIndexChanged.connect(self._apply_decision_filter)
        self._decision_search.textChanged.connect(self._apply_decision_filter)
        self._prev_decision.clicked.connect(lambda: self._move_decision(-1))
        self._next_decision.clicked.connect(lambda: self._move_decision(1))
        self._audit_tabs.addTab(self._decision_tab, "决策审计")
        splitter.addWidget(self._audit_tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _start_run(self) -> None:
        if self._worker is not None:
            return
        if not str(getattr(self._ctx.settings.provider, "api_key", "") or "").strip():
            QMessageBox.critical(self, "无法开始", "未配置 AI API Key")
            return
        start_ms = self._start.dateTime().toMSecsSinceEpoch()
        end_ms = self._end.dateTime().toMSecsSinceEpoch()
        if start_ms >= end_ms:
            QMessageBox.critical(self, "无法开始", "开始时间必须早于结束时间")
            return
        request = {
            "symbol": self._symbol.currentText(),
            "timeframe": self._timeframe.currentText(),
            "analysis_bar_count": self._bar_count.value(),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "initial_equity": self._initial_equity.value(),
            "risk_pct": self._risk_pct.value(),
            "max_leverage": self._max_leverage.value(),
            "maker_fee_pct": self._maker_fee.value(),
            "taker_fee_pct": self._taker_fee.value(),
            "slippage_ticks": self._slippage_ticks.value(),
            "ai_call_limit": self._ai_limit.value(),
            "reuse_cache": self._reuse_cache.isChecked(),
        }
        self._reset_results()
        self._set_running(True)
        self._worker = _BacktestWorker(self._ctx, request, self)
        self._worker.event_ready.connect(self._on_event)
        self._worker.summary_ready.connect(self._on_summary)
        self._worker.fatal_error.connect(self._on_fatal_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _stop_run(self) -> None:
        if self._worker is not None:
            self._worker.cancel_token.set()
            self._status.setText("正在停止…")
            self._stop_btn.setEnabled(False)

    def _on_event(self, event: BacktestEvent) -> None:
        self._status.setText(event.message)
        self._log.appendPlainText(f"[{event.kind}] {event.message}")
        if event.progress_total > 0:
            self._progress.setValue(
                int(event.progress_current / event.progress_total * 100)
            )

    def _on_summary(self, summary: BacktestSummary) -> None:
        has_gaps = summary.status == BacktestRunStatus.COMPLETED_WITH_ERRORS
        self._summary_labels["status"].setText(
            "完成但有缺口" if has_gaps else summary.status.value
        )
        self._summary_labels["status"].setStyleSheet(
            f"font-weight: 600; color: {T.ACCENT_WARNING};"
            if has_gaps
            else "font-weight: 600;"
        )
        self._summary_labels["equity"].setText(f"{summary.final_equity:,.2f}")
        self._summary_labels["return"].setText(f"{summary.net_return_pct:.2f}%")
        self._summary_labels["r"].setText(f"{summary.total_r:.2f}R")
        self._summary_labels["drawdown"].setText(f"{summary.max_drawdown_pct:.2f}%")
        self._summary_labels["win_rate"].setText(
            "—" if summary.win_rate_pct is None else f"{summary.win_rate_pct:.1f}%"
        )
        self._summary_labels["pf"].setText(
            "—" if summary.profit_factor is None else f"{summary.profit_factor:.2f}"
        )
        self._summary_labels["decisions"].setText(
            f"{summary.successful_decisions} / {summary.decisions} / "
            f"{summary.skipped_decisions} / {summary.api_calls} / {summary.cache_hits}"
        )
        if summary.skipped_decisions:
            failure_text = "，".join(
                f"{kind}={count}"
                for kind, count in sorted(summary.decision_failure_counts.items())
            )
            self._result_quality.setText(
                "本轮包含缺失决策，不等同于完整回测。"
                f"决策覆盖率 {summary.decision_coverage_pct:.1f}%，"
                f"跳过 {summary.skipped_decisions} 次"
                + (f"（{failure_text}）" if failure_text else "")
            )
            self._result_quality.show()
        else:
            self._result_quality.clear()
            self._result_quality.hide()
        self._render_trades(summary)
        self._render_equity(summary)
        self._render_decisions(summary)
        if summary.error:
            self._log.appendPlainText(f"[result] {summary.error}")

    def _on_fatal_error(self, message: str) -> None:
        self._status.setText("回测线程失败")
        self._log.appendPlainText(f"[fatal] {message}")
        QMessageBox.critical(self, "回测失败", message)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)
        if self._close_pending:
            self._close_pending = False
            self.close()

    def _render_trades(self, summary: BacktestSummary) -> None:
        self._trade_decision_ids = self._load_trade_decision_links(summary)
        self._trades.setRowCount(len(summary.trades))
        for row, trade in enumerate(summary.trades):
            values = (
                self._fmt_ts(trade.opened_at_ms),
                self._fmt_ts(trade.closed_at_ms),
                trade.direction,
                f"{trade.entry_price:g}",
                f"{trade.exit_price:g}",
                trade.exit_reason,
                f"{trade.net_pnl:,.2f}",
                f"{trade.r_multiple:.2f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                decision_id = self._trade_decision_ids[row]
                if decision_id is not None:
                    item.setData(Qt.ItemDataRole.UserRole, decision_id)
                    item.setToolTip("双击跳转到产生该交易的 AI 决策")
                self._trades.setItem(row, column, item)

    def _load_trade_decision_links(self, summary: BacktestSummary) -> list[str | None]:
        db_path = summary.run_dir / "run.sqlite"
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT kind, payload_json FROM events "
                "WHERE kind IN ('order_placed', 'position_opened') ORDER BY id"
            ).fetchall()
        order_to_decision: dict[str, str] = {}
        opened_to_decision: dict[int, str] = {}
        for kind, payload_json in rows:
            payload = json.loads(payload_json)
            if kind == "order_placed":
                order_to_decision[str(payload["order_id"])] = str(
                    payload["source_decision_id"]
                )
                continue
            order_id = str(payload["source_order_id"])
            opened_to_decision[int(payload["opened_at_ms"])] = order_to_decision[order_id]
        return [opened_to_decision.get(trade.opened_at_ms) for trade in summary.trades]

    def _jump_trade_to_decision(self, row: int, _column: int) -> None:
        item = self._trades.item(row, 0)
        decision_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not decision_id:
            QMessageBox.warning(self, "无法跳转", "该交易记录缺少来源决策映射")
            return
        key_index = self._decision_filter.findData("key")
        if key_index < 0:
            raise RuntimeError("决策筛选器缺少 key 选项")
        self._decision_filter.blockSignals(True)
        self._decision_search.blockSignals(True)
        self._decision_filter.setCurrentIndex(key_index)
        self._decision_search.clear()
        self._decision_filter.blockSignals(False)
        self._decision_search.blockSignals(False)
        self._apply_decision_filter(selected_decision_id=str(decision_id))
        self._audit_tabs.setCurrentWidget(self._decision_tab)

    def _render_equity(self, summary: BacktestSummary) -> None:
        self._equity_plot.clear()
        db_path = summary.run_dir / "run.sqlite"
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT timestamp_ms, equity FROM equity ORDER BY timestamp_ms"
                ).fetchall()
        except sqlite3.Error as exc:
            self._log.appendPlainText(f"[equity] {exc}")
            return
        if rows:
            self._equity_plot.plot(
                list(range(len(rows))),
                [float(row[1]) for row in rows],
                pen=pg.mkPen("#2f81f7", width=2),
            )

    def _render_decisions(self, summary: BacktestSummary) -> None:
        db_path = summary.run_dir / "run.sqlite"
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT decision_id, decision_time_ms, cache_hit, record_json "
                    "FROM decisions ORDER BY decision_time_ms"
                ).fetchall()
        except sqlite3.Error as exc:
            self._log.appendPlainText(f"[decisions] {exc}")
            return
        self._decision_entries = [
            build_audit_entry(
                decision_id=str(row[0]),
                decision_time_ms=int(row[1]),
                cache_hit=bool(row[2]),
                payload=json.loads(row[3]),
            )
            for row in rows
        ]
        self._apply_decision_filter()

    def _apply_decision_filter(
        self,
        *_signal_args: object,
        selected_decision_id: str | None = None,
    ) -> None:
        current_id = selected_decision_id or self._current_decision_id()
        filter_key = str(self._decision_filter.currentData())
        query = self._decision_search.text().strip().lower()
        self._visible_decision_indices = [
            index
            for index, entry in enumerate(self._decision_entries)
            if matches_filter(entry, filter_key)
            and (not query or query in entry.search_text)
        ]

        self._decisions.blockSignals(True)
        self._decisions.setRowCount(len(self._visible_decision_indices))
        for visible_row, source_index in enumerate(self._visible_decision_indices):
            entry = self._decision_entries[source_index]
            values = (
                format_local_time(entry.decision_time_ms),
                entry.action_label,
                entry.method_label,
                "—" if entry.entry_price is None else f"{entry.entry_price:g}",
                entry.status_label,
                "—" if entry.trade_confidence is None else f"{entry.trade_confidence}%",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, source_index)
                tooltip = value
                if entry.reason:
                    tooltip += f"\n\n{entry.reason}"
                item.setToolTip(tooltip)
                if column == 1:
                    item.setForeground(QColor(self._action_color(entry)))
                elif column == 4:
                    item.setForeground(QColor(self._status_color(entry)))
                self._decisions.setItem(visible_row, column, item)
        self._decisions.blockSignals(False)

        visible_set = set(self._visible_decision_indices)
        hidden_none = sum(
            entry.action == "none" and index not in visible_set
            for index, entry in enumerate(self._decision_entries)
        )
        count_text = f"显示 {len(self._visible_decision_indices)}/{len(self._decision_entries)}"
        if hidden_none:
            count_text += f"，已隐藏无操作 {hidden_none}"
        self._decision_count.setText(count_text)

        target_row = 0 if self._visible_decision_indices else -1
        if current_id:
            for row, source_index in enumerate(self._visible_decision_indices):
                if self._decision_entries[source_index].decision_id == current_id:
                    target_row = row
                    break
        if target_row >= 0:
            self._decisions.setCurrentCell(target_row, 0)
            self._show_decision_details(target_row, 0, -1, -1)
        else:
            self._clear_decision_details()
        self._sync_decision_navigation()

    def _show_decision_details(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if not 0 <= current_row < len(self._visible_decision_indices):
            self._clear_decision_details()
            return
        source_index = self._visible_decision_indices[current_row]
        entry = self._decision_entries[source_index]
        fields = summary_fields(entry)
        self._decision_summary.setRowCount(len(fields))
        for row, (label, value) in enumerate(fields.items()):
            label_item = QTableWidgetItem(label)
            value_item = QTableWidgetItem(value)
            label_item.setForeground(QColor(T.TEXT_SECONDARY))
            label_item.setToolTip(label)
            value_item.setToolTip(value)
            self._decision_summary.setItem(row, 0, label_item)
            self._decision_summary.setItem(row, 1, value_item)
        self._decision_summary.resizeRowsToContents()
        self._decision_basis.setPlainText(ai_basis_text(entry))
        self._execution_audit.setPlainText(execution_audit_text(entry))
        self._validation_attempts.setPlainText(validation_attempts_text(entry))
        self._decision_raw.setPlainText(
            json.dumps(
                entry.payload,
                ensure_ascii=False,
                indent=2,
            )
        )
        self._sync_decision_navigation()

    def _clear_decision_details(self) -> None:
        self._decision_summary.setRowCount(0)
        self._decision_basis.clear()
        self._execution_audit.clear()
        self._validation_attempts.clear()
        self._decision_raw.clear()

    def _current_decision_id(self) -> str | None:
        row = self._decisions.currentRow()
        if not 0 <= row < len(self._visible_decision_indices):
            return None
        return self._decision_entries[
            self._visible_decision_indices[row]
        ].decision_id

    def _move_decision(self, offset: int) -> None:
        target = self._adjacent_key_row(offset)
        if target is not None:
            self._decisions.setCurrentCell(target, 0)
            item = self._decisions.item(target, 0)
            if item is not None:
                self._decisions.scrollToItem(item)

    def _sync_decision_navigation(self) -> None:
        self._prev_decision.setEnabled(self._adjacent_key_row(-1) is not None)
        self._next_decision.setEnabled(self._adjacent_key_row(1) is not None)

    def _adjacent_key_row(self, offset: int) -> int | None:
        current = self._decisions.currentRow()
        if current < 0:
            return None
        stop = -1 if offset < 0 else len(self._visible_decision_indices)
        for row in range(current + offset, stop, offset):
            source_index = self._visible_decision_indices[row]
            if self._decision_entries[source_index].is_key:
                return row
        return None

    @staticmethod
    def _action_color(entry: DecisionAuditEntry) -> str:
        if entry.status in {"rejected", "failed"} or entry.action == "cancel":
            return T.ACCENT_DANGER
        if entry.action in {"place", "replace"}:
            return T.ACCENT_PRIMARY
        if entry.action == "keep":
            return T.ACCENT_WARNING
        return T.TEXT_SECONDARY

    @staticmethod
    def _status_color(entry: DecisionAuditEntry) -> str:
        if entry.status == "resolved":
            return T.ACCENT_SUCCESS
        if entry.status in {"rejected", "failed"}:
            return T.ACCENT_DANGER
        return T.TEXT_SECONDARY

    def _reset_results(self) -> None:
        for label in self._summary_labels.values():
            label.setText("—")
            label.setStyleSheet("font-weight: 600;")
        self._trades.setRowCount(0)
        self._decisions.setRowCount(0)
        self._trade_decision_ids = []
        self._decision_entries = []
        self._visible_decision_indices = []
        self._decision_count.setText("显示 0/0")
        self._clear_decision_details()
        self._result_quality.clear()
        self._result_quality.hide()
        self._equity_plot.clear()
        self._log.clear()
        self._progress.setValue(0)

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        for widget in (
            self._symbol,
            self._timeframe,
            self._bar_count,
            self._start,
            self._end,
            self._initial_equity,
            self._risk_pct,
            self._max_leverage,
            self._maker_fee,
            self._taker_fee,
            self._slippage_ticks,
            self._ai_limit,
            self._reuse_cache,
        ):
            widget.setEnabled(not running)

    def _sync_cost_label(self) -> None:
        is_zero = (
            self._maker_fee.value() == 0
            and self._taker_fee.value() == 0
            and self._slippage_ticks.value() == 0
        )
        self._zero_cost.setVisible(is_zero)

    @staticmethod
    def _double_spin(value: float, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        return widget

    @staticmethod
    def _fmt_ts(timestamp_ms: int) -> str:
        return format_local_time(timestamp_ms)

    def _align_date_inputs(self, timeframe: str) -> None:
        for widget in (self._start, self._end):
            aligned = self._aligned_timestamp_ms(
                widget.dateTime().toMSecsSinceEpoch(), timeframe
            )
            value = QDateTime.fromMSecsSinceEpoch(aligned)
            value.setTimeSpec(Qt.TimeSpec.UTC)
            widget.setDateTime(value)

    @staticmethod
    def _aligned_timestamp_ms(timestamp_ms: int, timeframe: str) -> int:
        interval = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }[timeframe]
        return int(timestamp_ms) // interval * interval

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._worker is not None:
            self._close_pending = True
            self._stop_run()
            event.ignore()
            return
        event.accept()
