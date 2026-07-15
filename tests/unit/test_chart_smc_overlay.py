from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame, VolumeMeta


def _frame() -> KlineFrame:
    chronological = [
        KlineBar(
            seq=10 - index,
            ts_open=index * 60_000,
            open=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            volume=100.0,
            closed=True,
        )
        for index in range(10)
    ]
    bars = tuple(reversed(chronological))
    return KlineFrame(
        symbol="BTCUSDT",
        timeframe="15m",
        volume_meta=VolumeMeta("traded", "test", "base_asset"),
        bars=bars,
        indicators=IndicatorBundle(
            ema20=tuple(100.0 for _ in bars),
            atr14=tuple(2.0 for _ in bars),
        ),
        snapshot_ts_local_ms=600_000,
    )


def _features() -> dict:
    events = [
        {
            "id": f"bos:bullish:{ts}",
            "kind": "bos" if index % 2 == 0 else "choch",
            "direction": "bullish" if index % 2 == 0 else "bearish",
            "level": 100.0 + index,
            "ts_open": ts,
        }
        for index, ts in enumerate((540_000, 480_000, 420_000, 360_000))
    ]
    events.extend(
        {
            "id": f"sweep:bullish:{ts}",
            "kind": "liquidity_sweep",
            "direction": "bullish",
            "level": 99.0 - index,
            "ts_open": ts,
        }
        for index, ts in enumerate((300_000, 240_000, 180_000, 120_000))
    )
    fvgs = [
        {
            "id": f"fvg:bullish:{ts}",
            "direction": "bullish",
            "lower": 99.0 - index,
            "upper": 99.5 - index,
            "created_ts": ts,
            "status": "active",
        }
        for index, ts in enumerate((540_000, 480_000, 420_000, 360_000))
    ]
    fvgs.insert(
        0,
        {
            "id": "fvg:bullish:invalid",
            "direction": "bullish",
            "lower": 95.0,
            "upper": 95.5,
            "created_ts": 300_000,
            "status": "invalidated",
        },
    )
    order_blocks = [
        {
            "id": f"ob:bearish:{ts}",
            "direction": "bearish",
            "lower": 103.0 + index,
            "upper": 104.0 + index,
            "origin_ts": ts,
            "status": "mitigated",
        }
        for index, ts in enumerate((540_000, 480_000, 420_000))
    ]
    return {
        "status": "available",
        "events": events,
        "fvgs": fvgs,
        "order_blocks": order_blocks,
    }


@pytest.fixture
def chart(qtbot):
    from pa_agent.gui.chart_widget import ChartWidget

    widget = ChartWidget()
    qtbot.addWidget(widget)
    widget.set_frame_now(_frame())
    return widget


def test_smc_overlay_defaults_hidden_and_toggle_redraws(chart) -> None:
    chart.set_smc_features(_features())
    assert chart._smc_visible is False
    assert chart._smc_items == []

    chart.set_smc_visible(True)
    assert len(chart._smc_items) == 10

    chart.set_smc_visible(False)
    assert chart._smc_items == []
    chart.set_smc_visible(True)
    assert len(chart._smc_items) == 10


def test_smc_overlay_limits_zones_and_has_no_text_labels(chart) -> None:
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QGraphicsRectItem

    chart.set_smc_visible(True)
    chart.set_smc_features(_features())

    regions = [item for item in chart._smc_items if isinstance(item, QGraphicsRectItem)]
    labels = [item for item in chart._smc_items if isinstance(item, pg.TextItem)]
    assert len(regions) == 5
    assert labels == []
    assert all(region.rect().width() > 0 for region in regions)


def test_clear_and_reset_do_not_change_other_chart_overlays(chart) -> None:
    chart.set_decision(
        {
            "order_type": "限价单",
            "order_direction": "做多",
            "entry_price": 100.0,
            "take_profit_price": 110.0,
            "stop_loss_price": 95.0,
        }
    )
    trade_item_count = len(chart._overlay._items)
    chart.set_smc_visible(True)
    chart.set_smc_features(_features())
    chart.clear_smc_overlay()

    assert len(chart._overlay._items) == trade_item_count
    assert chart._smc_features is None
    assert chart._smc_items == []

    chart.reset()
    assert chart._smc_items == []
    assert chart._latest_frame is None


def test_smc_toggle_does_not_change_manual_zoom(chart) -> None:
    chart.getViewBox().setRange(xRange=(2.0, 7.0), yRange=(96.0, 106.0), padding=0)
    before = chart.getViewBox().viewRange()
    chart.set_smc_features(_features())
    chart.set_smc_visible(True)
    after = chart.getViewBox().viewRange()

    assert after[0] == pytest.approx(before[0])
    assert after[1] == pytest.approx(before[1])


def test_main_window_toggle_persists_setting(monkeypatch) -> None:
    from pa_agent.config.settings import Settings
    from pa_agent.gui.main_window import MainWindow

    settings = Settings()
    chart = MagicMock()
    holder = SimpleNamespace(
        _ctx=SimpleNamespace(settings=settings),
        _chart_widget=chart,
    )
    saved = MagicMock()
    monkeypatch.setattr("pa_agent.config.settings.save_settings", saved)

    MainWindow._on_smc_overlay_toggled(holder, True)

    assert settings.general.show_smc_overlay is True
    chart.set_smc_visible.assert_called_once_with(True)
    saved.assert_called_once_with(settings)
