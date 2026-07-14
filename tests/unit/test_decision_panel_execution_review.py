from __future__ import annotations

import sys

import pytest
from PyQt6.QtWidgets import QApplication

from pa_agent.gui.decision_panel import DecisionPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_decision_panel_shows_resolved_execution(qapp: QApplication) -> None:
    panel = DecisionPanel()
    panel.show()
    panel.set_decision(
        {
            "order_type": "突破单",
            "order_direction": "做多",
            "entry_price": 101.0,
            "stop_loss_price": 98.0,
            "take_profit_price": 105.0,
            "execution_review": {
                "status": "resolved",
                "reason_code": "execution_method_resolved",
                "reason": "突破触发位尚未越过",
            },
        }
    )
    qapp.processEvents()

    assert panel._execution_review_label.isVisible()
    assert "执行解析通过" in panel._execution_review_label.text()
    assert "execution_method_resolved" in panel._execution_review_label.text()


def test_decision_panel_shows_rejected_execution(qapp: QApplication) -> None:
    panel = DecisionPanel()
    panel.show()
    panel.set_decision(
        {
            "order_type": "不下单",
            "execution_review": {
                "status": "rejected",
                "reason_code": "immediate_entry_missed",
                "reason": "即时入场已错过",
            },
        }
    )
    qapp.processEvents()

    assert panel._execution_review_label.isVisible()
    assert "执行解析拒绝" in panel._execution_review_label.text()
    assert "即时入场已错过" in panel._execution_review_label.text()
