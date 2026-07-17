"""External higher-timeframe (HTF) context: fetch, summarize, inject.

One extra frame is fetched at a mapped higher timeframe before analysis and
summarized程序化 into ``htf_text``. Every failure path returns "" so the
analysis proceeds exactly as before when HTF data is unavailable.
"""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

#: 跳一级映射：上一级周期与当前窗口高度重叠（trend_context 的长程切窗已近似覆盖），
#: 高周期名额留给窗口完全看不到的波段相位。
DEFAULT_HTF_MAP: dict[str, str] = {
    "1m": "15m",
    "2m": "30m",
    "3m": "30m",
    "5m": "1h",
    "10m": "2h",
    "15m": "4h",
    "30m": "4h",
    "45m": "4h",
    "1h": "1d",
    "2h": "1d",
    "3h": "1d",
    "4h": "1d",
    "1d": "1w",
}

#: HTF 摘要用的已收盘 K 线目标数量（不足时逐级降到 30/20）。
HTF_BAR_COUNT = 60

_FALLBACK_BAR_COUNTS = (HTF_BAR_COUNT, 30, 20)


def resolve_htf_timeframe(
    timeframe: str,
    override_map: dict[str, str] | None = None,
    supported: list[str] | None = None,
) -> str | None:
    """Map the trading timeframe to its HTF; None when no valid mapping exists."""
    tf = (timeframe or "").strip()
    if not tf:
        return None
    target: str | None = None
    if override_map:
        target = (override_map.get(tf) or "").strip() or None
    if target is None:
        target = DEFAULT_HTF_MAP.get(tf)
    if not target or target == tf:
        return None
    if supported is not None:
        if target in supported:
            return target
        if target != "1d" and "1d" in supported and tf != "1d":
            return "1d"
        return None
    return target


def _fmt_span(ms: float) -> str:
    days = ms / 86_400_000.0
    if days >= 2:
        return f"{days:.0f} 天"
    hours = ms / 3_600_000.0
    return f"{hours:.0f} 小时"


def _finite(value: object) -> float | None:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def build_htf_text(frame: Any, base_timeframe: str) -> str:
    """Render a compact program-computed summary of the HTF frame."""
    bars = list(getattr(frame, "bars", ()) or ())
    if len(bars) < 10:
        return ""
    from pa_agent.ai.trend_context import (
        compute_background_direction,
        detect_recent_spike,
    )

    htf_tf = getattr(frame, "timeframe", "?")
    close = float(bars[0].close)
    hh_bar = max(bars, key=lambda b: float(b.high))
    ll_bar = min(bars, key=lambda b: float(b.low))
    hh, ll = float(hh_bar.high), float(ll_bar.low)
    pos_pct = 100.0 * (close - ll) / (hh - ll) if hh > ll else 50.0
    net10 = 100.0 * (close - float(bars[9].close)) / float(bars[9].close)
    span = _fmt_span(float(bars[0].ts_open) - float(bars[-1].ts_open))

    indicators = getattr(frame, "indicators", None)
    ema_now = _finite(indicators.ema20[0]) if indicators else None
    ema_prev = (
        _finite(indicators.ema20[9])
        if indicators and len(indicators.ema20) > 9
        else None
    )
    atr = _finite(indicators.atr14[0]) if indicators else None

    ema_line = "EMA20 数据不足"
    if ema_now is not None:
        side = "上" if close >= ema_now else "下"
        ema_line = f"最新收盘 {close:.6g} 位于 EMA20 {side}方"
        if atr:
            ema_line += f"，距离 ≈ {abs(close - ema_now) / atr:.1f}×ATR"
        if ema_prev is not None and atr:
            delta = ema_now - ema_prev
            if abs(delta) < 0.1 * atr:
                slope = "走平"
            else:
                slope = "向上" if delta > 0 else "向下"
            ema_line += f"；EMA20 斜率{slope}"

    bg = compute_background_direction(frame)
    spike = detect_recent_spike(frame)

    lines = [
        f"## 外部高周期背景（{htf_tf}，程序按真实高周期 K 线计算）",
        "",
        f"- 周期：{htf_tf}（当前分析周期 {base_timeframe} 的上级背景）；"
        f"样本 {len(bars)} 根已收盘 K 线，跨度约 {span}",
        f"- 长程方向投票：{bg}；近端动能（末 10 根）：净变动 {net10:+.1f}%",
        f"- {ema_line}",
        f"- 窗口极值：最高 {hh:.6g}（K{hh_bar.seq}），最低 {ll:.6g}（K{ll_bar.seq}）；"
        f"现价位于区间 {pos_pct:.0f}% 处",
        f"- 尖峰检测：{spike or '无'}",
        "",
        "使用规则：本块仅用于判定相位与基调（如「高周期上升趋势的回撤段」）。"
        "**不否决**当前周期 direction，冲突时近期为主、背景作风险提示；"
        "**禁止**用高周期价位直接生成 entry / stop / TP。",
    ]
    return "\n".join(lines)


def fetch_htf_text(
    source: Any,
    symbol: str,
    timeframe: str,
    general: Any,
    *,
    cancel_token: object | None = None,
) -> str:
    """Fetch + summarize the HTF frame; returns "" on any failure."""
    try:
        if source is None or not symbol or not timeframe:
            return ""
        if general is not None and not getattr(general, "htf_context_enabled", True):
            return ""
        override = dict(getattr(general, "htf_timeframe_map", {}) or {}) if general else {}
        try:
            supported: list[str] | None = list(source.supported_timeframes())
        except Exception:  # noqa: BLE001
            supported = None
        htf_tf = resolve_htf_timeframe(timeframe, override, supported)
        if not htf_tf:
            return ""

        from pa_agent.data.snapshot import INDICATOR_WARMUP_BARS, build_analysis_frame

        bars = source.fetch_frame_once(
            symbol,
            htf_tf,
            HTF_BAR_COUNT + INDICATOR_WARMUP_BARS + 5,
            cancel_token=cancel_token,
        )
        if not bars:
            return ""
        frame = None
        for want in _FALLBACK_BAR_COUNTS:
            frame = build_analysis_frame(
                bars, want, symbol, htf_tf, volume_meta=source.volume_meta
            )
            if frame is not None:
                break
        if frame is None:
            return ""
        return build_htf_text(frame, timeframe)
    except Exception as exc:  # noqa: BLE001
        logger.debug("HTF context unavailable for %s %s: %s", symbol, timeframe, exc)
        return ""
