"""FreeChatSession — post-analysis free-chat session.

Maintains a conversation history anchored to a completed two-stage
AnalysisRecord and sends follow-up messages to the DeepSeek API.

Design reference: design.md §B.17
"""
from __future__ import annotations

import logging
import json
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from pa_agent.ai.deepseek_client import DeepSeekClient
    from pa_agent.ai.prompt_assembler import PromptAssembler
    from pa_agent.ai.session_ledger import SessionTokenLedger
    from pa_agent.config.settings import Settings
    from pa_agent.records.pending_writer import PendingWriter

from pa_agent.ai.deepseek_client import AIReply
from pa_agent.data.datetime_ts import format_epoch_for_display
from pa_agent.records.schema import AnalysisRecord, FollowupTurn
from pa_agent.util.threading import CancelToken
from pa_agent.util.timefmt import now_local_ms

logger = logging.getLogger(__name__)


def _derive_record_id(record: AnalysisRecord) -> str:
    """Derive the record basename (without extension) from an AnalysisRecord.

    Uses the same logic as ``_build_basename`` in pending_writer.py.
    """
    from datetime import datetime, timezone

    ms = record.meta.timestamp_local_ms
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone()
    ts_str = dt.strftime("%Y-%m-%d_%H-%m-%S")
    symbol = record.meta.symbol
    timeframe = record.meta.timeframe
    return f"{ts_str}_{symbol}_{timeframe}"


def _strip_reasoning(message: dict) -> dict:
    """Return a copy of *message* without the ``reasoning_content`` key."""
    return {k: v for k, v in message.items() if k != "reasoning_content"}


def _build_order_timing_anchor(record: AnalysisRecord) -> str:
    """Build the immutable order-timing facts for follow-up prompts."""
    kline_data = getattr(record, "kline_data", None) or []
    if not kline_data:
        return ""

    original_k1 = next(
        (bar for bar in kline_data if bar.get("seq") == 1),
        kline_data[0],
    )
    ts_open = original_k1.get("ts_open")
    if ts_open is None:
        return ""

    try:
        k1_time = format_epoch_for_display(float(ts_open), short=False)
    except (TypeError, ValueError, OverflowError):
        k1_time = "未知"

    stage2 = getattr(record, "stage2_decision", None) or {}
    decision = stage2.get("decision") or {}
    entry_bar = (stage2.get("bar_analysis") or {}).get("entry_bar") or {}

    order_type = decision.get("order_type") or "未提供"
    order_direction = decision.get("order_direction") or "未提供"
    order_summary = f"{order_type} / {order_direction}"
    for label, field in (
        ("entry", "entry_price"),
        ("TP1", "take_profit_price"),
        ("TP2", "take_profit_price_2"),
        ("SL", "stop_loss_price"),
    ):
        value = decision.get(field)
        if value is not None:
            order_summary += f" / {label}={value}"

    status_parts: list[str] = []
    if entry_bar.get("freshness"):
        status_parts.append(f"freshness={entry_bar['freshness']}")
    if entry_bar.get("strength"):
        status_parts.append(f"strength={entry_bar['strength']}")
    original_status = ", ".join(status_parts) or "未提供"

    return (
        "## 订单时序硬锚点（程序生成，必须严格遵守）\n"
        f"- 原始分析记录：{_derive_record_id(record)}\n"
        f"- 交易对象：{record.meta.symbol} / {record.meta.timeframe}\n"
        "- 原始分析K1（固定身份，不随后续图表的K1编号变化）："
        f"seq={original_k1.get('seq', 1)}，ts_open={ts_open}，时间={k1_time}，"
        f"O={original_k1.get('open')}，H={original_k1.get('high')}，"
        f"L={original_k1.get('low')}，C={original_k1.get('close')}\n"
        "- 订单生成时点：上述原始分析K1收盘之后\n"
        f"- 原始订单计划：{order_summary}\n"
        f"- 原始入场状态：{original_status}\n"
        "- 强制时序规则：原始分析K1及 ts_open 小于或等于该锚点的K线，"
        "不得用于判定本订单已成交、已止盈或已止损。\n"
        f"- 有效证据规则：只能使用 ts_open > {ts_open} 的后续K线或真实成交回报；"
        "没有此类证据时，必须回答未确认成交/仍待确认，禁止推定已经盈利或亏损。"
    )


class FreeChatSession:
    """Manages a free-chat conversation anchored to a completed analysis.

    Parameters
    ----------
    base_record:
        The fully completed AnalysisRecord from the two-stage pipeline.
    client:
        DeepSeekClient instance for API calls.
    assembler:
        PromptAssembler kept for future use. Follow-up chat builds its own
        advisory prompt instead of reusing the Stage 2 decision contract.
    pending_writer:
        PendingWriter for appending FollowupTurn entries to the JSONL
        sidecar file.
    ledger:
        SessionTokenLedger for accumulating token usage and cost.
    settings:
        Optional Settings object; used for ``reasoning_effort`` forwarding.
    kline_snapshot_fn:
        Optional callable that returns the latest closed K-line data as a
        text table string.  Called on each ``send()`` so the AI always
        sees the most recent market data.
    """

    #: When True, ``reasoning_content`` is preserved in assistant messages
    #: sent back to the API (for future tool-call scenarios).
    keep_reasoning_in_resend: bool = False

    def __init__(
        self,
        base_record: AnalysisRecord,
        client: "DeepSeekClient",
        assembler: "PromptAssembler",
        pending_writer: "PendingWriter",
        ledger: "SessionTokenLedger",
        settings: Optional["Settings"] = None,
        kline_snapshot_fn: Optional[Callable[[], str]] = None,
    ) -> None:
        self._base_record = base_record
        self._client = client
        self._assembler = assembler
        self._pending_writer = pending_writer
        self._ledger = ledger
        self._settings = settings
        self._kline_snapshot_fn = kline_snapshot_fn

        # Turn counter — incremented before each send so the first turn is 1.
        self._turn: int = 0

        # Full history including reasoning_content (for UI display and
        # persistence).  Each entry is a plain dict with at least
        # ``role`` and ``content``; assistant entries also carry
        # ``reasoning_content``.
        self._history_full: list[dict] = []

        # Derived record ID used as the JSONL sidecar basename.
        self._record_id: str = _derive_record_id(base_record)

        # Immutable timing facts are repeated in every follow-up user prompt so
        # a refreshed chart cannot redefine which candle was the analysis K1.
        self._order_timing_anchor: str = _build_order_timing_anchor(base_record)

        # ── Pre-build stable prefix (cached for all turns in this session) ────
        # These three messages are byte-for-byte identical across every turn of
        # the same session, so they form a stable prefix that the API can cache.
        # Building them once at session start avoids repeated JSON serialisation.
        self._cached_prefix: list[dict] = self._build_prefix(base_record)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def history_full(self) -> list[dict]:
        """Read-only view of the full message history (includes reasoning)."""
        return list(self._history_full)

    @property
    def record_id(self) -> str:
        """The record basename used for the JSONL sidecar file."""
        return self._record_id

    @staticmethod
    def _build_prefix(base_record: AnalysisRecord) -> list[dict]:
        """Build the stable prefix messages for this session (built once, reused each turn).

        Structure:
          [0] system  — follow-up advisor instructions (fully static across sessions)
          [1] user    — compact analysis reference (static within this session;
                        meta timestamps removed so the block stays stable)
          [2] assistant — Stage 2 original AI reply (static within this session;
                        lets the model see its own prose, not just the parsed JSON)

        Keeping these byte-identical across all turns of the same session means
        the API prefix cache is warm from turn 2 onwards, cutting prompt token
        cost significantly for multi-turn follow-up conversations.
        """
        prefix: list[dict] = []

        # [0] System — completely static, shared across all sessions
        prefix.append(
            {
                "role": "system",
                "content": (
                    "你是 PA Agent 的【追问助手】（post-analysis advisor），不是在执行新的完整两阶段分析。\n"
                    "你的目标是：优先、直接回答用户当前问题；必要时引用价格行为/关键价位/风险控制。\n"
                    "\n"
                    "严格规则：\n"
                    "1) 默认用自然语言回答；除非用户明确要求 JSON/决策树，否则不要输出二元决策树 JSON。\n"
                    "2) 如果用户问的是【已有仓位管理】（止损/止盈/减仓/持有/加仓）：\n"
                    "   - 只围绕持仓管理回答，不要重新跑完整下单决策。\n"
                    "   - 先给结论（可以/不建议/条件允许），再给依据（结构/关键位/信号），再给风险控制（最大亏损、触发条件）。\n"
                    "3) 如果用户问题信息不足，最多问 1-2 个澄清点（例如仓位大小、入场价、止损距离）。\n"
                    "4) 不要编造数据；以用户消息附带的「当前图表K线数据」为准（与发送追问时屏幕上冻结的图表一致）。\n"
                    "5) K线棒型描述（上影线/下影线/实体大小/涨跌方向）必须以「K1数据·程序计算」字段中的数值为准，\n"
                    "   禁止凭记忆或猜测描述棒型特征——程序计算的 upper_wick/lower_wick/body 是唯一可信来源。\n"
                    "6) 每轮消息中的「订单时序硬锚点」是程序生成的最高优先级时序事实；必须按其中规则区分原始分析K1与后续K线，\n"
                    "   禁止使用原始分析K1或更早K线，倒推该K1收盘后才生成的订单已经成交、止盈或止损。\n"
                ),
            }
        )

        # [1] User — compact analysis reference (stable within this session)
        # Exclude volatile meta fields (timestamps, api_key) that change every run
        # and would break prefix caching across analysis records.
        meta = getattr(base_record, "meta", None)
        meta_stable: dict = {}
        if meta is not None:
            raw = meta.model_dump()
            meta_stable = {
                "symbol": raw.get("symbol", ""),
                "timeframe": raw.get("timeframe", ""),
                "bar_count": raw.get("bar_count", 0),
                "decision_stance": raw.get("decision_stance", ""),
                "model": (raw.get("ai_provider") or {}).get("model", ""),
            }
        s1 = getattr(base_record, "stage1_diagnosis", None)
        s2 = getattr(base_record, "stage2_decision", None)
        ref = {
            "meta": meta_stable,
            "stage1_diagnosis": s1 or {},
            "stage2_decision": s2 or {},
        }
        prefix.append(
            {
                "role": "user",
                "content": (
                    "## 上次分析结果（仅供参考，不是新的决策任务）\n\n"
                    f"```json\n{json.dumps(ref, ensure_ascii=False, indent=2)}\n```\n"
                ),
            }
        )

        # [2] Assistant — synthesise a reliable recall summary from the *parsed*
        # stage2_decision (program-validated JSON), NOT from stage2_response.content
        # (the raw model output which may contain hallucinated descriptions such as
        # wrong bar types or non-existent wick characteristics).
        #
        # Using the raw response content is dangerous: the model treats whatever it
        # previously said as ground truth, so any hallucination in the original reply
        # (e.g. "K1 has a long upper wick") gets recycled verbatim in follow-up turns.
        # The parsed stage2_decision has already been validated and normalised by the
        # program, so it is the authoritative source for follow-up context.
        s2_decision = getattr(base_record, "stage2_decision", None) or {}
        kline_data = getattr(base_record, "kline_data", None) or []
        # Build a concise, factual assistant recall message from validated fields only.
        recall_parts: list[str] = []
        decision = s2_decision.get("decision") or {}
        order_type = decision.get("order_type", "不下单")
        order_dir = decision.get("order_direction")
        reasoning = decision.get("reasoning", "")
        watch_points = decision.get("watch_points") or []
        bar_analysis = s2_decision.get("bar_analysis") or {}
        # ── K1 factual description (derived directly from kline_data, not from
        #    the model's potentially erroneous prose) ──────────────────────────
        k1_desc = ""
        if kline_data:
            k1 = kline_data[0]  # seq=1, newest closed bar
            k1_open = k1.get("open", 0)
            k1_high = k1.get("high", 0)
            k1_low = k1.get("low", 0)
            k1_close = k1.get("close", 0)
            k1_vol = k1.get("volume", 0)
            upper_wick = round(k1_high - max(k1_open, k1_close), 3)
            lower_wick = round(min(k1_open, k1_close) - k1_low, 3)
            body = round(abs(k1_close - k1_open), 3)
            full_range = round(k1_high - k1_low, 3)
            body_ratio = round(body / full_range, 2) if full_range > 0 else 0
            direction_zh = "阴线" if k1_close < k1_open else ("阳线" if k1_close > k1_open else "平盘")
            k1_bar_type = bar_analysis.get("bar_type", "")
            k1_desc = (
                f"K1（最新已收盘）：{direction_zh}，开={k1_open}，高={k1_high}，"
                f"低={k1_low}，收={k1_close}，量={k1_vol}；"
                f"实体={body}（占比{body_ratio:.0%}），上影={upper_wick}，下影={lower_wick}；"
                f"程序分类：{k1_bar_type}。"
            )
        recall_parts.append(f"【上次决策结果】{order_type}" + (f"（{order_dir}）" if order_dir else ""))
        if k1_desc:
            recall_parts.append(f"【K1数据·程序计算】{k1_desc}")
        if reasoning:
            # Truncate to avoid token bloat; the key facts are already in k1_desc
            recall_parts.append(f"【决策推理摘要】{reasoning[:600]}" + ("…" if len(reasoning) > 600 else ""))
        if watch_points:
            recall_parts.append("【关注点】" + "；".join(watch_points[:3]))
        recall_content = "\n".join(recall_parts)
        if recall_content.strip():
            prefix.append(
                {
                    "role": "assistant",
                    "content": recall_content,
                }
            )

        return prefix

    def send(
        self,
        user_text: str,
        cancel_token: CancelToken,
        on_reasoning_token: "Callable[[str], None] | None" = None,
        on_content_token: "Callable[[str], None] | None" = None,
    ) -> AIReply:
        """Send *user_text* to the AI and return the reply.

        Steps
        -----
        1. Build ``history_for_api`` from:
           - A follow-up advisory system prompt.
           - A compact reference summary of the completed analysis.
           - All previous free-chat turns.
           - New user message
        2. Call ``client.chat(history_for_api, cancel_token=cancel_token)``.
        3. Append to ``_history_full`` (with ``reasoning_content`` preserved).
        4. Call ``ledger.add(reply.usage)`` and
           ``pending_writer.append_followup(record_id, turn)``.
        5. Return the AIReply.

        When *cancel_token* is already set before the call, a
        ``FollowupTurn`` with ``cancelled=True`` is persisted and the
        ``CancelledError`` is re-raised.
        """
        self._turn += 1
        turn_number = self._turn

        # ── 1. Build history_for_api ──────────────────────────────────────────
        history_for_api: list[dict] = list(self._cached_prefix)  # copy stable prefix

        # Previous free-chat turns from history_full
        preserve_mimo = False
        if self._settings is not None:
            from pa_agent.ai.mimo_compat import is_mimo_provider

            provider = getattr(self._settings, "provider", None)
            if provider is not None:
                preserve_mimo = is_mimo_provider(
                    getattr(provider, "base_url", ""),
                    getattr(provider, "model", ""),
                )
        for msg in self._history_full:
            if msg["role"] == "user":
                history_for_api.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                assistant_msg: dict = {"role": "assistant", "content": msg["content"]}
                if (self.keep_reasoning_in_resend or preserve_mimo) and msg.get(
                    "reasoning_content"
                ):
                    assistant_msg["reasoning_content"] = msg["reasoning_content"]
                history_for_api.append(assistant_msg)

        # New user message — prepend immutable timing anchor and latest K-lines.
        user_parts: list[str] = []
        if self._order_timing_anchor:
            user_parts.append(self._order_timing_anchor)
        if self._kline_snapshot_fn is not None:
            try:
                kline_table = self._kline_snapshot_fn()
                if kline_table:
                    user_parts.append(
                        "## 当前图表K线数据（发送追问时已刷新并冻结图表，与屏幕一致）\n\n"
                        f"{kline_table}"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("kline_snapshot_fn failed: %s", exc)

        user_parts.append(user_text)
        user_content = "\n\n---\n\n".join(user_parts)

        history_for_api.append({"role": "user", "content": user_content})

        # ── 2. Resolve reasoning_effort ───────────────────────────────────────
        reasoning_effort = "high"
        if self._settings is not None:
            reasoning_effort = getattr(
                self._settings.provider, "reasoning_effort", "high"
            )

        # ── 3. Check cancellation before API call ─────────────────────────────
        from pa_agent.ai.deepseek_client import CancelledError

        if cancel_token.is_set():
            # Persist a cancelled turn and re-raise
            cancelled_turn = FollowupTurn(
                turn=turn_number,
                ts_ms=now_local_ms(),
                user=user_text,
                ai_content="",
                ai_reasoning=None,
                usage={
                    "prompt_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                cancelled=True,
            )
            self._pending_writer.append_followup(self._record_id, cancelled_turn)
            raise CancelledError("FreeChatSession.send cancelled before API call")

        # ── 4. Call the API (streaming) ───────────────────────────────────────
        try:
            reply = self._client.stream_chat(
                history_for_api,
                on_reasoning_token=on_reasoning_token,
                on_content_token=on_content_token,
                cancel_token=cancel_token,
                reasoning_effort=reasoning_effort,
            )
        except CancelledError:
            # Persist a cancelled turn and re-raise
            cancelled_turn = FollowupTurn(
                turn=turn_number,
                ts_ms=now_local_ms(),
                user=user_text,
                ai_content="",
                ai_reasoning=None,
                usage={
                    "prompt_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                cancelled=True,
            )
            self._pending_writer.append_followup(self._record_id, cancelled_turn)
            raise

        # ── 5. Append to history_full (with reasoning preserved) ──────────────
        self._history_full.append({"role": "user", "content": user_text})
        self._history_full.append({
            "role": "assistant",
            "content": reply.content,
            "reasoning_content": reply.reasoning_content,
        })

        # ── 6. Accumulate usage in ledger ─────────────────────────────────────
        self._ledger.add(reply.usage)

        # ── 7. Persist the followup turn ──────────────────────────────────────
        usage_dict = {
            "prompt_tokens": reply.usage.prompt_tokens,
            "cached_prompt_tokens": reply.usage.cached_prompt_tokens,
            "completion_tokens": reply.usage.completion_tokens,
            "total_tokens": reply.usage.total_tokens,
        }
        followup_turn = FollowupTurn(
            turn=turn_number,
            ts_ms=now_local_ms(),
            user=user_text,
            ai_content=reply.content,
            ai_reasoning=reply.reasoning_content or None,
            usage=usage_dict,
            cancelled=False,
        )
        self._pending_writer.append_followup(self._record_id, followup_turn)

        logger.debug(
            "FreeChatSession.send: turn=%d tokens=%d/%d",
            turn_number,
            reply.usage.prompt_tokens,
            reply.usage.completion_tokens,
        )

        return reply
