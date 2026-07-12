"""Gold default symbol / exchange normalization."""
from __future__ import annotations

import pytest

from pa_agent.data.market_defaults import (
    BINANCE_DEFAULT_SYMBOL,
    BINANCE_SUPPORTED_SYMBOLS,
    GOLD_MT5_SYMBOL,
    GOLD_TV_EXCHANGE,
    GOLD_TV_SYMBOL,
    TV_CRYPTO_EXCHANGES,
    TV_GOLD_SYMBOL_BY_EXCHANGE,
    ashare_tv_probe_order,
    infer_ashare_tv_exchange,
    is_partial_tv_symbol_input,
    is_tv_exchange_auto,
    migrate_general_gold_defaults,
    normalize_gold_symbol_for_kind,
    normalize_gold_tv_exchange,
    resolve_tv_gold_pair,
    resolve_tv_pair,
    resolve_tv_fetch_pair,
    tv_auto_probe_plan,
    tv_forex_auto_probe_plan,
)
from pa_agent.data.tv_symbol_lookup import TvSymbolNotFoundError
from pa_agent.data.tradingview import TV_EXCHANGE_PRESETS


def test_crypto_symbol_migrates_to_gold():
    assert normalize_gold_symbol_for_kind("mt5", "BTCUSD") == GOLD_MT5_SYMBOL
    assert normalize_gold_symbol_for_kind("tradingview", "BTCUSDT") == GOLD_TV_SYMBOL


def test_mt5_suffix_on_tv_becomes_xauusd():
    assert normalize_gold_symbol_for_kind("tradingview", "XAUUSDm") == GOLD_TV_SYMBOL


def test_tv_exchange_auto_preserved():
    assert normalize_gold_tv_exchange("") == ""
    assert normalize_gold_tv_exchange("AUTO") == ""
    assert normalize_gold_tv_exchange("BINANCE") == ""
    assert normalize_gold_tv_exchange("OANDA") == "OANDA"


def test_tv_forex_auto_probe_tries_all_forex_presets():
    plan = tv_forex_auto_probe_plan("XAUUSD")
    exchanges = [ex for ex, _ in plan]
    assert exchanges == [
        ex
        for ex in TV_EXCHANGE_PRESETS
        if ex
        and ex not in {"SSE", "SZSE", "HKEX"}
        and ex not in TV_CRYPTO_EXCHANGES
        and ex in TV_GOLD_SYMBOL_BY_EXCHANGE
    ]
    assert ("OANDA", "XAUUSD") in plan
    assert ("TVC", "GOLD") in plan


def test_tv_auto_probe_ashare_still_two_venues():
    assert tv_auto_probe_plan("600519") == [("SSE", "600519"), ("SZSE", "600519")]


def test_tvc_xauusd_is_invalid_pair_fixed_to_gold():
    ex, sym, adjusted = resolve_tv_gold_pair("TVC", "XAUUSD")
    assert ex == "TVC"
    assert sym == "GOLD"
    assert adjusted is True


def test_oanda_xauusd_unchanged():
    ex, sym, adjusted = resolve_tv_gold_pair("OANDA", "XAUUSD")
    assert (ex, sym, adjusted) == ("OANDA", "XAUUSD", False)


def test_tv_ashare_code_not_rewritten_to_gold():
    ex, sym, adjusted = resolve_tv_pair("OANDA", "600519")
    assert ex == "SSE"
    assert sym == "600519"
    assert adjusted is True


def test_tv_ashare_szse_infer():
    assert infer_ashare_tv_exchange("000001") == "SZSE"
    ex, sym, adjusted = resolve_tv_pair("", "000001")
    assert ex == "" and sym == "000001" and adjusted is False


def test_tv_ashare_sse_explicit():
    ex, sym, adjusted = resolve_tv_pair("SSE", "600519")
    assert (ex, sym, adjusted) == ("SSE", "600519", False)


def test_auto_exchange_defers_ashare_resolution():
    ex, sym, adjusted = resolve_tv_pair("", "688981")
    assert ex == "" and sym == "688981" and adjusted is False
    assert is_tv_exchange_auto("")


def test_ashare_probe_order_puts_inferred_first():
    assert ashare_tv_probe_order("688981") == ("SSE", "SZSE")
    assert ashare_tv_probe_order("000001") == ("SZSE", "SSE")


def test_star_board_688981_must_be_sse_not_szse():
    assert infer_ashare_tv_exchange("688981") == "SSE"
    ex, sym, adjusted = resolve_tv_pair("SZSE", "688981")
    assert ex == "SSE" and sym == "688981" and adjusted is True


def test_tradingview_kind_keeps_ashare_symbol():
    assert normalize_gold_symbol_for_kind("tradingview", "600519") == "600519"


def test_numeric_hk_style_code_not_rewritten_to_xauusd():
    ex, sym, adjusted = resolve_tv_pair("", "00988")
    assert (ex, sym, adjusted) == ("", "00988", False)
    ex2, sym2, adjusted2 = resolve_tv_gold_pair("OANDA", "00988")
    assert sym2 == "00988" and adjusted2 is False


def test_partial_digit_input_not_forced_to_gold():
    assert is_partial_tv_symbol_input("00")
    ex, sym, adjusted = resolve_tv_pair("", "009")
    assert sym == "009" and adjusted is False


def test_migrate_general_fixes_tvc_xauusd():
    general = {
        "last_data_source": "tradingview",
        "last_symbol": "XAUUSD",
        "last_tradingview_exchange": "TVC",
    }
    migrate_general_gold_defaults(general)
    assert general["last_tradingview_exchange"] == "TVC"
    assert general["last_symbol"] == "GOLD"


@pytest.mark.parametrize(
    ("display_symbol", "feed_symbol"),
    [
        ("BTCUSDT", "BTCUSDT.P"),
        ("ETHUSDT", "ETHUSDT.P"),
        ("XAUUSDT", "XAUUSDT.P"),
        ("QQQUSDT", "QQQUSDT.P"),
    ],
)
def test_binance_symbols_resolve_to_perpetual_feed(display_symbol, feed_symbol):
    assert resolve_tv_fetch_pair("BINANCE", display_symbol) == ("BINANCE", feed_symbol)
    assert resolve_tv_fetch_pair("BINANCE", feed_symbol) == ("BINANCE", feed_symbol)


def test_binance_catalog_is_exactly_four_clean_symbols():
    assert BINANCE_SUPPORTED_SYMBOLS == ("BTCUSDT", "ETHUSDT", "XAUUSDT", "QQQUSDT")
    assert BINANCE_DEFAULT_SYMBOL == "BTCUSDT"


def test_binance_rejects_symbols_without_network_lookup():
    with pytest.raises(TvSymbolNotFoundError, match="仅支持"):
        resolve_tv_fetch_pair("BINANCE", "SOLUSDT")


def test_migrate_invalid_binance_symbol_to_default():
    general = {
        "last_data_source": "tradingview",
        "last_symbol": "SOLUSDT.P",
        "last_tradingview_exchange": "BINANCE",
    }
    migrate_general_gold_defaults(general)
    assert general["last_symbol"] == "BTCUSDT"
