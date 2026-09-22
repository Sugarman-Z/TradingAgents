"""LIVE integration tests: the akshare re-sourcing through the real routing chain.

Unlike the offline suites (test_akshare_news / test_akshare_fundamentals /
test_akshare_events, fully stubbed), these hit eastmoney and sina for real —
config -> route_to_vendor -> vendor -> HTTP -> point-in-time filter — so they
verify that the two vendor swaps actually CONNECT, not just that the logic is
right.

Run explicitly when you want to prove the chain:
    python -m pytest tests/test_akshare_live_integration.py -v

Requires network access to eastmoney/sina (domestic-direct, keyless). The
offline unit suites remain the default regression and never touch the network.
"""

from datetime import date, timedelta

import pytest

from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor

pytestmark = pytest.mark.integration

TICKER = "600519.SS"


def _route_akshare(category: str) -> None:
    """Point one data category at the akshare chain (local sentinel fallback)."""
    set_config({"data_vendors": {category: "akshare,local"}})


@pytest.mark.integration
def test_live_news_returns_real_eastmoney_items():
    _route_akshare("news_data")
    today = date.today()
    out = route_to_vendor(
        "get_news", TICKER,
        (today - timedelta(days=30)).isoformat(), today.isoformat(),
    )
    assert "## 600519.SS News" in out, out[:300]
    assert "published:" in out            # real publish timestamps parsed
    assert "NO_DATA_AVAILABLE" not in out
    assert "Error" not in out


@pytest.mark.integration
def test_live_news_window_filter_rejects_far_past():
    # stock_news_em only serves the 10 newest items; a 2020 window must keep
    # none of them and fall to the plain between-message (client-side filter).
    _route_akshare("news_data")
    out = route_to_vendor("get_news", TICKER, "2020-01-01", "2020-01-31")
    assert "No news found for 600519.SS between 2020-01-01 and 2020-01-31" in out


@pytest.mark.integration
def test_live_balance_sheet_converts_to_yi_and_filters_pit():
    _route_akshare("fundamental_data")
    out = route_to_vendor("get_balance_sheet", TICKER, "quarterly", date.today().isoformat())
    assert "## 600519.SS Balance Sheet" in out
    assert "亿" in out                     # monetary conversion applied
    assert "2026-06-30" in out             # announced 2026-08-15, visible today
    assert "NO_DATA_AVAILABLE" not in out


@pytest.mark.integration
def test_live_balance_sheet_withholds_unannounced_period():
    # As of 2026-07-01 the interim report (announced 2026-08-15) was NOT yet
    # public: the 2026-06-30 row must be withheld even though sina serves it.
    _route_akshare("fundamental_data")
    out = route_to_vendor("get_balance_sheet", TICKER, "quarterly", "2026-07-01")
    assert "Announced |" in out            # a real (filtered) table came back
    assert "2026-06-30" not in out         # look-ahead blocked, live data


@pytest.mark.integration
def test_live_income_statement_and_cashflow_headers():
    _route_akshare("fundamental_data")
    income = route_to_vendor("get_income_statement", TICKER, "quarterly", date.today().isoformat())
    assert "## 600519.SS Income Statement" in income
    assert "归属于母公司所有者的净利润" in income
    cashflow = route_to_vendor("get_cashflow", TICKER, "quarterly", date.today().isoformat())
    assert "## 600519.SS Cash Flow Statement" in cashflow
    assert "经营活动产生的现金流量净额" in cashflow


@pytest.mark.integration
def test_live_get_fundamentals_ratios_join():
    _route_akshare("fundamental_data")
    out = route_to_vendor("get_fundamentals", TICKER, date.today().isoformat())
    assert "## 600519.SS Key Ratios" in out
    assert "净资产收益率(%)" in out
    assert "Announced |" in out            # join produced announcement dates
    assert "NO_DATA_AVAILABLE" not in out


@pytest.mark.integration
def test_live_non_a_share_degrades_to_routing_sentinel():
    _route_akshare("news_data")
    out = route_to_vendor("get_news", "AAPL", "2026-09-01", "2026-09-15")
    assert "NO_DATA_AVAILABLE" in out      # akshare gate -> local -> sentinel


@pytest.mark.integration
def test_live_market_ohlcv_serves_real_bars(tmp_path):
    """get_stock_data through the routing chain: real eastmoney hfq bars."""
    _route_akshare("core_stock_apis")
    # Empty local dir -> the chain must actually fall through to akshare;
    # cache in tmp so live runs don't write vendor caches into the repo.
    set_config({
        "local_data_dir": str(tmp_path / "no_local_files"),
        "data_cache_dir": str(tmp_path / "cache"),
    })
    today = date.today()
    out = route_to_vendor(
        "get_stock_data", TICKER,
        (today - timedelta(days=14)).isoformat(), today.isoformat(),
    )
    assert out.startswith("# Stock data for 600519.SS (akshare"), out[:200]
    assert "Date,Open,High,Low,Close,Volume" in out
    assert "NO_DATA_AVAILABLE" not in out


@pytest.mark.integration
def test_live_index_symbol_uses_index_api_not_stock_api(tmp_path):
    """000001.SS is the SSE Composite, not Ping An Bank (namespace collision)."""
    _route_akshare("core_stock_apis")
    set_config({"data_cache_dir": str(tmp_path / "cache")})
    today = date.today()
    out = route_to_vendor(
        "get_stock_data", "000001.SS",
        (today - timedelta(days=14)).isoformat(), today.isoformat(),
    )
    assert "index daily" in out.splitlines()[0]
    rows = [ln for ln in out.splitlines() if ln[:2] == "20"]
    assert rows, "index window must contain real bars"
    # An index series around 3000-4000 points; Ping An Bank hfq would be 4-5 digits.
    closes = [float(ln.split(",")[4]) for ln in rows]
    assert all(500 < c < 10000 for c in closes), closes[:3]


@pytest.mark.integration
def test_live_indicator_via_chain(tmp_path):
    _route_akshare("technical_indicators")
    set_config({"data_cache_dir": str(tmp_path / "cache")})
    today = date.today()
    out = route_to_vendor(
        "get_indicators", TICKER, "rsi",
        (today - timedelta(days=7)).isoformat(), 7,
    )
    assert out.startswith("## rsi values from"), out[:200]
    assert "N/A: Not a trading day" in out or "RSI" in out  # calendar rendering
    assert "NO_DATA_AVAILABLE" not in out
