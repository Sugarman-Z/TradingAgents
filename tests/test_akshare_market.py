"""Offline unit tests for the akshare market-data vendor (no network).

Stubs ``ak.stock_zh_a_hist`` / ``ak.stock_zh_index_daily_em`` and runs the
real load/format/PIT/cache/stale logic on synthetic frames, the same pattern
as test_akshare_news / test_akshare_fundamentals.
"""

from datetime import datetime

import akshare as ak
import pandas as pd
import pytest

import tradingagents.dataflows.akshare_market as am
from tradingagents.dataflows.config import get_config, set_config
from tradingagents.dataflows.errors import NoMarketDataError


def _hist_frame(periods: int = 80, start: str = "2026-01-01") -> pd.DataFrame:
    """A stock_zh_a_hist-shaped frame (Chinese columns, ascending)."""
    dates = pd.bdate_range(start, periods=periods)
    closes = [1600.0 + i for i in range(len(dates))]
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "股票代码": ["600519"] * len(dates),
        "开盘": [c - 5 for c in closes],
        "收盘": closes,
        "最高": [c + 8 for c in closes],
        "最低": [c - 8 for c in closes],
        "成交量": [30000 + i for i in range(len(dates))],
        "成交额": [1.0e8] * len(dates),
        "振幅": [1.0] * len(dates),
        "涨跌幅": [0.1] * len(dates),
        "涨跌额": [1.0] * len(dates),
        "换手率": [0.2] * len(dates),
    })


def _index_frame(periods: int = 80, start: str = "2026-01-01") -> pd.DataFrame:
    """A stock_zh_index_daily_em-shaped frame (lowercase English columns)."""
    dates = pd.bdate_range(start, periods=periods)
    closes = [3000.0 + i for i in range(len(dates))]
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": [c - 5 for c in closes],
        "close": closes,
        "high": [c + 8 for c in closes],
        "low": [c - 8 for c in closes],
        "volume": [1.0e8 + i for i in range(len(dates))],
    })


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path):
    """Per-test cache dir (and data_vendors) so no state leaks between tests."""
    original_cache = get_config().get("data_cache_dir")
    original_local = get_config().get("local_data_dir")
    original_vendors = dict(get_config().get("data_vendors", {}))
    set_config({"data_cache_dir": str(tmp_path / "cache")})
    yield
    set_config({
        "data_cache_dir": original_cache,
        "local_data_dir": original_local,
        "data_vendors": original_vendors,
    })


@pytest.mark.unit
def test_get_stock_maps_columns_window_and_header(monkeypatch):
    monkeypatch.setattr(am.ak, "stock_zh_a_hist", lambda **kw: _hist_frame())

    out = am.get_stock("600519.SS", "2026-03-02", "2026-03-13")

    assert out.startswith("# Stock data for 600519.SS (akshare")
    assert "hfq" in out.splitlines()[0]          # adjustment declared
    assert "# Total records:" in out
    assert "Date,Open,High,Low,Close,Volume" in out  # canonical dialect, extras dropped
    assert "成交额" not in out and "换手率" not in out
    # Only rows inside the window, dates ascending strings.
    body_dates = [ln.split(",")[0] for ln in out.splitlines() if ln[:2] == "20"]
    assert body_dates == sorted(body_dates)
    assert datetime.strptime(body_dates[0], "%Y-%m-%d") >= datetime(2026, 3, 2)
    assert datetime.strptime(body_dates[-1], "%Y-%m-%d") <= datetime(2026, 3, 13)


@pytest.mark.unit
def test_index_namespace_routes_to_index_api(monkeypatch):
    """000001.SS is the SSE index; 000001.SZ is Ping An Bank — the suffix decides."""
    calls = {"hist": [], "index": []}

    def fake_hist(**kwargs):
        calls["hist"].append(kwargs)
        return _hist_frame()

    def fake_index(symbol=None):
        calls["index"].append(symbol)
        return _index_frame()

    monkeypatch.setattr(am.ak, "stock_zh_a_hist", fake_hist)
    monkeypatch.setattr(am.ak, "stock_zh_index_daily_em", fake_index)

    out = am.get_stock("000001.SS", "2026-03-02", "2026-03-13")
    assert calls["index"] == ["sh000001"]
    assert calls["hist"] == []
    assert "index daily" in out.splitlines()[0]

    out = am.get_stock("000001.SZ", "2026-03-02", "2026-03-13")
    assert calls["hist"] and calls["hist"][0]["symbol"] == "000001"
    assert calls["hist"][0]["adjust"] == "hfq"
    assert calls["index"] == ["sh000001"]  # unchanged: still exactly one index call


@pytest.mark.unit
def test_empty_payload_crash_is_no_data(monkeypatch):
    def boom(**kwargs):
        raise KeyError("data")

    monkeypatch.setattr(am.ak, "stock_zh_a_hist", boom)
    with pytest.raises(NoMarketDataError):
        am.load_ohlcv("600519.SS", "2026-03-13")


@pytest.mark.unit
def test_empty_frame_is_no_data(monkeypatch):
    monkeypatch.setattr(am.ak, "stock_zh_a_hist", lambda **kw: pd.DataFrame())
    with pytest.raises(NoMarketDataError):
        am.load_ohlcv("600519.SS", "2026-03-13")


@pytest.mark.unit
def test_cache_suppresses_second_fetch(monkeypatch):
    fetches = []

    def fake_hist(**kwargs):
        fetches.append(kwargs)
        return _hist_frame()

    monkeypatch.setattr(am.ak, "stock_zh_a_hist", fake_hist)

    first = am.load_ohlcv("600519.SS", "2026-03-13")
    second = am.load_ohlcv("600519.SS", "2026-02-10")  # historical: cache reusable

    assert len(fetches) == 1
    assert first["Date"].max() <= pd.Timestamp("2026-03-13")
    assert second["Date"].max() <= pd.Timestamp("2026-02-10")


@pytest.mark.unit
def test_pit_trims_rows_after_curr_date(monkeypatch):
    monkeypatch.setattr(am.ak, "stock_zh_a_hist", lambda **kw: _hist_frame())
    frame = am.load_ohlcv("600519.SS", "2026-02-02")
    assert not frame.empty
    assert frame["Date"].max() <= pd.Timestamp("2026-02-02")


@pytest.mark.unit
def test_stale_coverage_is_rejected(monkeypatch):
    # Frame ends around 2026-01-30 (80 business days from 2026-01-01 minus
    # buffer); asking for March is > 10 days beyond coverage -> stale refusal.
    monkeypatch.setattr(am.ak, "stock_zh_a_hist", lambda **kw: _hist_frame(periods=20))
    with pytest.raises(NoMarketDataError, match="stale"):
        am.load_ohlcv("600519.SS", "2026-03-13")


@pytest.mark.unit
def test_indicator_output_shape(monkeypatch):
    monkeypatch.setattr(am.ak, "stock_zh_a_hist", lambda **kw: _hist_frame())
    out = am.get_indicator("600519.SS", "rsi", "2026-03-13", 10)

    assert out.startswith("## rsi values from 2026-03-03 to 2026-03-13")
    # One line per calendar day, weekends marked non-trading.
    lines = [ln for ln in out.splitlines() if ln[:2] == "20"]
    assert len(lines) == 11
    assert any("N/A: Not a trading day" in ln for ln in lines)
    assert "RSI:" in out  # shared indicator catalog description


@pytest.mark.unit
def test_unsupported_indicator_rejected_before_fetch(monkeypatch):
    called = []
    monkeypatch.setattr(am.ak, "stock_zh_a_hist", lambda **kw: called.append(1) or _hist_frame())
    with pytest.raises(ValueError, match="not supported"):
        am.get_indicator("600519.SS", "stoch_rsi", "2026-03-13", 10)
    assert called == []


@pytest.mark.unit
def test_route_to_vendor_falls_through_local_to_akshare(monkeypatch, tmp_path):
    """e2e through the router: no local file -> the akshare vendor serves."""
    from tradingagents.dataflows.interface import route_to_vendor

    set_config({
        "data_vendors": {"core_stock_apis": "local,akshare"},
        "local_data_dir": str(tmp_path / "no_local_files"),
    })
    monkeypatch.setattr(am.ak, "stock_zh_a_hist", lambda **kw: _hist_frame())

    out = route_to_vendor("get_stock_data", "600519.SS", "2026-03-02", "2026-03-13")

    assert out.startswith("# Stock data for 600519.SS (akshare")
    assert "NO_DATA_AVAILABLE" not in out
