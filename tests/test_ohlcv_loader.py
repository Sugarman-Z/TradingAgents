"""Offline unit tests for the frame-level OHLCV chain dispatcher.

load_configured_ohlcv must mirror route_to_vendor's fallthrough semantics at
the DataFrame level: NoMarketDataError falls through, real errors are
remembered, the unconfigured "default" chain keeps the yfinance loader.
"""

import pandas as pd
import pytest

import tradingagents.dataflows.ohlcv_loader as ol
from tradingagents.dataflows.config import get_config, set_config
from tradingagents.dataflows.errors import NoMarketDataError


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=10)
    return pd.DataFrame({
        "Date": dates,
        "Open": [10.0] * 10, "High": [11.0] * 10,
        "Low": [9.0] * 10, "Close": [10.5] * 10, "Volume": [100] * 10,
    })


@pytest.fixture(autouse=True)
def _restore_vendors():
    original = dict(get_config().get("data_vendors", {}))
    yield
    set_config({"data_vendors": original})


@pytest.mark.unit
def test_local_first_then_akshare(monkeypatch):
    set_config({"data_vendors": {"core_stock_apis": "local,akshare"}})
    monkeypatch.setattr(
        ol, "load_local_ohlcv",
        lambda symbol: (_ for _ in ()).throw(NoMarketDataError(symbol, symbol, "no file")),
    )
    served = _frame()
    monkeypatch.setattr(ol, "load_akshare_ohlcv", lambda symbol, curr_date: served)

    assert ol.load_configured_ohlcv("600519.SS", "2026-03-13") is served


@pytest.mark.unit
def test_local_frame_is_trimmed_to_curr_date(monkeypatch):
    set_config({"data_vendors": {"core_stock_apis": "local"}})
    frame = _frame()
    frame.loc[len(frame)] = [pd.Timestamp("2026-06-01")] + [99.0] * 4 + [999]
    monkeypatch.setattr(ol, "load_local_ohlcv", lambda symbol: frame)

    out = ol.load_configured_ohlcv("600519.SS", "2026-01-15")

    assert out["Date"].max() <= pd.Timestamp("2026-01-15")
    assert (out["Close"] == 10.5).all()  # the planted future row is gone


@pytest.mark.unit
def test_all_vendors_no_data_raises_no_market_data(monkeypatch):
    set_config({"data_vendors": {"core_stock_apis": "local,akshare"}})
    def _raise(symbol, *a, **kw):
        raise NoMarketDataError(symbol, symbol, "none")
    monkeypatch.setattr(ol, "load_local_ohlcv", _raise)
    monkeypatch.setattr(ol, "load_akshare_ohlcv", _raise)

    with pytest.raises(NoMarketDataError):
        ol.load_configured_ohlcv("600519.SS", "2026-03-13")


@pytest.mark.unit
def test_unconfigured_chain_keeps_yfinance_loader(monkeypatch):
    set_config({"data_vendors": {"core_stock_apis": "default"}})
    served = _frame()
    monkeypatch.setattr(ol, "load_ohlcv", lambda symbol, curr_date: served)

    assert ol.load_configured_ohlcv("AAPL", "2026-03-13") is served


@pytest.mark.unit
def test_frameless_vendor_is_skipped_loudly(monkeypatch):
    set_config({"data_vendors": {"core_stock_apis": "alpha_vantage"}})
    with pytest.raises(RuntimeError, match="No frame-capable vendor"):
        ol.load_configured_ohlcv("AAPL", "2026-03-13")


@pytest.mark.unit
def test_real_error_surfaces_when_no_vendor_serves(monkeypatch):
    set_config({"data_vendors": {"core_stock_apis": "local"}})
    def _boom(symbol):
        raise OSError("network down")
    monkeypatch.setattr(ol, "load_local_ohlcv", _boom)

    with pytest.raises(OSError, match="network down"):
        ol.load_configured_ohlcv("600519.SS", "2026-03-13")
