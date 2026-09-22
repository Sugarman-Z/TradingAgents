"""Offline unit tests for the akshare per-stock news vendor.

Same harness as the events vendor tests: stub ``ak.stock_news_em`` through
the module-under-test's handle, feed DataFrames that mirror the real probe
(all-string columns, ``发布时间`` as 'YYYY-MM-DD HH:MM:SS' strings), and
assert the window filtering, no-news sentinels, and the code-form gate.
"""

import pandas as pd
import pytest

import tradingagents.dataflows.akshare_news as an
from tradingagents.dataflows.errors import NoMarketDataError

_COLS = ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]

# Sentinel: eastmoney returned nothing and akshare raised KeyError('code').
_EMPTY = object()


def _news_row(title, published, source="eastmoney"):
    return ["600519", title, f"{title}的内容摘要", published, source, "http://example.com/x"]


def _table(*rows):
    return pd.DataFrame(list(rows), columns=_COLS)


def _install(monkeypatch, table):
    calls = []

    def fake_news_em(symbol):
        calls.append(symbol)
        if table is _EMPTY:
            raise KeyError("code")
        return table

    fake_news_em.calls = calls
    monkeypatch.setattr(an.ak, "stock_news_em", fake_news_em)
    return fake_news_em


@pytest.mark.unit
def test_happy_path_filters_future_items(monkeypatch):
    fake = _install(monkeypatch, _table(
        _news_row("中报业绩说明会", "2026-09-08 10:00:00"),
        _news_row("未来公告", "2026-09-12 09:00:00"),  # after the window end
    ))
    out = an.get_news("600519.SS", "2026-09-01", "2026-09-10")
    assert fake.calls == ["600519"]  # bare 6-digit code passed to akshare
    assert "中报业绩说明会" in out
    assert "未来公告" not in out      # look-ahead blocked
    assert "## 600519.SS News, from 2026-09-01 to 2026-09-10" in out
    assert "published: 2026-09-08 10:00" in out
    assert "newest 1 of 2 items" in out  # kept/total documented, not hidden


@pytest.mark.unit
def test_window_is_inclusive_of_end_day(monkeypatch):
    _install(monkeypatch, _table(
        _news_row("窗口末日新闻", "2026-09-10 23:59:00"),
        _news_row("次日新闻", "2026-09-11 00:30:00"),
    ))
    out = an.get_news("600519.sS".upper(), "2026-09-01", "2026-09-10")
    assert "窗口末日新闻" in out       # end day's items kept (in_window half-open +1day)
    assert "次日新闻" not in out       # next-morning item dropped


@pytest.mark.unit
def test_all_items_outside_window_gets_plain_no_news(monkeypatch):
    _install(monkeypatch, _table(
        _news_row("旧闻", "2026-08-01 09:00:00"),
    ))
    out = an.get_news("600519.SS", "2026-09-01", "2026-09-10")
    assert "No news found for 600519.SS between 2026-09-01 and 2026-09-10" in out


@pytest.mark.unit
def test_empty_payload_is_a_data_state_not_an_error(monkeypatch):
    fake = _install(monkeypatch, _EMPTY)
    out = an.get_news("600519.SS", "2026-09-01", "2026-09-10")
    assert "No news found for 600519.SS on eastmoney" in out
    assert fake.calls == ["600519"]


@pytest.mark.unit
def test_non_a_share_code_raises(monkeypatch):
    fake = _install(monkeypatch, _EMPTY)
    with pytest.raises(NoMarketDataError):
        an.get_news("AAPL", "2026-09-01", "2026-09-10")
    assert fake.calls == []  # rejected at the code gate, vendor untouched


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["600519", "600519.SS", "600519.SZ"])
def test_code_forms_resolve_alike(monkeypatch, ticker):
    _install(monkeypatch, _table(
        _news_row("分红实施公告", "2026-06-25 08:30:00"),
    ))
    out = an.get_news(ticker, "2026-06-01", "2026-06-30")
    assert "分红实施公告" in out
