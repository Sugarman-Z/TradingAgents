"""Offline unit tests for the akshare fundamentals vendor (sina statements).

Same harness as the news/events suites: stub the akshare handles through the
module-under-test, feed DataFrames that mirror the live probe (报告日/公告日期
as 'YYYYMMDD' strings), and assert the point-in-time announcement filter, the
annual-frequency filter, the sina exchange-prefix mapping, and the
announcement-date join behind get_fundamentals.
"""

import pytest

import tradingagents.dataflows.akshare_fundamentals as anf
from tradingagents.dataflows.errors import NoMarketDataError

# Column subsets mirroring the live probe (trimmed names the vendor reads).
_BS_COLS = ["报告日", "公告日期", "货币资金", "存货", "资产总计", "负债合计",
            "归属于母公司股东权益合计"]
_IS_COLS = ["报告日", "公告日期", "营业总收入", "净利润", "归属于母公司所有者的净利润",
            "基本每股收益"]
_IND_COLS = ["日期", "摊薄每股收益(元)", "净资产收益率(%)", "净利润增长率(%)", "流动比率"]


def _make_table(cols):
    def table(*rows):
        data = [tuple(r) for r in rows]
        return _FakeDF(cols, data)

    return table


class _FakeDF:
    """Minimal DataFrame stand-in: iterrows + .empty + column access via get."""

    def __init__(self, columns, data):
        self.columns = list(columns)
        self._rows = [dict(zip(self.columns, r)) for r in data]

    @property
    def empty(self):
        return not self._rows

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, _RowView(row)

    def __len__(self):
        return len(self._rows)


class _RowView:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, key, default=None):
        return self._m.get(key, default)


def _install(monkeypatch, statements, indicators=None):
    """Patch both handles; `statements` maps sina_symbol -> table / _EMPTY."""
    report_calls = []
    indicator_calls = []

    def fake_report_sina(stock, symbol):
        report_calls.append((stock, symbol))
        table = statements[symbol]
        if table is _EMPTY:
            raise TypeError("'NoneType' object is not subscriptable")
        return table

    def fake_indicator(symbol, start_year):
        indicator_calls.append((symbol, start_year))
        if indicators is _EMPTY:
            raise TypeError("'NoneType' object is not subscriptable")
        return indicators

    fake_report_sina.calls = report_calls
    fake_indicator.calls = indicator_calls
    monkeypatch.setattr(anf.ak, "stock_financial_report_sina", fake_report_sina)
    monkeypatch.setattr(anf.ak, "stock_financial_analysis_indicator", fake_indicator)
    return fake_report_sina, fake_indicator


_EMPTY = object()


def _bs_row(period, announced, cash=50.0):
    return (period, announced, cash, 30.0, 3000.0, 1000.0, 2000.0)


def _is_row(period, announced, rev=900.0):
    return (period, announced, rev, 400.0, 300.0)


def _ind_row(period, eps=35.0):
    return (period, eps, 30.0, -2.0, 4.5)


@pytest.mark.unit
def test_balance_sheet_happy_path_with_pit_filter(monkeypatch):
    fake, _ = _install(monkeypatch, {"资产负债表": _make_table(_BS_COLS)(
        ("20260630", "20260815"),
        ("20251231", "20260417"),
        ("20260930", "20270115"),   # announced after the analysis date -> dropped
    )})
    out = anf.get_balance_sheet("600519.SS", "quarterly", "2026-09-10")
    assert fake.calls == [("sh600519", "资产负债表")]  # prefix mapping + bare call
    assert "2026-06-30" in out and "2025-12-31" in out
    assert "2026-09-30" not in out                    # unannounced period withheld
    assert "as of 2026-09-10" in out
    assert "latest published revision" in out          # restatement caveat stated


@pytest.mark.unit
def test_annual_frequency_keeps_december_periods_only(monkeypatch):
    _install(monkeypatch, {"资产负债表": _make_table(_BS_COLS)(
        ("20260630", "20260815"),
        ("20251231", "20260417"),
        ("20240630", "20240815"),
    )})
    out = anf.get_balance_sheet("600519.SS", "annual", "2026-09-10")
    assert "2025-12-31" in out
    assert "2024-06-30" not in out                     # interim period filtered
    assert "sina, annual," in out


@pytest.mark.unit
def test_all_unannounced_periods_get_sentinel(monkeypatch):
    _install(monkeypatch, {"资产负债表": _make_table(_BS_COLS)(
        ("20260930", "20270115"),
    )})
    out = anf.get_balance_sheet("600519.SS", "quarterly", "2026-09-10")
    assert "not yet public at this analysis date" in out
    assert "Do not fabricate figures" in out


@pytest.mark.unit
def test_empty_payload_is_a_data_state(monkeypatch):
    _install(monkeypatch, {"资产负债表": _EMPTY})
    out = anf.get_balance_sheet("600519.SS", "quarterly", "2026-09-10")
    assert "No balance sheet data available for 600519.SS on sina" in out


@pytest.mark.unit
def test_non_a_share_and_beijing_codes_raise(monkeypatch):
    fake, _ = _install(monkeypatch, {})
    with pytest.raises(NoMarketDataError):
        anf.get_balance_sheet("AAPL", "quarterly", "2026-09-10")
    with pytest.raises(NoMarketDataError):
        anf.get_balance_sheet("830799.BJ", "quarterly", "2026-09-10")
    assert fake.calls == []                            # rejected before any vendor call


@pytest.mark.unit
def test_get_fundamentals_joins_announcement_dates(monkeypatch):
    # The indicator interface has no announcement date; publication must come
    # from the income statement join. A period the join cannot prove public is
    # withheld even though its indicator row exists.
    _install(
        monkeypatch,
        {"利润表": _make_table(_IS_COLS)(
            ("20260630", "20260815"),
            ("20251231", "20260417"),
        )},
        indicators=_make_table(_IND_COLS)(
            _ind_row("20260630", eps=35.57),
            _ind_row("20251231", eps=65.66),
            _ind_row("20260930", eps=40.0),   # no announced income row -> withheld
        ),
    )
    out = anf.get_fundamentals("600519.SS", "2026-09-10")
    assert "2026-06-30" in out and "2025-12-31" in out
    assert "2026-09-30" not in out
    assert "35.57" in out
    assert "matching report was publicly announced" in out


@pytest.mark.unit
def test_get_fundamentals_requires_announced_periods(monkeypatch):
    _install(
        monkeypatch,
        {"利润表": _EMPTY},
        indicators=_make_table(_IND_COLS)(_ind_row("20260630")),
    )
    out = anf.get_fundamentals("600519.SS", "2026-09-10")
    assert "No publicly announced report periods" in out


@pytest.mark.unit
@pytest.mark.parametrize("ticker,prefix", [("600519", "sh600519"), ("000001.SZ", "sz000001")])
def test_code_forms_map_to_sina_prefixes(monkeypatch, ticker, prefix):
    fake, _ = _install(monkeypatch, {"资产负债表": _make_table(_BS_COLS)(
        ("20260630", "20260815"),
    )})
    anf.get_balance_sheet(ticker, "quarterly", "2026-09-10")
    assert fake.calls == [(prefix, "资产负债表")]
