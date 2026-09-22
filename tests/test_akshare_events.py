"""Offline unit tests for the akshare A-share event vendor.

All eastmoney access is stubbed at the ``ak.stock_yysj_em`` boundary (the
house pattern: patch the vendor handle through the module under test), and
the clock is pinned through the module-level ``get_current_date`` binding —
the from-import seam that lets the live path run under a fixed "today".
Schedule cells use plain ``datetime.date`` / ``None``, mirroring what the
real akshare 1.x returns for this table.

Covered behaviours:
- point-in-time: schedules are withheld for runs dated in the past
- live path: the earliest upcoming disclosure wins across the open report
  periods, including the Jan-Apr annual+Q1 overlap
- data states: schedule-not-published (akshare raises TypeError on eastmoney's
  empty payload) falls back to the statutory deadline; settled rows are
  skipped; overdue filings get their own notice; unknown tickers get a
  no-record sentinel instead of an empty answer
- symbol forms: bare / .SS / .SZ codes all resolve to the same 6-digit match
"""

from datetime import date

import pandas as pd
import pytest

import tradingagents.dataflows.akshare_events as ev
from tradingagents.dataflows.errors import NoMarketDataError

_COLS = [
    "序号", "股票代码", "股票简称", "首次预约时间",
    "一次变更日期", "二次变更日期", "三次变更日期", "实际披露时间",
]

# Sentinel: this report period's schedule is not published yet — eastmoney
# serves an empty payload and akshare crashes on it with TypeError.
_NOT_PUBLISHED = object()


def _row(code, scheduled, actual=None, changes=(None, None, None)):
    """One schedule-table row; dates as datetime.date like real akshare 1.x."""
    return [1, code, "Fake Name", scheduled, *changes, actual]


def _table(*rows):
    return pd.DataFrame(list(rows), columns=_COLS)


def _install(monkeypatch, tables, today=None):
    """Patch the vendor handle (and optionally pin the clock); returns the fake."""
    calls = []

    def fake_yysj(symbol, date):
        calls.append(date)
        table = tables[date]
        if table is _NOT_PUBLISHED:
            raise TypeError("'NoneType' object is not subscriptable")
        return table

    fake_yysj.calls = calls
    monkeypatch.setattr(ev.ak, "stock_yysj_em", fake_yysj)
    if today is not None:
        monkeypatch.setattr(ev, "get_current_date", lambda: today)
    return fake_yysj


@pytest.mark.unit
def test_non_a_share_code_raises_without_touching_vendor(monkeypatch):
    fake = _install(monkeypatch, {})
    with pytest.raises(NoMarketDataError):
        ev.get_earnings_calendar("AAPL", "2026-08-01")
    assert fake.calls == []  # rejected at the code-form gate, no network


@pytest.mark.unit
def test_schedule_withheld_for_historical_run(monkeypatch):
    fake = _install(monkeypatch, {})  # real clock: 2001 is always in the past
    out = ev.get_earnings_calendar("600519.SS", "2001-01-01")
    assert "withheld" in out
    assert "no historical vintage" in out
    assert fake.calls == []  # withheld notice must not query anything


@pytest.mark.unit
def test_scheduled_disclosure_happy_path(monkeypatch):
    fake = _install(monkeypatch, {
        "20260630": _table(_row("600519", date(2026, 8, 15))),
        "20260930": _NOT_PUBLISHED,
    }, today="2026-08-01")
    out = ev.get_earnings_calendar("600519.SS", "2026-08-01")
    assert "2026 interim report (period 2026-06-30)" in out
    assert "2026-08-15" in out
    assert "(in 14 days)" in out
    assert "2026-08-31" in out  # statutory deadline shown alongside
    assert fake.calls == ["20260630", "20260930"]  # only open seasons queried


@pytest.mark.unit
def test_unpublished_schedule_falls_back_to_statutory_deadline(monkeypatch):
    _install(monkeypatch, {"20260930": _NOT_PUBLISHED}, today="2026-09-15")
    out = ev.get_earnings_calendar("600519.SS", "2026-09-15")
    assert "2026 Q3 report (period 2026-09-30)" in out
    assert "due by 2026-10-31" in out
    assert "(in 46 days)" in out


@pytest.mark.unit
def test_annual_q1_overlap_earliest_upcoming_wins(monkeypatch):
    # Jan-Apr three seasons coexist: the annual (2025-12-31 period) and Q1
    # (2026-03-31 period) are both open on 2026-04-10; the earliest scheduled
    # disclosure must win even though it comes from the *earlier* period.
    fake = _install(monkeypatch, {
        "20251231": _table(_row("600519", date(2026, 4, 17))),
        "20260331": _table(_row("600519", date(2026, 4, 25))),
        "20260630": _NOT_PUBLISHED,
    }, today="2026-04-10")
    out = ev.get_earnings_calendar("600519.SS", "2026-04-10")
    assert "2025 annual report (period 2025-12-31)" in out
    assert "2026-04-17" in out
    assert "(in 7 days)" in out
    assert fake.calls == ["20251231", "20260331", "20260630"]


@pytest.mark.unit
def test_settled_disclosure_is_skipped(monkeypatch):
    # By 2026-08-20 the interim (disclosed 08-15) is settled: the answer must
    # look forward to the Q3 report, not echo the settled one.
    _install(monkeypatch, {
        "20260630": _table(_row("600519", date(2026, 8, 15), actual=date(2026, 8, 15))),
        "20260930": _NOT_PUBLISHED,
    }, today="2026-08-20")
    out = ev.get_earnings_calendar("600519.SS", "2026-08-20")
    assert "Q3 report" in out
    assert "interim" not in out


@pytest.mark.unit
def test_overdue_filing_gets_its_own_notice(monkeypatch):
    # Scheduled 08-15 but no actual disclosure by 08-20 -> overdue, which
    # outranks the next period's not-yet-published schedule.
    _install(monkeypatch, {
        "20260630": _table(_row("600519", date(2026, 8, 15))),
        "20260930": _NOT_PUBLISHED,
    }, today="2026-08-20")
    out = ev.get_earnings_calendar("600519.SS", "2026-08-20")
    assert "overdue" in out
    assert "2026-08-15" in out


@pytest.mark.unit
def test_reschedule_dates_are_listed(monkeypatch):
    _install(monkeypatch, {
        "20260630": _table(_row(
            "600519", date(2026, 8, 15),
            changes=(date(2026, 8, 12), date(2026, 8, 14), None),
        )),
        "20260930": _NOT_PUBLISHED,
    }, today="2026-08-01")
    out = ev.get_earnings_calendar("600519.SS", "2026-08-01")
    assert "Rescheduled on: 2026-08-12, 2026-08-14" in out


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["600519", "600519.SS", "600519.SZ"])
def test_bare_and_suffixed_codes_resolve_alike(monkeypatch, ticker):
    _install(monkeypatch, {
        "20260630": _table(_row("600519", date(2026, 8, 15))),
        "20260930": _NOT_PUBLISHED,
    }, today="2026-08-01")
    out = ev.get_earnings_calendar(ticker, "2026-08-01")
    assert "2026-08-15" in out


@pytest.mark.unit
def test_unknown_ticker_gets_no_record_sentinel(monkeypatch):
    # Tables exist for every queried period but never list our code (Beijing
    # exchange, delisted, or unscheduled): an explicit no-record sentinel, not
    # the deadline fallback and not an empty answer.
    other = _table(_row("000001", date(2026, 8, 15)))
    _install(monkeypatch, {
        "20260630": other,
        "20260930": _table(_row("000001", date(2026, 10, 28))),
    }, today="2026-08-01")
    out = ev.get_earnings_calendar("600519.SS", "2026-08-01")
    assert "No disclosure-schedule record" in out
    assert "Do not invent dates" in out


# ---------------------------------------------------------------- earnings history

_YJBB_COLS = [
    "股票代码", "每股收益", "营业总收入-营业总收入", "营业总收入-同比增长",
    "净利润-净利润", "净利润-同比增长", "最新公告日期",
]


def _yjbb_row(code, announced, eps=1.50, rev=2.0e9, rev_yoy=10.0, np_=1.0e9, np_yoy=8.0):
    """One all-market earnings-table row; column names mirror stock_yjbb_em."""
    return [code, eps, rev, rev_yoy, np_, np_yoy, announced]


def _yjbb_table(*rows):
    return pd.DataFrame(list(rows), columns=_YJBB_COLS)


def _install_yjbb(monkeypatch, tables, today=None):
    """Patch stock_yjbb_em; tables maps 'YYYYMMDD' -> DataFrame / _NOT_PUBLISHED."""
    calls = []

    def fake_yjbb(date):
        calls.append(date)
        table = tables[date]
        if table is _NOT_PUBLISHED:
            raise TypeError("'NoneType' object is not subscriptable")
        return table

    fake_yjbb.calls = calls
    monkeypatch.setattr(ev.ak, "stock_yjbb_em", fake_yjbb)
    if today is not None:
        monkeypatch.setattr(ev, "get_current_date", lambda: today)
    return fake_yjbb


@pytest.mark.unit
def test_earnings_history_happy_path(monkeypatch):
    fake = _install_yjbb(monkeypatch, {
        "20260630": _yjbb_table(
            _yjbb_row("600519", date(2026, 7, 18), eps=8.2, rev=9.0e9, rev_yoy=18.0, np_=4.0e9, np_yoy=15.0)
        ),
        "20260331": _yjbb_table(_yjbb_row("600519", date(2026, 4, 25))),
        "20251231": _NOT_PUBLISHED,  # older periods may serve nothing
        "20250930": _NOT_PUBLISHED,
    }, today="2026-08-01")
    out = ev.get_earnings_history("600519.SS", "2026-08-01")
    assert "2026-06-30" in out
    assert "8.20" in out and "18.00%" in out and "15.00%" in out
    assert "2026-07-18" in out  # announcement date shown, not just period
    assert "2026-03-31" in out  # older settled quarter kept, newest first
    assert "positive in 2/2 settled quarters" in out
    assert fake.calls == ["20260630", "20260331", "20251231", "20250930"]


@pytest.mark.unit
def test_earnings_history_withholds_unannounced_rows(monkeypatch):
    # The interim row exists today but was only announced 2026-08-15 — after
    # the 2026-08-01 analysis date. Serving it would leak the future; the
    # answer must fall back to the quarters that were public by then.
    _install_yjbb(monkeypatch, {
        "20260630": _yjbb_table(_yjbb_row("600519", date(2026, 8, 15))),
        "20260331": _yjbb_table(_yjbb_row("600519", date(2026, 4, 25))),
        "20251231": _NOT_PUBLISHED,
        "20250930": _NOT_PUBLISHED,
    }, today="2026-08-01")
    out = ev.get_earnings_history("600519.SS", "2026-08-01")
    assert "2026-03-31" in out       # settled quarter kept
    assert "2026-06-30" not in out   # unannounced row withheld
    assert "positive in 1/1" in out


@pytest.mark.unit
def test_earnings_history_sentinel_when_nothing_was_public(monkeypatch):
    _install_yjbb(monkeypatch, {
        "20260630": _yjbb_table(_yjbb_row("600519", date(2026, 8, 15))),
        "20260331": _yjbb_table(_yjbb_row("600519", date(2026, 8, 10))),
        "20251231": _NOT_PUBLISHED,
        "20250930": _NOT_PUBLISHED,
    }, today="2026-08-01")
    out = ev.get_earnings_history("600519.SS", "2026-08-01")
    assert "No settled earnings history" in out
    assert "Do not invent figures" in out


@pytest.mark.unit
def test_earnings_history_newly_listed_gets_sentinel(monkeypatch):
    _install_yjbb(monkeypatch, {
        "20260630": _yjbb_table(_yjbb_row("000001", date(2026, 7, 18))),
        "20260331": _yjbb_table(_yjbb_row("000001", date(2026, 4, 25))),
        "20251231": _NOT_PUBLISHED,
        "20250930": _NOT_PUBLISHED,
    }, today="2026-08-01")
    out = ev.get_earnings_history("600519.SS", "2026-08-01")
    assert "newly listed" in out


# ---------------------------------------------------------------- corporate actions

_FHPS_COLS = ["报告期", "预案公告日", "股权登记日", "除权除息日", "方案进度", "现金分红-现金分红比例描述"]


def _fhps_row(report_period, announced, progress="实施分配", desc="10派30.00元(含税)",
              record=None, ex_date=None):
    return [report_period, announced, record, ex_date, progress, desc]


def _fhps_table(*rows):
    return pd.DataFrame(list(rows), columns=_FHPS_COLS)


def _install_fhps(monkeypatch, table):
    calls = []

    def fake_fhps(symbol):
        calls.append(symbol)
        return table

    fake_fhps.calls = calls
    monkeypatch.setattr(ev.ak, "stock_fhps_detail_em", fake_fhps)
    return fake_fhps


@pytest.mark.unit
def test_corporate_actions_happy_path(monkeypatch):
    fake = _install_fhps(monkeypatch, _fhps_table(
        _fhps_row(date(2025, 12, 31), date(2026, 3, 26),
                  record=date(2026, 6, 4), ex_date=date(2026, 6, 5)),
        _fhps_row(date(2024, 12, 31), date(2025, 3, 20)),  # kept: within 3y
        _fhps_row(date(2022, 12, 31), date(2023, 3, 20)),  # dropped: outside 3y
    ))
    out = ev.get_corporate_actions("600519.SS", "2026-08-01")
    assert fake.calls == ["600519"]  # bare 6-digit code passed to akshare
    assert "10派30.00元(含税)" in out
    assert "2026-06-05" in out  # ex-date shown
    assert "2025-03-20" in out  # 2024 plan kept (announced within window)
    assert "2023-03-20" not in out  # 2022 report period is outside the cutoff
    assert out.index("| Announced |") < out.index("10派30.00")  # newest first ordering header before rows


@pytest.mark.unit
def test_corporate_actions_withholds_future_plans(monkeypatch):
    _install_fhps(monkeypatch, _fhps_table(
        _fhps_row(date(2026, 6, 30), date(2026, 8, 15), desc="10派5.00元(含税)"),  # announced after as_of
        _fhps_row(date(2025, 12, 31), date(2026, 3, 26)),
    ))
    out = ev.get_corporate_actions("600519.SS", "2026-08-01")
    assert "2026-03-26" in out                         # announced plan kept
    assert "10派5.00" not in out.split("withheld")[0]  # future plan not in the table body
    assert "withheld as future information" in out
    assert "1 record" in out


@pytest.mark.unit
def test_corporate_actions_sentinel_for_zero_dividend(monkeypatch):
    _install_fhps(monkeypatch, _fhps_table(
        _fhps_row(date(2025, 12, 31), date(2026, 3, 26), progress="不分配", desc=""),
        _fhps_row(None, None),  # undatable row cannot be proven public -> dropped
    ))
    out = ev.get_corporate_actions("600519.SS", "2026-08-01")
    assert "No corporate actions" in out
    assert "Zero-dividend stretches are common" in out
