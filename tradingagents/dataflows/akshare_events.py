"""akshare-based A-share event data functions.

Serves the ``event_data`` category for Chinese A-share tickers via eastmoney
endpoints (keyless, domestic-direct). Functions take the framework's canonical
ticker form (e.g. ``600519.SS``) and translate to the bare 6-digit code that
akshare expects.

Two behaviours worth knowing before changing this module:

- Empty-payload quirk: when a disclosure schedule has not been published yet
  (e.g. the Q3 schedule queried in mid-September), eastmoney returns an empty
  payload and akshare crashes on it with ``TypeError``. That is a *data state*
  (schedule not out yet), not a vendor error, so it is translated into an
  informative string. Real failures (network, schema) propagate so the routing
  layer's OPTIONAL_CATEGORIES machinery logs and degrades them uniformly.
- Point-in-time rule: disclosure schedules carry no vintage — they change after
  publication (the per-order change columns prove it) and the table records no
  "published on" timestamp. A run dated in the past therefore gets a
  withholding notice instead of today's table, mirroring
  ``withhold_live_profile``.
"""

import calendar
import logging
from datetime import date, datetime

import akshare as ak
import pandas as pd

from .akshare_common import a_share_code as _a_share_code
from .akshare_common import fmt as _fmt
from .akshare_common import to_date as _to_date
from .akshare_common import to_float as _to_float
from .utils import get_current_date

logger = logging.getLogger(__name__)

# Statutory disclosure deadlines by report period (exchange rules): Q1 by
# Apr 30, interim by Aug 31, Q3 by Oct 31, annual by Apr 30 of the NEXT year
# (hence the year-shift flag). Used as the fallback answer when a period's
# schedule is not published yet.
_DEADLINES = {
    3: (4, 30, False),    # Q1 report      (period ends Mar 31)
    6: (8, 31, False),    # interim report (period ends Jun 30)
    9: (10, 31, False),   # Q3 report      (period ends Sep 30)
    12: (4, 30, True),    # annual report  (period ends Dec 31)
}
_PERIOD_LABELS = {3: "Q1", 6: "interim", 9: "Q3", 12: "annual"}

# Unlike yfinance's single-frame earnings_dates, akshare's earnings table is
# one paginated all-market request PER report period, so the history window is
# capped at four quarters to keep tool latency bounded (~3s per period).
_HISTORY_QUARTERS = 4
_ACTIONS_LOOKBACK_YEARS = 3


def _quarter_end(d: date) -> date:
    """Last day of the calendar quarter containing ``d``."""
    m = ((d.month + 2) // 3) * 3
    return date(d.year, m, calendar.monthrange(d.year, m)[1])


def _shift_quarters(d: date, k: int) -> date:
    """Quarter-end ``k`` quarters after the one containing ``d``."""
    end = _quarter_end(d)
    total = end.year * 12 + (end.month - 1) + 3 * k
    y, m0 = divmod(total, 12)
    m = m0 + 1
    return date(y, m, calendar.monthrange(y, m)[1])


def _disclosure_deadline(period_end: date) -> date:
    """Statutory latest disclosure date for the report period ending ``period_end``."""
    month, day, next_year = _DEADLINES[period_end.month]
    return date(period_end.year + (1 if next_year else 0), month, day)


def _period_label(period_end: date) -> str:
    return f"{period_end.year} {_PERIOD_LABELS[period_end.month]} report (period {period_end.isoformat()})"


def _yi(value) -> float | None:
    """Yuan -> 100M yuan (亿) for compact LLM-facing numbers, or None."""
    number = _to_float(value)
    return None if number is None else number / 1e8


def get_earnings_calendar(ticker: str, curr_date: str) -> str:
    """Upcoming periodic-report disclosure date for an A-share ticker.

    Queries eastmoney's disclosure-schedule table (``stock_yysj_em``) for the
    report periods whose disclosure season is still open as of ``curr_date``
    and reports the earliest scheduled disclosure after that date, falling
    back to the statutory deadline when the schedule is not published yet.

    Args:
        ticker: Canonical ticker, e.g. "600519.SS" / "000001.SZ".
        curr_date: Analysis date in yyyy-mm-dd format (point-in-time anchor).

    Returns:
        A formatted markdown string for the agent; never raises for data
        states (schedule unpublished / no record), only for real failures.
    """
    canonical, code = _a_share_code(ticker)

    today = date.fromisoformat(get_current_date())
    as_of = datetime.strptime(curr_date, "%Y-%m-%d").date()

    # Schedules have no vintage: only a live run may see today's table.
    if as_of < today:
        return (
            f"# Earnings schedule for {canonical}\n"
            f"# Point-in-time as of: {curr_date}\n\n"
            f"Schedule is withheld for this date. The disclosure-schedule table "
            f"has no historical vintage — schedules change after publication and "
            f"the table records no publication timestamp — so serving today's "
            f"table into a {curr_date} analysis would leak post-decision "
            f"information. Settled report results up to {curr_date} are "
            f"available from the earnings-history tool."
        )

    # Report periods whose disclosure season is still open as of as_of: every
    # quarter-end P with statutory deadline(P) >= as_of — a disclosure of P can
    # still land on a future date even after P has passed (a company may simply
    # not have filed yet). Scanning the previous, current, and next quarter-end
    # covers the whole open set; three seasons coexist only in the Jan-Apr
    # annual+Q1 overlap.
    nxt = _quarter_end(as_of)
    current = nxt if nxt == as_of else _shift_quarters(as_of, -1)
    periods = []
    for p in (_shift_quarters(current, -1), current, nxt):
        if _disclosure_deadline(p) >= as_of and p not in periods:
            periods.append(p)

    upcoming = None            # earliest (scheduled, period_end, row) after as_of
    overdue = None             # (scheduled, period_end) due on/before as_of, undisclosed
    missing_period = None      # earliest period whose schedule table was unavailable

    for period_end in periods:
        period = period_end.strftime("%Y%m%d")
        try:
            table = ak.stock_yysj_em(symbol="沪深A股", date=period)
        except TypeError:
            # akshare crashes on eastmoney's empty payload when the schedule
            # for this period is not published yet — a data state, not an error.
            logger.info("No disclosure schedule published yet for period %s", period)
            if missing_period is None:
                missing_period = period_end
            continue
        if table is None or table.empty:
            if missing_period is None:
                missing_period = period_end
            continue

        rows = table[
            table["股票代码"].astype(str).str.split(".").str[0].str.zfill(6) == code
        ]
        if rows.empty:
            continue
        row = rows.iloc[0]
        scheduled = _to_date(row.get("首次预约时间"))
        actual = _to_date(row.get("实际披露时间"))

        if actual is not None and actual <= as_of:
            continue  # settled before the analysis date — not an upcoming event
        if scheduled is not None and scheduled > as_of:
            if upcoming is None or scheduled < upcoming[0]:
                upcoming = (scheduled, period_end, row)
        elif overdue is None:
            overdue = (scheduled, period_end)

    if upcoming is not None:
        scheduled, period_end, row = upcoming
        lines = [
            f"## {canonical} upcoming report disclosure (as of {curr_date}, akshare)",
            f"- Report: {_period_label(period_end)}",
            f"- Scheduled disclosure: {scheduled.isoformat()} "
            f"(in {(scheduled - as_of).days} days)",
        ]
        changes = [
            d.isoformat()
            for d in (
                _to_date(row.get("一次变更日期")),
                _to_date(row.get("二次变更日期")),
                _to_date(row.get("三次变更日期")),
            )
            if d is not None
        ]
        if changes:
            lines.append(f"- Rescheduled on: {', '.join(changes)}")
        lines.append(f"- Statutory deadline: {_disclosure_deadline(period_end).isoformat()}")
        return "\n".join(lines) + "\n"

    if overdue is not None:
        scheduled, period_end = overdue
        shown = scheduled.isoformat() if scheduled is not None else "an unknown date"
        return (
            f"## {canonical} report disclosure (as of {curr_date}, akshare)\n"
            f"- Report: {_period_label(period_end)}\n"
            f"- Scheduled disclosure: {shown} — already due as of {curr_date} but "
            f"no actual disclosure recorded yet; treat the report as overdue.\n"
        )

    if missing_period is not None:
        # Fall back to the statutory deadline of the EARLIEST period whose
        # schedule is not out yet — that period is the next possible event,
        # even if a later period's table happened to be fetchable.
        deadline = _disclosure_deadline(missing_period)
        return (
            f"## {canonical} upcoming report disclosure (as of {curr_date}, akshare)\n"
            f"- Report: {_period_label(missing_period)}\n"
            f"- No published schedule yet; by exchange rules disclosure is due by "
            f"{deadline.isoformat()} (in {(deadline - as_of).days} days).\n"
        )

    return (
        f"No disclosure-schedule record found for {canonical} (schedules cover "
        f"SZ/SH A-shares; the ticker may be Beijing-exchange listed, delisted, "
        f"or simply unscheduled). Do not invent dates."
    )


def get_earnings_history(ticker: str, curr_date: str) -> str:
    """Settled quarterly reports (up to four) for an A-share ticker.

    Pulls eastmoney's all-market earnings table (``stock_yjbb_em``) for each of
    the last report periods and keeps the ticker's row when it had been publicly
    announced by ``curr_date`` — so a historical run only ever sees what was
    public then (the announcement-date filter is the look-ahead guard here,
    which is why history, unlike schedules, needs no withholding).

    Args:
        ticker: Canonical ticker, e.g. "600519.SS" / "000001.SZ".
        curr_date: Analysis date in yyyy-mm-dd format (point-in-time anchor).

    Returns:
        A markdown table (newest first) plus a one-line YoY summary; an
        informative sentinel when nothing was settled by ``curr_date``.
    """
    canonical, code = _a_share_code(ticker)
    as_of = datetime.strptime(curr_date, "%Y-%m-%d").date()

    # Last _HISTORY_QUARTERS report periods ending on/before as_of, newest first.
    latest = _quarter_end(as_of)
    if latest > as_of:
        latest = _shift_quarters(as_of, -1)
    periods = [_shift_quarters(latest, -k) for k in range(_HISTORY_QUARTERS)]

    history = []
    for period_end in periods:
        period = period_end.strftime("%Y%m%d")
        try:
            table = ak.stock_yjbb_em(date=period)
        except TypeError:
            # Same empty-payload quirk as the schedule table (period too old,
            # or eastmoney serving nothing for it) — a data state, skip it.
            logger.info("No earnings table available for period %s", period)
            continue
        if table is None or table.empty:
            continue
        match = table[
            table["股票代码"].astype(str).str.split(".").str[0].str.zfill(6) == code
        ]
        if match.empty:
            continue  # newly listed back then, or not in this period's table
        row = match.iloc[0]
        announced = _to_date(row.get("最新公告日期"))
        if announced is None or announced > as_of:
            # The row exists today but was not public at as_of — serving it
            # would leak post-decision information. Skip, don't round down.
            continue
        history.append({
            "period": period_end.isoformat(),
            "eps": _to_float(row.get("每股收益")),
            "rev_yi": _yi(row.get("营业总收入-营业总收入")),
            "rev_yoy": _to_float(row.get("营业总收入-同比增长")),
            "np_yi": _yi(row.get("净利润-净利润")),
            "np_yoy": _to_float(row.get("净利润-同比增长")),
            "announced": announced.isoformat(),
        })

    if not history:
        return (
            f"No settled earnings history found for {canonical} on or before "
            f"{curr_date} (the company may be newly listed). Do not invent figures."
        )

    lines = [
        f"## {canonical} earnings history (as of {curr_date}, akshare, "
        f"{len(history)} settled quarters, newest first)",
        "| Period | EPS | Revenue (100M CNY) | Rev YoY | Net profit (100M CNY) | NP YoY | Announced |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in history:
        lines.append(
            f"| {r['period']} | {_fmt(r['eps'])} | {_fmt(r['rev_yi'])} | "
            f"{_fmt(r['rev_yoy'], suffix='%')} | {_fmt(r['np_yi'])} | "
            f"{_fmt(r['np_yoy'], suffix='%')} | {r['announced']} |"
        )
    np_values = [v for v in (r["np_yoy"] for r in history) if v is not None]
    positive = sum(1 for v in np_values if v > 0)
    lines.append(f"Net profit YoY positive in {positive}/{len(history)} settled quarters.")
    return "\n".join(lines) + "\n"


def get_corporate_actions(
    ticker: str,
    curr_date: str,
    lookback_years: int | None = None,
) -> str:
    """Dividend / share-conversion records for an A-share ticker.

    One call to eastmoney's per-stock plan table (``stock_fhps_detail_em``),
    trimmed to plans publicly announced by ``curr_date`` (a plan announced
    after the analysis date is future information and is withheld) within the
    lookback window.

    Args:
        ticker: Canonical ticker, e.g. "600519.SS" / "000001.SZ".
        curr_date: Analysis date in yyyy-mm-dd format (point-in-time anchor).
        lookback_years: Years of history to keep; None for the default.

    Returns:
        A markdown table (newest announcement first) or an informative
        sentinel; zero-dividend stretches are normal and stated plainly.
    """
    canonical, code = _a_share_code(ticker)
    as_of = datetime.strptime(curr_date, "%Y-%m-%d").date()
    if lookback_years is None:
        lookback_years = _ACTIONS_LOOKBACK_YEARS
    # Calendar-year granularity keeps the Feb-29 arithmetic trivial.
    cutoff = date(as_of.year - int(lookback_years), 1, 1)

    try:
        table = ak.stock_fhps_detail_em(symbol=code)
    except TypeError:
        table = None
    if table is None or table.empty:
        return (
            f"No corporate-action data available for {canonical}. "
            f"Do not invent records."
        )

    kept = []
    withheld = 0
    for _, row in table.iterrows():
        # A plan is only knowable at as_of once announced; when the proposal
        # date is missing fall back to the row's last-update date, and drop
        # rows that carry neither (they cannot be dated -> conservative).
        announced = _to_date(row.get("预案公告日")) or _to_date(row.get("最新公告日期"))
        if announced is None or announced > as_of:
            withheld += 1
            continue
        report_period = _to_date(row.get("报告期"))
        if report_period is not None and report_period < cutoff:
            continue
        raw_desc = row.get("现金分红-现金分红比例描述")
        raw_progress = row.get("方案进度")
        desc = raw_desc.strip() if isinstance(raw_desc, str) else ""
        progress = raw_progress.strip() if isinstance(raw_progress, str) else ""
        if not desc or progress == "不分配":
            continue  # nothing to act on; an all-empty table hits the sentinel
        kept.append((
            announced,
            report_period,
            desc,
            progress,
            _to_date(row.get("股权登记日")),
            _to_date(row.get("除权除息日")),
        ))
    kept.sort(key=lambda item: item[0], reverse=True)

    if not kept:
        return (
            f"No corporate actions (cash dividends / share conversions) announced "
            f"for {canonical} in the last {lookback_years} years up to {curr_date}. "
            f"Zero-dividend stretches are common; do not invent records."
        )

    lines = [
        f"## {canonical} corporate actions (as of {curr_date}, akshare, "
        f"last {lookback_years} years, newest first)",
        "| Announced | Report period | Plan | Progress | Record date | Ex-date |",
        "|---|---|---|---|---|---|",
    ]
    for announced, report_period, desc, progress, record, ex_date in kept:
        lines.append(
            f"| {announced.isoformat()} | {report_period.isoformat() if report_period else 'n/a'} "
            f"| {desc} | {progress} | "
            f"{record.isoformat() if record else '-'} | {ex_date.isoformat() if ex_date else '-'} |"
        )
    if withheld:
        lines.append(
            f"({withheld} record{'s' if withheld != 1 else ''} dated after "
            f"{curr_date} withheld as future information.)"
        )
    return "\n".join(lines) + "\n"
