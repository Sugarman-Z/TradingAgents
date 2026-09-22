"""akshare-based A-share fundamentals (sina statements + ratio indicators).

Serves the four ``fundamental_data`` tools. The three statements come from
``ak.stock_financial_report_sina`` — one call per statement returns the FULL
history with per-row announcement dates, trimmed here to the line items a
trading analyst actually reads (the raw tables carry 147/83/71 columns).
``get_fundamentals`` uses ``ak.stock_financial_analysis_indicator`` for
per-period ratios; that interface has no announcement date, so publication is
derived by joining periods against the income statement's announcement dates.

Point-in-time rule (shared with akshare_events/akshare_news): a row survives
only when its announcement date is on/before the analysis date. Caveat, stated
in every report: sina serves the LATEST revision of each period, so a
restatement re-dates the row and a historical run loses pre-restatement
visibility rather than leaking the future — fail safe by construction.

Conventions (code gate, cell coercion, crash-on-empty as data states) live in
``akshare_common``.
"""

import logging
from datetime import date, datetime

import akshare as ak

from .akshare_common import a_share_code, fmt, to_date, to_float
from .errors import NoMarketDataError
from .utils import get_current_date

logger = logging.getLogger(__name__)

# Statements are one full-history call each (~1-2s); eight reported periods
# bound the report without losing the recent trend.
_MAX_ROWS = 8

_BALANCE_SHEET_COLUMNS = [
    "货币资金", "应收账款", "存货", "流动资产合计", "资产总计",
    "短期借款", "应付账款", "流动负债合计", "负债合计",
    "未分配利润", "归属于母公司股东权益合计",
]
_INCOME_STATEMENT_COLUMNS = [
    "营业总收入", "营业成本", "销售费用", "管理费用", "研发费用", "财务费用",
    "营业利润", "利润总额", "净利润", "归属于母公司所有者的净利润", "基本每股收益",
]
_CASHFLOW_COLUMNS = [
    "经营活动产生的现金流量净额", "投资活动产生的现金流量净额",
    "筹资活动产生的现金流量净额", "现金及现金等价物净增加额",
    "期末现金及现金等价物余额",
]
_INDICATOR_COLUMNS = [
    "摊薄每股收益(元)", "每股净资产_调整前(元)", "每股经营性现金流(元)",
    "销售毛利率(%)", "净资产收益率(%)", "主营业务收入增长率(%)",
    "净利润增长率(%)", "总资产周转率(次)", "流动比率", "速动比率",
    "应收账款周转天数(天)",
]

# Statement cells are raw yuan; they render as 100M CNY (亿) for compact,
# arithmetic-safe LLM consumption — except per-share items, which stay in yuan.
_PER_SHARE_COLUMNS = {"基本每股收益", "稀释每股收益"}


def _fmt_cell(value, column: str) -> str:
    """Statement cell: yuan -> 100M CNY (亿); per-share items stay in yuan."""
    number = to_float(value)
    if number is None:
        return "n/a"
    if column in _PER_SHARE_COLUMNS:
        return f"{number:.2f}"
    return f"{number / 1e8:.2f}亿"


def _as_of(curr_date: str | None) -> date:
    """Analysis date, defaulting to today when the caller omitted it."""
    if curr_date:
        return datetime.strptime(curr_date, "%Y-%m-%d").date()
    return date.fromisoformat(get_current_date())


def _sina_stock(canonical: str, code: str) -> str:
    """Map a canonical ticker to sina's exchange-prefixed form (sh600519)."""
    if canonical.endswith(".SS"):
        return f"sh{code}"
    if canonical.endswith(".SZ"):
        return f"sz{code}"
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "2", "3")):
        return f"sz{code}"
    raise NoMarketDataError(
        canonical, canonical,
        detail="sina fundamentals cover SH/SZ listings only (Beijing-exchange codes unsupported)",
    )


def _period_key(value) -> str:
    """Normalize a period cell ('20260630', '2026-06-30', Timestamp) to YYYYMMDD."""
    if value is None:
        return ""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else digits


def _iso_period(period: str) -> str:
    return f"{period[:4]}-{period[4:6]}-{period[6:]}" if len(period) == 8 else (period or "n/a")


def _fetch_statement(stock: str, sina_symbol: str):
    """One full-history statement table, or None for empty-payload data states."""
    try:
        return ak.stock_financial_report_sina(stock=stock, symbol=sina_symbol)
    except TypeError:
        logger.info("No %s payload from sina for %s", sina_symbol, stock)
        return None


def _statement_report(
    ticker: str,
    curr_date: str | None,
    freq: str | None,
    sina_symbol: str,
    label: str,
    columns: list[str],
    title: str,
) -> str:
    """Shared pipeline: fetch -> point-in-time -> frequency -> trim -> render."""
    canonical, code = a_share_code(ticker)
    stock = _sina_stock(canonical, code)
    as_of = _as_of(curr_date)
    frequency = (freq or "quarterly").strip().lower()

    table = _fetch_statement(stock, sina_symbol)
    if table is None or table.empty:
        return f"No {label} data available for {canonical} on sina"

    rows = []
    for _, row in table.iterrows():
        period = _period_key(row.get("报告日"))
        announced = to_date(row.get("公告日期"))
        if len(period) != 8 or announced is None or announced > as_of:
            # Undatable rows can't be proven public; future-dated rows are the
            # future. Both are dropped rather than rounded toward leakage.
            continue
        if frequency == "annual" and not period.endswith("1231"):
            continue
        rows.append((period, announced, row))
    rows.sort(key=lambda item: item[0], reverse=True)
    rows = rows[:_MAX_ROWS]
    if not rows:
        return (
            f"No {label} data publicly announced for {canonical} on or before "
            f"{as_of.isoformat()} — the newest reports are not yet public at "
            f"this analysis date. Do not fabricate figures."
        )

    lines = [
        f"## {canonical} {title} (as of {as_of.isoformat()}, sina, {frequency}, newest first)",
        "# Figures are the latest published revision of each period; a period is",
        "# shown only when its announcement date is on/before the analysis date.",
        "# Monetary figures in 100M CNY (亿); per-share items in CNY.",
        "| Period | " + " | ".join(columns) + " | Announced |",
        "|---|" + "---|" * (len(columns) + 1),
    ]
    for period, announced, row in rows:
        cells = " | ".join(_fmt_cell(row.get(column), column) for column in columns)
        lines.append(f"| {_iso_period(period)} | {cells} | {announced.isoformat()} |")
    return "\n".join(lines) + "\n"


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Retrieve balance sheet data for an A-share ticker from sina.

    Args:
        ticker: Canonical ticker, e.g. "600519.SS" / "000001.SZ".
        freq: "quarterly" (all report periods) or "annual" (Dec-31 periods).
        curr_date: Analysis date in yyyy-mm-dd format; None means today.

    Returns:
        Markdown table of key line items, newest first, point-in-time filtered.
    """
    return _statement_report(
        ticker, curr_date, freq, "资产负债表", "balance sheet",
        _BALANCE_SHEET_COLUMNS, "Balance Sheet",
    )


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Retrieve income statement data for an A-share ticker from sina."""
    return _statement_report(
        ticker, curr_date, freq, "利润表", "income statement",
        _INCOME_STATEMENT_COLUMNS, "Income Statement",
    )


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Retrieve cash flow statement data for an A-share ticker from sina."""
    return _statement_report(
        ticker, curr_date, freq, "现金流量表", "cash flow statement",
        _CASHFLOW_COLUMNS, "Cash Flow Statement",
    )


def _fetch_indicator(code: str, as_of: date):
    """Ratio-indicator table from sina; None for empty-payload data states."""
    try:
        return ak.stock_financial_analysis_indicator(symbol=code, start_year=str(as_of.year - 3))
    except TypeError:
        logger.info("No financial indicator payload from sina for %s", code)
        return None


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Retrieve key per-period financial ratios for an A-share ticker from sina.

    The indicator interface carries no announcement date, so publication is
    derived by joining periods against the income statement's announcement
    dates (statements share one disclosure schedule): a ratio row is kept only
    when its period was publicly announced on/before the analysis date.

    Args:
        ticker: Canonical ticker, e.g. "600519.SS" / "000001.SZ".
        curr_date: Analysis date in yyyy-mm-dd format; None means today.

    Returns:
        Markdown table of key ratios, newest first, point-in-time filtered.
    """
    canonical, code = a_share_code(ticker)
    stock = _sina_stock(canonical, code)
    as_of = _as_of(curr_date)

    # Publication oracle: announcement date per period from the income statement.
    income = _fetch_statement(stock, "利润表")
    announced: dict[str, date] = {}
    if income is not None and not income.empty:
        for _, row in income.iterrows():
            period = _period_key(row.get("报告日"))
            announced_on = to_date(row.get("公告日期"))
            if len(period) == 8 and announced_on is not None and announced_on <= as_of:
                announced[period] = announced_on
    if not announced:
        return (
            f"No publicly announced report periods found for {canonical} on or "
            f"before {as_of.isoformat()} — do not fabricate figures."
        )

    table = _fetch_indicator(code, as_of)
    if table is None or table.empty:
        return f"No financial-ratio indicator data available for {canonical} on sina"

    rows = []
    for _, row in table.iterrows():
        period = _period_key(row.get("日期"))
        if period in announced:
            rows.append((period, announced[period], row))
    rows.sort(key=lambda item: item[0], reverse=True)
    rows = rows[:_MAX_ROWS]
    if not rows:
        return (
            f"No indicator rows fall in periods publicly announced for "
            f"{canonical} on or before {as_of.isoformat()}."
        )

    lines = [
        f"## {canonical} Key Ratios (as of {as_of.isoformat()}, sina, newest first)",
        "# Periods kept only when the matching report was publicly announced by",
        "# the analysis date (announcement dates from the income statement).",
        "| Period | " + " | ".join(_INDICATOR_COLUMNS) + " | Announced |",
        "|---|" + "---|" * (len(_INDICATOR_COLUMNS) + 1),
    ]
    for period, announced_on, row in rows:
        cells = " | ".join(fmt(row.get(column)) for column in _INDICATOR_COLUMNS)
        lines.append(f"| {_iso_period(period)} | {cells} | {announced_on.isoformat()} |")
    return "\n".join(lines) + "\n"
