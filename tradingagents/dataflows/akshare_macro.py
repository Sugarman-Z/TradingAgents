"""akshare-based China macro vendor (NBS/PBOC series via eastmoney & chinamoney).

Serves ``get_macro_indicators`` for the ``macro_data`` category as the domestic
replacement for FRED: CPI, PPI, money supply, GDP, LPR, Shibor, urban
unemployment — all keyless, domestic-direct, and verified fresh through the
current month.

Point-in-time rule, macro edition: statistical series carry a RELEASE LAG, not
an announcement-date column. A reference month is only served once its release
date (month + lag) is on/before the analysis date — CPI/PPI ~M+1 day 9, money
supply ~M+1 day 15, unemployment ~M+1 day 20, GDP ~quarter-end +30d. LPR
fixings are public on their own date; Shibor is daily. Without these guards a
historical run would read a statistic months before any human could have.

Monthly labels arrive as Chinese strings ('2026年08月份') and GDP as
'2026年第1季度' (cumulative rows contain '-' and are dropped to avoid double
counting) — parsing is digit-based and defensive throughout.
"""

import logging
from calendar import monthrange
from datetime import date, datetime, timedelta

import akshare as ak
from dateutil.relativedelta import relativedelta

from .akshare_common import fmt, to_float
from .utils import get_current_date

logger = logging.getLogger(__name__)

# FRED-precedent default window and row cap (macro tables are monthly/daily;
# a daily Shibor year would otherwise flood the agent's context).
DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40

# Monthly series -> (months added to the reference month, extra days). The
# values sit at the conservative end of each official release calendar.
_MONTHLY_RELEASE_LAG = {
    "cpi": (1, 9),
    "ppi": (1, 9),
    "money_supply": (1, 15),
    "unemployment": (1, 20),
}
_GDP_RELEASE_LAG_DAYS = 30  # quarterly GDP lands mid-month after quarter end

_UNEMPLOYMENT_ITEM = "全国城镇调查失业率"


def _as_of(curr_date: str | None) -> date:
    """Analysis date, defaulting to today when the caller omitted it."""
    if curr_date:
        return datetime.strptime(curr_date, "%Y-%m-%d").date()
    return date.fromisoformat(get_current_date())


def _digits(value) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _month_key(label) -> str:
    """'2026年08月份' / '202608' / date -> '202608'; '' when unusable."""
    digits = _digits(label)
    return digits[:6] if len(digits) >= 6 else ""


def _month_label(key: str) -> str:
    return f"{key[:4]}-{key[4:]}"


def _month_release(key: str, lag: tuple[int, int]) -> date:
    """Release date of a monthly statistic: reference month + (months, days)."""
    year, month = int(key[:4]), int(key[4:6])
    return date(year, month, 1) + relativedelta(months=+lag[0], days=+lag[1])


def _month_end(key: str) -> date:
    year, month = int(key[:4]), int(key[4:6])
    return date(year, month, monthrange(year, month)[1])


def _find_col(df, *needles: str, exclude: str | None = None):
    """First column whose name contains every needle (and no `exclude`), or None.

    Column names drift between fetches on some eastmoney endpoints (e.g. CPI
    alternates between 全国-当月同比增长 and 全国-同比增长), so value columns
    are located by substring, never by exact name.
    """
    for column in getattr(df, "columns", []):
        if exclude and exclude in column:
            continue
        if all(needle in column for needle in needles):
            return column
    return None


def _render(title: str, note: str, columns: list[str], rows: list[tuple]) -> str:
    lines = [f"## {title}", f"# {note}", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows[:MAX_ROWS]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _macro_cpi(as_of: date, look_back_days: int) -> str:
    df = ak.macro_china_cpi()
    lag = _MONTHLY_RELEASE_LAG["cpi"]
    window_start = as_of - timedelta(days=look_back_days)
    level_col = _find_col(df, "全国", "当月", exclude="同比")
    yoy_col = _find_col(df, "全国", "同比增长")
    mom_col = _find_col(df, "全国", "环比增长")
    rows = []
    for _, row in df.iterrows():
        key = _month_key(row.get("月份"))
        if len(key) != 6:
            continue
        if _month_release(key, lag) > as_of or _month_end(key) < window_start:
            continue
        rows.append((
            _month_label(key),
            fmt(row.get(level_col)),
            fmt(row.get(yoy_col), suffix="%"),
            fmt(row.get(mom_col), suffix="%"),
        ))
    rows.sort(reverse=True)
    if not rows:
        return "DATA_UNAVAILABLE: no CPI observation public at this analysis date."
    return _render(
        "中国宏观: CPI 居民消费价格 (NBS via akshare)",
        "同比/环比为官方百分比; 参考月仅在发布日(次月9日)之后展示。",
        ["月份", "CPI当月(指数)", "同比(%)", "环比(%)"],
        rows,
    )


def _macro_ppi(as_of: date, look_back_days: int) -> str:
    df = ak.macro_china_ppi()
    lag = _MONTHLY_RELEASE_LAG["ppi"]
    window_start = as_of - timedelta(days=look_back_days)
    level_col = _find_col(df, "当月", exclude="同比")
    yoy_col = _find_col(df, "同比增长")
    rows = []
    for _, row in df.iterrows():
        key = _month_key(row.get("月份"))
        if len(key) != 6:
            continue
        if _month_release(key, lag) > as_of or _month_end(key) < window_start:
            continue
        rows.append((
            _month_label(key),
            fmt(row.get(level_col)),
            fmt(row.get(yoy_col), suffix="%"),
        ))
    rows.sort(reverse=True)
    if not rows:
        return "DATA_UNAVAILABLE: no PPI observation public at this analysis date."
    return _render(
        "中国宏观: PPI 工业生产者出厂价格 (NBS via akshare)",
        "当月为指数水平; 参考月仅在发布日(次月9日)之后展示。",
        ["月份", "PPI当月(指数)", "同比(%)"],
        rows,
    )


def _macro_money_supply(as_of: date, look_back_days: int) -> str:
    df = ak.macro_china_money_supply()
    lag = _MONTHLY_RELEASE_LAG["money_supply"]
    window_start = as_of - timedelta(days=look_back_days)
    m2_col = _find_col(df, "M2", "同比增长")
    m1_col = _find_col(df, "(M1)", "同比增长")
    m0_col = _find_col(df, "(M0)", "同比增长")
    rows = []
    for _, row in df.iterrows():
        key = _month_key(row.get("月份"))
        if len(key) != 6:
            continue
        if _month_release(key, lag) > as_of or _month_end(key) < window_start:
            continue
        rows.append((
            _month_label(key),
            fmt(row.get(m2_col), suffix="%") if m2_col else "n/a",
            fmt(row.get(m1_col), suffix="%") if m1_col else "n/a",
            fmt(row.get(m0_col), suffix="%") if m0_col else "n/a",
        ))
    rows.sort(reverse=True)
    if not rows:
        return "DATA_UNAVAILABLE: no money-supply observation public at this analysis date."
    return _render(
        "中国宏观: 货币供应 M2/M1/M0 同比 (NBS via akshare)",
        "参考月仅在发布日(次月15日)之后展示。",
        ["月份", "M2同比(%)", "M1同比(%)", "M0同比(%)"],
        rows,
    )


def _macro_gdp(as_of: date, look_back_days: int) -> str:
    df = ak.macro_china_gdp()
    growth_col = _find_col(df, "生产总值", "同比增长")
    window_start = as_of - timedelta(days=look_back_days)
    rows = []
    for _, row in df.iterrows():
        label = str(row.get("季度") or "")
        if "-" in label:
            continue  # cumulative rows (第1-2季度/第1-4季度) duplicate quarters
        digits = _digits(label)
        if len(digits) < 5:
            continue
        year, quarter = int(digits[:4]), int(digits[4])
        quarter_end = date(year, quarter * 3, monthrange(year, quarter * 3)[1])
        if quarter_end + timedelta(days=_GDP_RELEASE_LAG_DAYS) > as_of:
            continue
        if quarter_end < window_start:
            continue
        rows.append((
            f"{year}Q{quarter}",
            fmt(row.get(growth_col), suffix="%") if growth_col else "n/a",
        ))
    rows.sort(reverse=True)
    if not rows:
        return "DATA_UNAVAILABLE: no GDP observation public at this analysis date."
    return _render(
        "中国宏观: GDP 当季同比 (NBS via akshare)",
        "仅单季值(累计行剔除); 参考季仅在季末+30天后展示。",
        ["季度", "GDP当季同比(%)"],
        rows,
    )


def _macro_lpr(as_of: date, look_back_days: int) -> str:
    df = ak.macro_china_lpr()
    window_start = as_of - timedelta(days=look_back_days)
    rows = []
    for _, row in df.iterrows():
        digits = _digits(row.get("TRADE_DATE"))
        if len(digits) != 8:
            continue
        fixing = datetime.strptime(digits, "%Y%m%d").date()
        if fixing > as_of or fixing < window_start:
            continue  # a fixing is public on its own announcement date (20th)
        rows.append((fixing.isoformat(), fmt(row.get("LPR1Y"), suffix="%"), fmt(row.get("LPR5Y"), suffix="%")))
    rows.sort(reverse=True)
    if not rows:
        return "DATA_UNAVAILABLE: no LPR fixing in this window."
    return _render(
        "中国宏观: LPR 贷款市场报价利率 (PBOC via akshare)",
        "每月20日报价(遇节假日顺延), 报价当日即为公开。",
        ["报价日", "1Y LPR", "5Y LPR"],
        rows,
    )


def _macro_shibor(as_of: date, look_back_days: int) -> str:
    df = ak.macro_china_shibor_all()
    tenors = ["O/N", "1W", "3M", "1Y"]
    columns = {tenor: _find_col(df, tenor, "定价") for tenor in tenors}
    window_start = as_of - timedelta(days=look_back_days)
    rows = []
    for _, row in df.iterrows():
        digits = _digits(row.get("日期"))
        if len(digits) != 8:
            continue
        day = datetime.strptime(digits, "%Y%m%d").date()
        if day > as_of or day < window_start:
            continue
        rows.append((day.isoformat(), *[fmt(row.get(columns[t])) for t in tenors]))
    rows.sort(reverse=True)
    if not rows:
        return "DATA_UNAVAILABLE: no Shibor fixing in this window."
    return _render(
        "中国宏观: Shibor 银行间同业拆放利率 (全国银行间同业拆借中心 via akshare)",
        "日频定价(%)。",
        ["日期", "O/N(%)", "1W(%)", "3M(%)", "1Y(%)"],
        rows,
    )


def _macro_unemployment(as_of: date, look_back_days: int) -> str:
    df = ak.macro_china_urban_unemployment()
    lag = _MONTHLY_RELEASE_LAG["unemployment"]
    window_start = as_of - timedelta(days=look_back_days)
    rows = []
    for _, row in df.iterrows():
        if str(row.get("item") or "").strip() != _UNEMPLOYMENT_ITEM:
            continue
        key = _month_key(row.get("date"))
        if len(key) != 6:
            continue
        if _month_release(key, lag) > as_of or _month_end(key) < window_start:
            continue
        rows.append((_month_label(key), fmt(row.get("value"), suffix="%")))
    rows.sort(reverse=True)
    if not rows:
        return "DATA_UNAVAILABLE: no unemployment observation public at this analysis date."
    return _render(
        "中国宏观: 城镇调查失业率 (NBS via akshare)",
        "参考月仅在发布日(次月20日)之后展示。",
        ["月份", "失业率(%)"],
        rows,
    )


_EXTRACTORS = {
    "cpi": _macro_cpi,
    "ppi": _macro_ppi,
    "money_supply": _macro_money_supply,
    "gdp": _macro_gdp,
    "lpr": _macro_lpr,
    "shibor": _macro_shibor,
    "unemployment": _macro_unemployment,
}

SUPPORTED_INDICATORS = sorted(_EXTRACTORS) + ["m2"]


def get_macro_indicators(
    indicator: str,
    curr_date: str | None = None,
    look_back_days: int | None = None,
) -> str:
    """Retrieve a China macro indicator time series from akshare (NBS/PBOC).

    Args:
        indicator: One of 'cpi', 'ppi', 'money_supply' (alias 'm2'), 'gdp',
            'lpr', 'shibor', 'unemployment'.
        curr_date: Analysis date in yyyy-mm-dd format; None means today.
        look_back_days: Trailing window length; None means 365 days.

    Returns:
        Markdown table (newest first) with release-lag point-in-time guards
        applied; an informative message for unsupported names or empty data.
    """
    key = (indicator or "").strip().lower()
    key = {"m2": "money_supply"}.get(key, key)
    extractor = _EXTRACTORS.get(key)
    if extractor is None:
        return (
            f"Unsupported macro indicator '{indicator}' for the akshare vendor. "
            f"Supported aliases: {', '.join(SUPPORTED_INDICATORS)}. Do not guess a series."
        )
    as_of = _as_of(curr_date)
    look_back = look_back_days or DEFAULT_LOOKBACK_DAYS
    try:
        return extractor(as_of, look_back)
    except TypeError:
        # akshare's crash-on-empty payload quirk — a data state, not an error.
        logger.info("No %s payload from akshare macro interfaces", key)
        return f"DATA_UNAVAILABLE: {key} series returned no data; do not fabricate values."
