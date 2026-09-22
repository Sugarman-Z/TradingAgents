"""Shared conventions for the akshare vendor modules.

Every akshare-backed vendor repeats the same three moves: translate the
framework's canonical ticker to the bare 6-digit code eastmoney/sina expect,
coerce spreadsheet cells defensively (akshare returns str / NaT /
datetime.date depending on interface and version), and treat akshare's
crash-on-empty payloads as data states rather than vendor errors. Keeping
those moves here means a new vendor module starts from tested primitives
instead of copying them (akshare_events was the first client; akshare_news
the second).
"""

import pandas as pd

from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol


def a_share_code(ticker: str) -> tuple[str, str]:
    """Canonical ticker + bare 6-digit akshare code, or raise NoMarketDataError.

    Non-A-share tickers (e.g. "AAPL") raise so the routing layer decides —
    for a configured chain this means the next vendor gets its turn, and if
    none can serve the symbol the caller receives the NO_DATA_AVAILABLE
    sentinel instead of a vendor-specific guess.
    """
    canonical = normalize_symbol(ticker)
    code = canonical.split(".")[0]
    if not (code.isdigit() and len(code) == 6):
        raise NoMarketDataError(
            ticker, canonical,
            detail="the akshare vendors serve 6-digit A-share codes only",
        )
    return canonical, code


def to_date(value):
    """Best-effort conversion of an akshare cell to a date; None if absent."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.to_datetime(value).date()
    except (TypeError, ValueError):
        return None


def to_float(value):
    """Best-effort conversion of an akshare cell to a float; None if absent."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, ndigits: int = 2, suffix: str = "") -> str:
    """Format an optional number for a markdown report; 'n/a' when absent."""
    number = to_float(value)
    return "n/a" if number is None else f"{number:.{ndigits}f}{suffix}"
