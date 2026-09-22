"""akshare-based A-share per-stock news fetching.

Serves ``get_news`` for the ``news_data`` category via eastmoney's per-stock
news feed (keyless, domestic-direct). Shares the akshare conventions with
``akshare_events`` via ``akshare_common``: canonical-to-bare code gate,
defensive cell coercion, and crash-on-empty payloads treated as data states
while real failures propagate to the routing layer.

Data shape (akshare 1.18.94, live-probed): exactly the 10 newest items, no
pagination and no date parameters — the window filter is applied client-side
on the ``发布时间`` column, so the effective lookback is "the last 10 items",
which for liquid names spans days and for quiet names weeks. The ceiling is
documented in the report rather than hidden.
"""

import logging
from datetime import datetime

import akshare as ak
import pandas as pd

from .akshare_common import a_share_code, to_date
from .date_window import in_window

logger = logging.getLogger(__name__)


def _news_time(value) -> datetime | None:
    """Parse a publish-timestamp cell to a datetime; None if absent/unparseable."""
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts.to_pydatetime()


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Retrieve recent news for an A-share ticker from eastmoney.

    Args:
        ticker: Canonical ticker, e.g. "600519.SS" / "000001.SZ".
        start_date: Window start in yyyy-mm-dd format.
        end_date: Window end in yyyy-mm-dd format.

    Returns:
        Formatted markdown string; a plain no-news line when eastmoney has no
        items at all or none inside the window. Real failures propagate.
    """
    canonical, code = a_share_code(ticker)
    try:
        table = ak.stock_news_em(symbol=code)
    except (TypeError, KeyError) as exc:
        # akshare crashes on eastmoney's empty payload with TypeError and on a
        # zero-item result with KeyError('code') — both data states (no
        # coverage for this symbol), not vendor errors.
        logger.info("No news payload for %s (%s)", canonical, type(exc).__name__)
        table = None
    if table is None or table.empty:
        return f"No news found for {canonical} on eastmoney (no items returned)"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    news_str = ""
    kept = 0
    for _, row in table.iterrows():
        published = _news_time(row.get("发布时间"))
        # Same look-ahead-safe window rule as every other dated source: an
        # item outside [start, end+1day) is dropped, an undated item only
        # survives a window that reaches the present.
        if not in_window(published, start_dt, end_dt):
            continue
        title = str(row.get("新闻标题") or "").strip() or "(no title)"
        source = str(row.get("文章来源") or "").strip() or "unknown"
        content = str(row.get("新闻内容") or "").strip()
        link = str(row.get("新闻链接") or "").strip()
        stamp = published.strftime("%Y-%m-%d %H:%M") if published is not None else "undated"
        news_str += f"### {title} (source: {source}, published: {stamp})\n"
        if content:
            news_str += f"{content}\n"
        if link:
            news_str += f"Link: {link}\n"
        news_str += "\n"
        kept += 1

    if kept == 0:
        return f"No news found for {canonical} between {start_date} and {end_date}"

    body = (
        f"## {canonical} News, from {start_date} to {end_date} "
        f"(eastmoney, newest {kept} of {len(table)} items):\n\n{news_str}"
    )
    return body
