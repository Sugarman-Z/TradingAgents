"""akshare-based A-share daily OHLCV (back-adjusted/hfq) market vendor.

Serves ``get_stock_data`` and ``get_indicators`` for the ``core_stock_apis`` /
``technical_indicators`` categories via eastmoney (keyless, domestic-direct),
so a run without pre-downloaded local files still has prices — the intended
chain is ``"local,akshare"``: local hfq files first, akshare as the fallback
(the #988 configured-chain principle; no vendor is contacted that the user
did not list).

Stock vs index namespaces (the collision local_vendor also guards): a bare
6-digit code is ambiguous — ``000001.SS`` is the SSE Composite index while
``000001.SZ`` is Ping An Bank — so the suffix decides: ``.SS`` + ``000xxx``
and ``.SZ`` + ``399xxx`` are index symbols served by ``stock_zh_index_daily_em``
(``sh000001`` / ``sz399001``); every other 6-digit code is a stock served by
``stock_zh_a_hist`` with ``adjust="hfq"`` to match the local files' back
adjustment (historical rows never shift on a later dividend, so cached frames
stay point-in-time safe). Index series have no adjustment concept and pass
through as eastmoney reports them.

Caching mirrors stockstats_utils.load_ohlcv (the yfinance loader): one file
per symbol per day under ``data_cache_dir`` covering [today-5y, today]
(cache superset, read-time trim), poisoned/empty caches are misses, and a
current-day request re-fetches on a 15-minute TTL (#1150) so today's bar
converges to the official close after publish. akshare's crash-on-empty
payloads (TypeError / KeyError / JSONDecodeError) are data states
(NoMarketDataError) like every other akshare vendor; network failures
propagate to the routing layer.
"""

import json
import logging
import os
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

from .akshare_common import a_share_code
from .config import get_config
from .errors import NoMarketDataError
from .local_vendor import _INDICATOR_PARAMS
from .stockstats_utils import (
    _assert_ohlcv_not_stale,
    _clean_dataframe,
    _fill_price_gaps,
    _needs_same_day_refresh,
)
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

# stock_zh_a_hist (eastmoney) columns -> canonical pipeline columns. Extra
# columns (成交额/振幅/涨跌幅/...) are dropped: the report channel is exactly
# Date,Open,High,Low,Close,Volume — the same dialect as the local and yfinance
# vendors, so downstream prompts cannot tell the vendors apart.
_STOCK_COLUMNS = {
    "日期": "Date", "开盘": "Open", "最高": "High", "最低": "Low",
    "收盘": "Close", "成交量": "Volume",
}
# stock_zh_index_daily_em returns lowercase English columns.
_INDEX_COLUMNS = {
    "date": "Date", "open": "Open", "high": "High",
    "low": "Low", "close": "Close", "volume": "Volume",
}

# The report channel is exactly these six columns — same dialect as the local
# and yfinance vendors, so downstream prompts cannot tell the vendors apart.
_CANONICAL_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

_HISTORY_YEARS = 5


def _index_em_symbol(canonical: str, code: str) -> str | None:
    """eastmoney index symbol for index-namespace tickers, else None.

    ``000001.SS`` -> ``sh000001`` (SSE Composite); ``399001.SZ`` -> ``sz399001``
    (SZSE Component). Everything else — including bare ``000001``, which by
    local_vendor's prefix rule is the Shenzhen stock — is a stock symbol.
    """
    if canonical.endswith(".SS") and code.startswith("000"):
        return f"sh{code}"
    if canonical.endswith(".SZ") and code.startswith("399"):
        return f"sz{code}"
    return None


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Daily OHLCV frame (hfq stocks / raw indexes), cached per symbol per day.

    Rows after ``curr_date`` are trimmed (point-in-time), the shared staleness
    guard rejects frames whose coverage ends far before the request, and the
    disk cache follows the yfinance loader's rules (superset window, poison =
    miss, same-day TTL).
    """
    canonical, code = a_share_code(symbol)
    index_symbol = _index_em_symbol(canonical, code)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date).normalize()
    today_date = pd.Timestamp.today().normalize()

    # Cache-superset window: fetch 5y to today once per symbol per day; every
    # request trims client-side, so a short window costs no extra HTTP.
    start_date = today_date - pd.DateOffset(years=_HISTORY_YEARS)
    kind = "index" if index_symbol else "hfq"
    safe = safe_ticker_component(canonical)
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe}-akshare-{kind}-{start_date:%Y-%m-%d}-{today_date:%Y-%m-%d}.csv",
    )

    # A cached file may be empty if a prior fetch failed; treat unusable
    # caches as misses instead of serving the poisoned file forever.
    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        if (
            not cached.empty
            and "Close" in cached.columns
            and not _needs_same_day_refresh(data_file, curr_date_dt, today_date)
        ):
            data = cached

    if data is None:
        try:
            if index_symbol:
                raw = ak.stock_zh_index_daily_em(symbol=index_symbol)
                frame = raw.rename(columns=_INDEX_COLUMNS)
            else:
                raw = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=today_date.strftime("%Y%m%d"),
                    adjust="hfq",
                )
                frame = raw.rename(columns=_STOCK_COLUMNS)
        except (TypeError, KeyError, json.JSONDecodeError) as exc:
            # akshare crashes on eastmoney's empty payload instead of
            # returning an empty frame — that is "no data for this symbol",
            # not a vendor failure (same convention as akshare_news/events).
            raise NoMarketDataError(
                symbol, canonical,
                f"akshare returned no payload ({type(exc).__name__}: {exc})",
            ) from exc
        if frame.empty:
            raise NoMarketDataError(symbol, canonical, "akshare returned no rows")
        missing = set(_CANONICAL_COLUMNS) - set(frame.columns)
        if missing:
            # A dialect change on akshare's side is a loud error, not no-data:
            # the routing layer should remember it, not answer "unavailable".
            # (Checked only on non-empty frames: an empty one has no columns
            # and is simply a no-data case, handled above.)
            raise RuntimeError(
                f"akshare column dialect changed for {canonical}: missing "
                f"{sorted(missing)} in {list(frame.columns)}"
            )
        frame = frame[_CANONICAL_COLUMNS]  # drop extras (股票代码/成交额/换手率/...)
        frame.to_csv(data_file, index=False, encoding="utf-8")
        data = frame

    data = _clean_dataframe(data)
    # Point-in-time trim first, then gap-fill so indicators compute on a
    # continuous series (same order as the yfinance loader).
    data = data[data["Date"] <= curr_date_dt]
    data = _fill_price_gaps(data)
    _assert_ohlcv_not_stale(data, curr_date, symbol, canonical)
    return data.reset_index(drop=True)


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Daily OHLCV (hfq stocks / raw indexes) CSV within [start_date, end_date]."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    df = load_ohlcv(symbol, end_date)
    window = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    if window.empty:
        raise NoMarketDataError(
            symbol, symbol, f"akshare has no rows between {start_date} and {end_date}"
        )

    window = window.copy()
    for col in ("Open", "High", "Low", "Close"):
        window[col] = window[col].round(2)
    window["Date"] = window["Date"].dt.strftime("%Y-%m-%d")

    canonical, code = a_share_code(symbol)
    source = (
        "eastmoney index daily, unadjusted"
        if _index_em_symbol(canonical, code)
        else "eastmoney daily, back-adjusted/hfq; volume in lots (手, 100 shares)"
    )
    header = (
        f"# Stock data for {symbol} (akshare {source}) "
        f"from {start_date} to {end_date}\n"
        f"# Total records: {len(window)}\n"
    )
    last = df["Date"].max()
    if last < end_dt:
        header += (
            f"# NOTE: series only covers up to {last:%Y-%m-%d}; "
            f"end_date is beyond coverage\n"
        )

    return header + "\n" + window.to_csv(index=False)


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    """One stockstats indicator over [curr_date - look_back_days, curr_date].

    Rendering is kept verbatim in lockstep with local_vendor.get_indicator
    (one line per calendar day, N/A on non-trading days) so the analyst
    prompts see an identical channel regardless of which vendor served.
    """
    if indicator not in _INDICATOR_PARAMS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: "
            f"{list(_INDICATOR_PARAMS.keys())}"
        )

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_dt - timedelta(days=look_back_days)

    df = load_ohlcv(symbol, curr_date)  # point-in-time: nothing after curr_date
    if df.empty:
        raise NoMarketDataError(symbol, symbol, f"no rows on/before {curr_date}")

    from stockstats import wrap

    data = wrap(df.copy())
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    data[indicator]  # accessing the column triggers stockstats to compute it

    values = {}
    for _, row in data.iterrows():
        value = row[indicator]
        values[row["Date"]] = "N/A" if pd.isna(value) else str(value)

    ind_string = ""
    day = curr_dt
    while day >= before:
        date_str = day.strftime("%Y-%m-%d")
        value = values.get(date_str, "N/A: Not a trading day (weekend or holiday)")
        ind_string += f"{date_str}: {value}\n"
        day -= timedelta(days=1)

    return (
        f"## {indicator} values from {before:%Y-%m-%d} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + _INDICATOR_PARAMS.get(indicator, "No description available.")
    )
