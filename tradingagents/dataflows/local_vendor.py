"""Local-file data vendor for A-share daily OHLCV (back-adjusted / hfq).

Serves the pipeline entirely from files under ``local_data_dir`` (default
``<repo>/data/a_share``, override with ``TRADINGAGENTS_LOCAL_DATA_DIR``) so a
run needs no market-data network access — the constraint that motivated this
vendor is that Yahoo and the other US vendors are unreachable from a
mainland-China network.

Layout: either flat files per symbol (``600519.SS.csv``) or, preferentially
resolved, exchange subdirectories with bare-code files
(``SH/price_600519.csv``, ``SZ/price_000001.csv``, ``BJ/price_810011.csv``).
The exchange suffix also disambiguates the index namespace from
same-numbered stocks (000001.SS 上证指数 vs 000001.SZ 平安银行):

    SH/price_600519.csv    Kweichow Moutai (Shanghai)
    SZ/price_000001.csv    Ping An Bank (Shenzhen)
    SH/price_000001.csv    SSE Composite index (benchmark for .SS tickers)
    BJ/price_810011.csv    Beijing exchange

Columns (case-insensitive, order-free): a date column (``date``/``timetag``,
either ``YYYY-MM-DD`` or compact ``YYYYMMDD``), plus ``open, high, low,
close, volume`` (``volumn`` accepted) — one row per trading day, ascending,
BACK-adjusted (hfq) so historical rows never shift on a later
dividend/split: point-in-time safe, and stable under the per-symbol caching
philosophy of this data layer. Extra columns (``amount``, ``open_ineterst``)
are ignored. ``.parquet`` is tried before ``.csv`` in the flat layout
(parquet needs ``pyarrow``).

Methods the local dataset does not carry (fundamentals, news, insider
transactions) raise :class:`NoMarketDataError` explicitly instead of
returning empty data: the router turns that into the NO_DATA sentinel so
analysts report the gap honestly, and the configured vendor chain never
falls through to an online vendor that would answer with wrong-locale data
(the #988 "no silent fallback" principle).
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from .config import get_config
from .errors import NoMarketDataError
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

# Indicators the local vendor can compute, via stockstats, from the OHLCV
# series. Names must stay in lockstep with y_finance's indicator tool so the
# market analyst's prompts keep working unchanged.
_INDICATOR_PARAMS = {
    "close_50_sma": "50 SMA: medium-term trend direction and dynamic support/resistance.",
    "close_200_sma": "200 SMA: long-term trend benchmark for golden/death crosses.",
    "close_10_ema": "10 EMA: responsive short-term average, prone to noise.",
    "macd": "MACD: momentum via EMA differences; watch crossovers and divergence.",
    "macds": "MACD signal line; crossovers with MACD trigger trades.",
    "macdh": "MACD histogram: gap between MACD and its signal line.",
    "rsi": "RSI: momentum; 70/30 overbought/oversold thresholds.",
    "boll": "Bollinger middle band (20 SMA): dynamic benchmark.",
    "boll_ub": "Bollinger upper band (+2 SD): overbought / breakout zone.",
    "boll_lb": "Bollinger lower band (-2 SD): oversold / breakdown zone.",
    "atr": "ATR: average true range, a volatility measure for stop sizing.",
    "vwma": "VWMA: volume-weighted moving average; confirms trends.",
    "mfi": "MFI: money flow index; >80 overbought, <20 oversold.",
}

# Accepted (lowercased) header names per canonical column.
_DATE_ALIASES = ("date", "datetime", "timestamp", "timetag", "交易日")
_COLUMN_MAP = {
    "open": "Open", "开盘": "Open",
    "high": "High", "最高": "High",
    "low": "Low", "最低": "Low",
    "close": "Close", "收盘": "Close", "adj close": "Close", "adjclose": "Close",
    "volume": "Volume", "volumn": "Volume", "成交量": "Volume",
}

# Exchange-subdirectory layout: suffix -> subdirectory, and bare-code first
# digit -> subdirectory (6/5/9 Shanghai incl. funds and B-shares, 8/4 Beijing,
# everything else Shenzhen).
_SUFFIX_TO_DIR = {".SS": "SH", ".SZ": "SZ", ".BJ": "BJ"}
_EXCHANGE_BY_PREFIX = {"6": "SH", "5": "SH", "9": "SH", "8": "BJ", "4": "BJ"}


def _local_data_dir() -> str:
    """Configured local-data directory, defaulting to <repo>/data/a_share."""
    configured = get_config().get("local_data_dir")
    if configured:
        return configured
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(repo_root, "data", "a_share")


def _normalize_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Rename whatever header dialect the file uses to the canonical columns."""
    rename = {}
    date_src = None
    for col in df.columns:
        low = str(col).strip().lower()
        if low in _DATE_ALIASES and date_src is None:
            date_src = col
            rename[col] = "Date"
        elif low in _COLUMN_MAP:
            target = _COLUMN_MAP[low]
            if target == "Close" and "Close" in rename.values():
                continue  # keep the first close-like column (plain close wins)
            rename[col] = target

    if date_src is None:
        raise NoMarketDataError(symbol, symbol, "file has no date column (expected 'date')")

    df = df.rename(columns=rename)
    missing = {"Open", "High", "Low", "Close", "Volume"} - set(df.columns)
    if missing:
        raise NoMarketDataError(
            symbol, symbol,
            f"missing columns {sorted(missing)}; expected date,open,high,low,close,volume",
        )

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    # Compact integer dates (20240816) must not hit to_datetime's numeric
    # path — it would read them as epoch NANOSECONDS and land in 1970.
    if pd.api.types.is_numeric_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(
            df["Date"].astype("int64").astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), errors="coerce")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )


def _resolve_exchange(symbol: str) -> tuple[str | None, str | None]:
    """('600519.SS' -> ('600519', 'SH')); bare '600519' infers the exchange.

    Returns (None, None) for non-A-share-style symbols so the caller can
    report a clean no-data error instead of guessing.
    """
    compact = symbol.strip().upper()
    if compact.endswith((".SS", ".SZ", ".BJ")):
        code, suffix = compact.split(".", 1)
        return code, _SUFFIX_TO_DIR[f".{suffix}"]
    if compact.isdigit() and len(compact) == 6:
        return compact, _EXCHANGE_BY_PREFIX.get(compact[0], "SZ")
    return None, None


def load_local_ohlcv(symbol: str) -> pd.DataFrame:
    """Load and normalize the local file for ``symbol`` (e.g. ``600519.SS``).

    Candidate order: flat ``<safe>.parquet`` / ``<safe>.csv``, then the
    exchange-subdirectory layout ``<DIR>/price_<code>.csv``. Raises
    NoMarketDataError when no file exists or the file is unusable, so the
    router emits its single unambiguous no-data sentinel.
    """
    directory = _local_data_dir()
    safe = safe_ticker_component(symbol)
    code, exchange = _resolve_exchange(symbol)

    candidates = [os.path.join(directory, safe + ext) for ext in (".parquet", ".csv")]
    if code and exchange:
        candidates.append(os.path.join(directory, exchange, f"price_{code}.csv"))

    for path in candidates:
        if os.path.isfile(path):
            frame = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
            frame = _normalize_frame(frame, symbol)
            if frame.empty:
                raise NoMarketDataError(symbol, safe, f"file {path} has no usable rows")
            return frame

    if code and exchange:
        raise NoMarketDataError(symbol, safe, f"no local file for {code} under {os.path.join(directory, exchange)}")
    raise NoMarketDataError(symbol, safe, f"no local file under {directory}")


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Daily OHLCV (back-adjusted) CSV within [start_date, end_date]."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    df = load_local_ohlcv(symbol)
    window = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    if window.empty:
        raise NoMarketDataError(
            symbol, symbol, f"local file has no rows between {start_date} and {end_date}"
        )

    window = window.copy()
    for col in ("Open", "High", "Low", "Close"):
        window[col] = window[col].round(2)
    window["Date"] = window["Date"].dt.strftime("%Y-%m-%d")

    header = (
        f"# Stock data for {symbol} (local A-share, back-adjusted/hfq) "
        f"from {start_date} to {end_date}\n"
        f"# Total records: {len(window)}\n"
    )
    last = df["Date"].max()
    if last < end_dt:
        header += f"# NOTE: local file only covers up to {last:%Y-%m-%d}; end_date is beyond coverage\n"

    return header + "\n" + window.to_csv(index=False)


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    """One stockstats indicator over [curr_date - look_back_days, curr_date].

    Mirrors y_finance.get_stock_stats_indicators_window's output shape (one
    line per calendar day, N/A on non-trading days) so the analyst prompts
    see an identical channel regardless of the configured vendor. The
    indicator is computed on the full local history up to curr_date, so
    warm-up windows (e.g. MACD's 26-day EMA) are satisfied even for short
    look-backs.
    """
    if indicator not in _INDICATOR_PARAMS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: "
            f"{list(_INDICATOR_PARAMS.keys())}"
        )

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_dt - timedelta(days=look_back_days)

    df = load_local_ohlcv(symbol)
    df = df[df["Date"] <= curr_dt]  # point-in-time: nothing after the trade date
    if df.empty:
        raise NoMarketDataError(symbol, symbol, f"no local rows on/before {curr_date}")

    # Same bulk pattern as y_finance: compute once over the whole window.
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


# ---------------------------------------------------------------------------
# Explicitly unsupported: the local dataset is OHLCV-only. Each raises
# NoMarketDataError so the router returns its honest NO_DATA sentinel rather
# than silently falling through to an online vendor (see module docstring).
# ---------------------------------------------------------------------------

def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    raise NoMarketDataError(ticker, detail="local dataset carries OHLCV only (no fundamentals)")


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    raise NoMarketDataError(ticker, detail="local dataset carries OHLCV only (no balance sheet)")


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    raise NoMarketDataError(ticker, detail="local dataset carries OHLCV only (no cashflow)")


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    raise NoMarketDataError(ticker, detail="local dataset carries OHLCV only (no income statement)")


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    raise NoMarketDataError(ticker, detail="local dataset carries OHLCV only (no news)")


def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 50) -> str:
    raise NoMarketDataError("global", detail="local dataset carries OHLCV only (no news)")


def get_insider_transactions(symbol: str) -> str:
    raise NoMarketDataError(symbol, detail="local dataset carries OHLCV only (no insider data)")


def get_macro_indicators(*args, **kwargs) -> str:
    raise NoMarketDataError("macro", detail="local dataset carries OHLCV only (no macro data)")


def get_prediction_markets(*args, **kwargs) -> str:
    raise NoMarketDataError(
        "prediction", detail="local dataset carries OHLCV only (no prediction markets)"
    )
