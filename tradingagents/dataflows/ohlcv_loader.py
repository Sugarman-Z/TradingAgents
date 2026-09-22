"""Frame-level OHLCV dispatch across the configured core_stock_apis chain.

``route_to_vendor`` serves the tool layer's string outputs; some consumers
need the DataFrame itself — the verification snapshot
(market_data_validator) and the reflection layer's realized-return lookup
(trading_graph._fetch_returns). This dispatcher walks the SAME configured
vendor chain with the SAME fallthrough semantics as route_to_vendor:
``NoMarketDataError`` moves on to the next vendor (that vendor genuinely has
no data for the symbol), real errors are remembered and surface only when no
vendor could serve the frame — a broken primary stays visible instead of
being hidden behind a fallback's verdict (#989).

Every branch guarantees the same contract: canonical columns
(Date/Open/High/Low/Close/Volume), ascending dates, and nothing after
``curr_date`` (point-in-time).
"""

import logging

import pandas as pd

from .akshare_market import load_ohlcv as load_akshare_ohlcv
from .errors import NoMarketDataError
from .interface import get_vendor
from .local_vendor import load_local_ohlcv
from .stockstats_utils import load_ohlcv

logger = logging.getLogger(__name__)


def _local_frame(symbol: str, curr_date: str) -> pd.DataFrame:
    """Local files under the dispatcher's PIT contract.

    load_local_ohlcv returns full history (its own consumers re-trim), but
    the dispatcher itself promises nothing after ``curr_date`` — applying it
    here keeps every branch of the chain interchangeable.
    """
    frame = load_local_ohlcv(symbol)
    cutoff = pd.Timestamp(curr_date)
    return frame[frame["Date"] <= cutoff]


def load_configured_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV frame from the first core_stock_apis vendor that has the symbol.

    An unconfigured chain ("default") keeps the legacy behavior of the
    yfinance loader. Vendors without a frame interface (alpha_vantage) are
    skipped with a warning rather than silently dropping the request.
    """
    configured = get_vendor("core_stock_apis", "get_stock_data")
    chain = [v.strip() for v in configured.split(",") if v.strip()]
    if not chain or chain == ["default"]:
        return load_ohlcv(symbol, curr_date)

    last_no_data: NoMarketDataError | None = None
    first_error: Exception | None = None
    for vendor in chain:
        try:
            if vendor == "local":
                return _local_frame(symbol, curr_date)
            if vendor == "akshare":
                return load_akshare_ohlcv(symbol, curr_date)
            if vendor == "yfinance":
                return load_ohlcv(symbol, curr_date)
            logger.warning(
                "Vendor %r has no frame loader; skipped for '%s' frame request.",
                vendor, symbol,
            )
            continue
        except NoMarketDataError as e:
            last_no_data = e
        except Exception as e:
            logger.warning("Frame loader %r failed for '%s': %s", vendor, symbol, e)
            if first_error is None:
                first_error = e

    if last_no_data is not None:
        raise last_no_data
    if first_error is not None:
        raise first_error
    raise RuntimeError(
        f"No frame-capable vendor in core_stock_apis chain {chain!r} for '{symbol}'"
    )
