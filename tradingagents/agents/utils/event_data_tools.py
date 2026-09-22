from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_earnings_calendar(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve the next periodic-report disclosure schedule for an A-share ticker:
    the scheduled disclosure date with days-until, any reschedule history, and
    the statutory deadline when the schedule is not published yet. Withheld for
    runs dated in the past (schedules carry no historical vintage).
    Uses the configured event_data vendor.
    Args:
        ticker (str): Ticker symbol of the company (e.g. "600519.SS")
        curr_date (str): Current date in yyyy-mm-dd format
    Returns:
        str: A report of the upcoming earnings disclosure schedule
    """
    return route_to_vendor("get_earnings_calendar", ticker, curr_date)


@tool
def get_earnings_history(
    ticker: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve the last settled quarterly reports (up to four) for an A-share
    ticker: EPS, revenue and net profit with YoY growth, filtered to what was
    publicly announced by curr_date.
    Uses the configured event_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date in yyyy-mm-dd format
    Returns:
        str: A report of settled quarterly earnings with a YoY summary
    """
    return route_to_vendor("get_earnings_history", ticker, curr_date)


@tool
def get_corporate_actions(
    ticker: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_years: Annotated[int | None, "Years of history; omit for 3"] = None,
) -> str:
    """
    Retrieve dividend and share-conversion (corporate action) records for an
    A-share ticker: plan terms, progress, record and ex-dividend dates,
    filtered to what was publicly announced by curr_date.
    Uses the configured event_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date in yyyy-mm-dd format
        lookback_years (int): Years of history to keep; omit for the default
    Returns:
        str: A report of corporate action records, newest first
    """
    return route_to_vendor("get_corporate_actions", ticker, curr_date, lookback_years)
