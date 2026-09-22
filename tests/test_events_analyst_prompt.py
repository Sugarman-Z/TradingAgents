"""Guard the events analyst prompt against tool-signature drift (#1116 pattern).

The prompt advertises the three event tools by signature; if a tool's args
drift (or the prompt is reworded to a stale signature) the LLM starts
hallucinating calls, so the wording is locked here. Also locks the report
channel name: the analyst must write ``events_report`` — the state field, the
analyst node spec, and the five downstream consumers all agree on it.
"""
import inspect

import pytest

import tradingagents.agents.analysts.events_analyst as ea
from tradingagents.agents.utils.event_data_tools import (
    get_corporate_actions,
    get_earnings_calendar,
    get_earnings_history,
)


@pytest.mark.unit
def test_event_tools_take_curr_date():
    for tool in (get_earnings_calendar, get_earnings_history, get_corporate_actions):
        assert "curr_date" in tool.args.keys()


@pytest.mark.unit
def test_events_prompt_matches_tool_signatures():
    src = inspect.getsource(ea)
    assert "get_earnings_calendar(ticker, curr_date)" in src
    assert "get_earnings_history(ticker, curr_date)" in src
    assert "get_corporate_actions(ticker, curr_date, lookback_years)" in src


@pytest.mark.unit
def test_events_analyst_writes_events_report_channel():
    src = inspect.getsource(ea)
    assert '"events_report"' in src
    assert '"event_report"' not in src  # the singular form was a silent-drop bug
