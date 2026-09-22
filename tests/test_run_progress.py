"""Report hygiene + debug-run node progress helpers.

- strip_final_proposal_prefix: the analyst boilerplate line ("FINAL TRANSACTION
  PROPOSAL: **X**") is team protocol, not report content — saved reports must
  not start with it.
- _progress_node: debug runs print a start/finish line per real node so a live
  run reads as a step list instead of minutes of unexplained silence.
"""

import pytest

from tradingagents.agents.utils.agent_utils import strip_final_proposal_prefix
from tradingagents.graph.setup import _progress_node, NODE_DESCRIPTIONS


@pytest.mark.unit
def test_strip_removes_leading_proposal_line():
    text = "FINAL TRANSACTION PROPOSAL: **HOLD**\n\n---\n\n# 贵州茅台基本面深度研究报告"
    stripped = strip_final_proposal_prefix(text)
    assert not stripped.startswith("FINAL TRANSACTION PROPOSAL")
    assert stripped.startswith("---")   # the report body follows the protocol line
    assert "贵州茅台基本面深度研究报告" in stripped


@pytest.mark.unit
def test_strip_handles_buy_sell_and_slash_forms():
    for head in (
        "FINAL TRANSACTION PROPOSAL: **SELL**\n正文",
        "FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**\n正文",
        "final transaction proposal: **buy**\n正文",
    ):
        assert strip_final_proposal_prefix(head).startswith("正文")


@pytest.mark.unit
def test_strip_keeps_mid_document_mentions():
    text = "# 报告\n\n正文提到 FINAL TRANSACTION PROPOSAL: **SELL** 属于正文内容"
    assert "FINAL TRANSACTION PROPOSAL" in strip_final_proposal_prefix(text)


@pytest.mark.unit
def test_strip_empty_and_none_passthrough():
    assert strip_final_proposal_prefix("") == ""
    assert strip_final_proposal_prefix(None) is None


@pytest.mark.unit
def test_progress_node_prints_start_done_and_returns_result(capsys):
    def fn(state):
        return {"x": state["v"] * 2}

    wrapped = _progress_node("Trader", fn)
    out = wrapped({"v": 21})
    captured = capsys.readouterr().out
    assert "[node] Trader — 交易员决策" in captured
    assert "[done] Trader" in captured
    assert out == {"x": 42}


@pytest.mark.unit
def test_node_descriptions_cover_all_flow_nodes():
    expected = {
        "Market Analyst", "Sentiment Analyst", "News Analyst", "Fundamentals Analyst",
        "Event Analyst", "Bull Researcher", "Bear Researcher", "Research Manager",
        "Trader", "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst",
        "Portfolio Manager",
    }
    assert expected.issubset(NODE_DESCRIPTIONS.keys())
