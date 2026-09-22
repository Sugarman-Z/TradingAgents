"""Offline unit tests for the akshare China-macro vendor.

Stubs the seven akshare macro handles through the module under test and
asserts what makes macro different from company-level data: a reference month
is only served AFTER its release date (release-lag point-in-time guard),
cumulative GDP rows are dropped, long-format unemployment is filtered to the
headline item, and column drift is absorbed by substring matching.
"""

import pandas as pd
import pytest

import tradingagents.dataflows.akshare_macro as anm

_CPI_COLS = ["月份", "全国-当月", "全国-当月同比增长", "全国-环比增长"]


def _df(columns, rows):
    return pd.DataFrame(list(rows), columns=list(columns))


def _install(monkeypatch, mapping):
    """Patch ak.macro_china_<name>; mapping keys are the <name> suffixes."""
    for name, table in mapping.items():
        monkeypatch.setattr(anm.ak, f"macro_china_{name}", lambda t=table: t)


@pytest.mark.unit
def test_cpi_serves_only_public_months_and_sorts_newest_first(monkeypatch):
    # as_of 2026-09-16: the August CPI (released ~Sep 10) is public; the
    # September CPI (released ~Oct 10) must be withheld by the lag guard.
    _install(monkeypatch, {"cpi": _df(_CPI_COLS, [
        ("2026年09月份", 100.5, 0.4, 0.3),
        ("2026年08月份", 100.8, 0.8, 0.2),
        ("2026年07月份", 100.6, 0.5, 0.1),
    ])})
    out = anm.get_macro_indicators("cpi", "2026-09-16", 365)
    assert "2026-08" in out
    assert "2026-09" not in out        # release ~Oct 10 -> withheld
    assert "2026-07" in out
    assert out.index("2026-08") < out.index("2026-07")  # newest first
    assert "次月9日" in out             # the guard is documented in the report


@pytest.mark.unit
def test_gdp_drops_cumulative_rows_and_guards_release(monkeypatch):
    _install(monkeypatch, {"gdp": _df(["季度", "国内生产总值-同比增长"], [
        ("2026年第1-2季度", 5.2),    # cumulative -> dropped
        ("2026年第1季度", 5.4),      # quarter-end 2026-03-31 + 30d <= as_of
        ("2026年第3季度", 5.0),      # quarter-end 2026-09-30 + 30d > as_of -> withheld
    ])})
    out = anm.get_macro_indicators("gdp", "2026-09-16", 365*3)
    assert "2026Q1" in out
    assert "1-2" not in out             # cumulative row never rendered
    assert "2026Q3" not in out          # release guard withholds
    assert "累计行剔除" in out


@pytest.mark.unit
def test_lpr_fixings_within_window_newest_first(monkeypatch):
    from datetime import date

    _install(monkeypatch, {"lpr": _df(["TRADE_DATE", "LPR1Y", "LPR5Y"], [
        (date(2026, 7, 20), 3.0, 3.5),
        (date(2026, 8, 20), 3.0, 3.5),
    ])})
    out = anm.get_macro_indicators("lpr", "2026-09-16", 365)
    assert "2026-08-20" in out and "2026-07-20" in out
    assert out.index("2026-08-20") < out.index("2026-07-20")  # newest first
    assert "3.00%" in out


@pytest.mark.unit
def test_unemployment_filters_headline_item_and_guards_release(monkeypatch):
    _install(monkeypatch, {"urban_unemployment": _df(["date", "item", "value"], [
        ("202608", "全国城镇25—59岁劳动力失业率 ", 4.4),   # non-headline item -> dropped
        ("202608", "全国城镇调查失业率 ", 5.2),             # release ~Sep 20 > as_of -> withheld
        ("202607", "全国城镇调查失业率 ", 5.2),             # release ~Aug 20 <= as_of -> kept
    ])})
    out = anm.get_macro_indicators("unemployment", "2026-09-16", 365)
    assert "2026-07" in out
    assert "2026-08" not in out        # headline item matched, but lag guard
    assert "次月20日" in out


@pytest.mark.unit
def test_money_supply_column_drift_via_substring_match(monkeypatch):
    _install(monkeypatch, {"money_supply": _df(
        ["月份", "货币和准货币(M2)-同比增长", "货币(M1)-同比增长", "流通中货币(M0)-同比增长"],
        [("2026年08月份", 7.0, 2.3, 9.0)],
    )})
    out = anm.get_macro_indicators("m2", "2026-09-16", 365)   # 'm2' alias resolves
    assert "M2同比(%)" in out
    assert "7.00%" in out
    assert "2.30%" in out


@pytest.mark.unit
def test_unknown_indicator_lists_supported_names(monkeypatch):
    out = anm.get_macro_indicators("core_pce", "2026-09-16")
    assert "Unsupported macro indicator 'core_pce'" in out
    assert "cpi" in out and "shibor" in out    # supported list shown for self-correction
