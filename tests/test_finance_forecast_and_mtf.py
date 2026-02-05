from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from commands.finance import _mc_forecast, _parse_mtf_spec


def test_finance_command_slash_arg_count_within_limit() -> None:
    src = Path("commands/finance.py").read_text(encoding="utf-8")
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "finance":
            names = [a.arg for a in node.args.args]
            assert len(names) - 2 <= 25
            return
    raise AssertionError("finance command not found")


def test_parse_mtf_spec_normalizes_month_shorthand_and_reports_note() -> None:
    frames, notes = _parse_mtf_spec("1d,6m,1y")
    assert frames[1][1] == "6mo"
    assert any("normalized mtf token" in n for n in notes)


def test_mc_forecast_has_reproducible_seed_and_fan() -> None:
    idx = pd.date_range("2023-01-01", periods=260, freq="B")
    close = pd.Series(
        100
        * np.exp(
            np.cumsum(
                np.random.default_rng(11).normal(0.0002, 0.012, size=len(idx))
            )
        ),
        index=idx,
    )

    r1 = _mc_forecast(
        close,
        ticker="7203.T",
        horizon_days=20,
        paths=800,
        model="ewma_bootstrap",
    )
    r2 = _mc_forecast(
        close,
        ticker="7203.T",
        horizon_days=20,
        paths=800,
        model="ewma_bootstrap",
    )

    assert r1["ok"] and r2["ok"]
    assert r1["seed"] == r2["seed"]
    assert r1["fan"]["p50"] == r2["fan"]["p50"]
    assert len(r1["fan"]["x"]) == 20


def test_parse_mtf_spec_normalizes_single_month_token() -> None:
    frames, notes = _parse_mtf_spec("1m")
    assert frames[0][1] == "1mo"
    assert any("normalized mtf token" in n for n in notes)


def test_compare_defaults_to_jp_etf_benchmark_in_source() -> None:
    src = Path("commands/finance.py").read_text(encoding="utf-8")
    assert "1306.T" in src
    assert "998405.T" not in src
