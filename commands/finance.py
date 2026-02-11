from __future__ import annotations
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportCallIssue=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false

import asyncio
import hashlib
import inspect
import json
import logging
import math
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from statistics import NormalDist

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction, safe_reply, tag_error_embed, tag_error_text

log = logging.getLogger(__name__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import Search

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SYMBOLS_FILE = DATA_DIR / "finance_symbols.json"
WATCH_FILE = DATA_DIR / "finance_watchlist.json"

TICKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-^=]{0,31}$")
CODE_ONLY_RE = re.compile(r"^\d{4,5}$")
JPX_CODE_RE = re.compile(r"^[0-9][0-9A-Z]{3}$")
INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
FINANCE_ACTIONS = (
    "summary",
    "quote",
    "chart",
    "candle",
    "compare",
    "mtf",
    "ta",
    "forecast",
    "news",
    "symbols",
    "search",
    "lookup",
    "screener_local",
    "watch_add",
    "watch_remove",
    "watch_list",
    "data",
)
FINANCE_ACTION_SET = set(FINANCE_ACTIONS)
FINANCE_KV_KEYS = (
    "action",
    "period",
    "interval",
    "query",
    "symbol",
    "ticker",
    "kind",
    "limit",
    "region",
    "section",
    "horizon_days",
    "paths",
    "min_mcap",
    "min_market_cap",
    "max_pe",
    "threshold_pct",
    "check_every_s",
    "channel_id",
    "auto_adjust",
    "remove_all",
    "max_rows",
    "preset",
    "theme",
    "events",
    "ui",
    "compare",
    "bench",
    "base",
    "mtf",
    "lookup_kind",
    "forecast_model",
)
FINANCE_ACTION_CHOICES = [
    app_commands.Choice(name=action, value=action) for action in FINANCE_ACTIONS
]

DATA_SECTIONS: dict[str, str] = {
    "fast_info": "t.fast_info (dict)",
    "info": "t.info (dict, may fail depending on Yahoo changes)",
    "history_metadata": "t.history_metadata (dict)",
    "calendar": "t.calendar (dict)",
    "news": "t.news (list of dict)",
    "options": "t.options (list of expiries)",
    "dividends": "t.dividends (Series)",
    "splits": "t.splits (Series)",
    "actions": "t.actions (DataFrame)",
    "financials": "t.financials (DataFrame)",
    "quarterly_financials": "t.quarterly_financials (DataFrame)",
    "income_stmt": "t.income_stmt (DataFrame)",
    "quarterly_income_stmt": "t.quarterly_income_stmt (DataFrame)",
    "balance_sheet": "t.balance_sheet (DataFrame)",
    "quarterly_balance_sheet": "t.quarterly_balance_sheet (DataFrame)",
    "cashflow": "t.cashflow (DataFrame)",
    "quarterly_cashflow": "t.quarterly_cashflow (DataFrame)",
    "earnings": "t.earnings (DataFrame)",
    "quarterly_earnings": "t.quarterly_earnings (DataFrame)",
    "earnings_dates": "t.earnings_dates (DataFrame)",
    "recommendations": "t.recommendations (DataFrame)",
    "upgrades_downgrades": "t.upgrades_downgrades (DataFrame)",
    "major_holders": "t.major_holders (DataFrame)",
    "institutional_holders": "t.institutional_holders (DataFrame)",
    "mutualfund_holders": "t.mutualfund_holders (DataFrame)",
    "sustainability": "t.sustainability (DataFrame)",
    "sec_filings": "t.sec_filings (dict-ish)",
    "shares": "t.shares (DataFrame or dict)",
    "funds_data": "t.funds_data (FundsData or None)",
}


def _setup_jp_font() -> None:
    candidates = ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Yu Gothic"]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return


_setup_jp_font()


@dataclass(frozen=True)
class ListedSymbol:
    name: str
    ticker: str


class SymbolRegistry:
    def __init__(self, symbols: list[ListedSymbol]) -> None:
        self._symbols = symbols
        self._by_name: dict[str, ListedSymbol] = {s.name: s for s in symbols}

    @classmethod
    def load(cls) -> "SymbolRegistry":
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not SYMBOLS_FILE.exists():
            starter = {
                "symbols": [{"name": "Toyota Motor Corporation", "ticker": "7203.T"}]
            }
            SYMBOLS_FILE.write_text(
                json.dumps(starter, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        raw = json.loads(SYMBOLS_FILE.read_text(encoding="utf-8"))
        symbols: list[ListedSymbol] = []
        for row in raw.get("symbols", []):
            name = str(row.get("name", "")).strip()
            ticker = str(row.get("ticker", "")).strip()
            if not name or not ticker:
                continue
            symbols.append(ListedSymbol(name=name, ticker=ticker))
        return cls(symbols)

    @property
    def symbols(self) -> list[ListedSymbol]:
        return list(self._symbols)

    def find_by_name(self, name: str) -> ListedSymbol | None:
        return self._by_name.get(name)

    def resolve_strict(self, raw: str) -> tuple[str, str]:
        """
        Rules:
        - If raw looks like a ticker, pass it through as-is.
        - Otherwise, only allow an exact match on registry names.
        - Ambiguous inputs like `/finance Toyota` must error.
        """
        s = (raw or "").strip()
        if not s:
            raise ValueError("symbol_empty")

        if CODE_ONLY_RE.match(s) or JPX_CODE_RE.match(s):
            cands = []
            for sym in self._symbols:
                if sym.ticker.startswith(s + "."):
                    cands.append(f"{sym.ticker}:{sym.name}")
            if cands:
                raise LookupError("ticker_missing_suffix:" + " ".join(cands))
            raise LookupError("ticker_missing_suffix")

        if TICKER_RE.match(s):
            return s, s

        hit = self._by_name.get(s)
        if hit:
            return hit.ticker, hit.name

        import difflib

        names = [x.name for x in self._symbols]
        suggestions = difflib.get_close_matches(s, names, n=5, cutoff=0.35)
        if suggestions:
            raise LookupError("symbol_unknown:" + ", ".join(suggestions))
        raise LookupError("symbol_unknown")


class TTLCache:
    def __init__(self) -> None:
        self._d: dict[Any, tuple[float, Any]] = {}

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def get(self, key: Any) -> Any | None:
        item = self._d.get(key)
        if not item:
            return None
        exp, val = item
        if exp < self._now():
            self._d.pop(key, None)
            return None
        return val

    def set(self, key: Any, val: Any, ttl_s: float) -> None:
        self._d[key] = (self._now() + float(ttl_s), val)


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _fmt_num(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "N/A"
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return f"{x:,.{nd}f}" if nd >= 0 else f"{x:,}"
        return str(x)
    except Exception:
        return "N/A"


def _fmt_pct(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "N/A"
        return f"{float(x) * 100:.{nd}f}%"
    except Exception:
        return "N/A"


def _fmt_signed_num(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "N/A"
        v = float(x)
        return f"{v:+,.{nd}f}"
    except Exception:
        return "N/A"


def _period_days_approx(period: str) -> int | None:
    p = (period or "").strip().lower()
    if not p:
        return None
    if p.endswith("d"):
        return int(p[:-1])
    if p.endswith("wk"):
        return int(p[:-2]) * 7
    if p.endswith("mo"):
        return int(p[:-2]) * 30
    if p.endswith("y"):
        return int(p[:-1]) * 365
    if p == "ytd":
        now = datetime.now(timezone.utc)
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return max(1, (now - start).days)
    if p == "max":
        return 10_000
    return None


def _validate_period_interval(period: str, interval: str) -> None:
    """
    yfinance docs: Intraday data cannot extend last 60 days.
    """
    iv = (interval or "1d").strip()
    if iv in INTRADAY_INTERVALS:
        days = _period_days_approx(period or "")
        if days is not None and days > 60:
            raise ValueError("intraday_limit_60d")


async def _to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


class YFinanceClient:
    """
    yfinance uses synchronous I/O, so wrap it with to_thread.
    .info is fragile, so prefer fast_info and treat info as best-effort.
    """

    def __init__(self) -> None:
        self._cache = TTLCache()

    async def get_quote(self, ticker: str) -> dict[str, Any]:
        key = ("quote", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        def _sync():
            t = yf.Ticker(ticker)
            out: dict[str, Any] = {"ticker": ticker}
            try:
                out["fast_info"] = dict(t.fast_info or {})
            except Exception:
                out["fast_info"] = {}
            try:
                out["info"] = dict(t.info or {})
            except Exception as e:
                out["info"] = {"_error": repr(e)}
            return out

        data = await _to_thread(_sync)
        self._cache.set(key, data, ttl_s=15.0)
        return data

    async def get_history(
        self,
        ticker: str,
        *,
        period: str = "1mo",
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        _validate_period_interval(period, interval)

        key = ("hist", ticker, period, interval, auto_adjust)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        def _sync():
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=auto_adjust)
            if df is None:
                return pd.DataFrame()
            return df

        df = await _to_thread(_sync)
        ttl = 60.0 if interval not in INTRADAY_INTERVALS else 20.0
        self._cache.set(key, df, ttl_s=ttl)
        return df

    async def get_section(self, ticker: str, section: str) -> Any:
        section = section.strip()
        if section not in DATA_SECTIONS:
            raise ValueError("bad_section")

        def _sync():
            t = yf.Ticker(ticker)
            return getattr(t, section)

        return await _to_thread(_sync)


def _build_quote_view(data: dict[str, Any]) -> dict[str, Any]:
    fi = data.get("fast_info") or {}
    info = data.get("info") or {}

    price = _pick(fi, "last_price", "lastPrice", default=_pick(info, "regularMarketPrice"))
    prev = _pick(
        fi,
        "previous_close",
        "previousClose",
        default=_pick(info, "regularMarketPreviousClose"),
    )
    open_ = _pick(fi, "open", default=_pick(info, "regularMarketOpen"))
    day_low = _pick(fi, "day_low", default=_pick(info, "regularMarketDayLow"))
    day_high = _pick(fi, "day_high", default=_pick(info, "regularMarketDayHigh"))
    ylow = _pick(fi, "year_low", default=_pick(info, "fiftyTwoWeekLow"))
    yhigh = _pick(fi, "year_high", default=_pick(info, "fiftyTwoWeekHigh"))
    vol = _pick(fi, "volume", default=_pick(info, "regularMarketVolume"))
    mcap = _pick(fi, "market_cap", default=_pick(info, "marketCap"))
    cur = _pick(fi, "currency", default=_pick(info, "currency", default=""))

    change = None
    change_pct = None
    try:
        if price is not None and prev not in (None, 0):
            change = float(price) - float(prev)
            change_pct = change / float(prev)
    except Exception:
        pass

    return {
        "price": price,
        "prev_close": prev,
        "open": open_,
        "day_low": day_low,
        "day_high": day_high,
        "52w_low": ylow,
        "52w_high": yhigh,
        "volume": vol,
        "market_cap": mcap,
        "currency": cur,
        "change": change,
        "change_pct": change_pct,
    }


def _display_name_from_quote(ticker: str, display: str, data: dict[str, Any]) -> str:
    if display != ticker:
        return display
    info = data.get("info") or {}
    name = _pick(info, "shortName", "longName", "name", "displayName")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return display


def _plot_line_chart(df: pd.DataFrame, title: str) -> BytesIO:
    if df is None or df.empty:
        raise ValueError("empty_history")

    idx = df.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    except Exception:
        pass

    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

    fig = plt.figure(figsize=(9, 4), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(idx, close, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _json_compact(obj: Any, max_len: int = 800) -> str:
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(s) <= max_len:
        return s
    truncated = s[: max_len - 3] + "..."
    return json.dumps(
        {"truncated": True, "preview": truncated},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _to_csv_bytes(df: Any, max_rows: int = 50) -> bytes:
    if df is None:
        return b""
    if isinstance(df, pd.Series):
        df = df.to_frame(name="value")
    if isinstance(df, pd.DataFrame):
        df2 = df.head(max_rows)
        return df2.to_csv(index=True).encode("utf-8")
    return str(df).encode("utf-8")


def _compact_dict(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in source:
            out[key] = source.get(key)
    return out


def _summarize_finance_payload(payload: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    action = (meta.get("action") or payload.get("action") or "").strip().lower()
    summary: dict[str, Any] = {
        "action": action or None,
        "ticker": payload.get("ticker"),
        "display": payload.get("display"),
    }
    if payload.get("symbol_input"):
        summary["symbol_input"] = payload.get("symbol_input")
    if payload.get("yahoo_symbol"):
        summary["yahoo_symbol"] = payload.get("yahoo_symbol")
    if payload.get("asof_jst"):
        summary["asof_jst"] = payload.get("asof_jst")
    if payload.get("params"):
        summary["params"] = payload.get("params")

    if action in {"summary", "quote"}:
        quote = payload.get("quote") or {}
        if isinstance(quote, dict):
            summary["quote"] = _compact_dict(
                quote,
                [
                    "price",
                    "prev_close",
                    "open",
                    "day_low",
                    "day_high",
                    "52w_low",
                    "52w_high",
                    "volume",
                    "market_cap",
                    "currency",
                    "change",
                    "change_pct",
                ],
            )
    elif action == "ta":
        latest = payload.get("latest")
        if isinstance(latest, dict):
            summary["latest_indicators"] = _compact_dict(
                latest,
                [
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "bb_mid",
                    "bb_upper",
                    "bb_lower",
                ],
            )
    elif action == "mtf":
        summary["replay"] = payload.get("replay")
        notes = payload.get("notes")
        if isinstance(notes, list):
            summary["notes"] = notes[:5]
    elif action == "compare":
        summary["bench"] = payload.get("bench")
        summary["base"] = payload.get("base")
        summary["outperformance"] = payload.get("outperformance")
        summary["replay"] = payload.get("replay")
    elif action == "forecast":
        forecast = payload.get("forecast")
        if isinstance(forecast, dict):
            summary["forecast"] = _compact_dict(
                forecast,
                ["s0", "mu", "sigma", "horizon_days", "paths", "end_quantiles", "var95"],
            )
    elif action == "news":
        items = payload.get("items")
        if isinstance(items, list):
            summary["news_items"] = items[:5]
    elif action == "symbols":
        summary["count"] = payload.get("count")
        symbols = payload.get("symbols")
        if isinstance(symbols, list):
            summary["symbols_preview"] = symbols[:10]
    elif action == "search":
        summary["query"] = payload.get("query")
        summary["region"] = payload.get("region")
        results = payload.get("results")
        if isinstance(results, list):
            summary["results_preview"] = results[:10]
            summary["result_count"] = len(results)
    elif action == "lookup":
        summary["query"] = payload.get("query")
        summary["kind"] = payload.get("kind")
        summary["rows"] = payload.get("rows")
    elif action == "screener_local":
        summary["min_market_cap"] = payload.get("min_market_cap")
        summary["max_pe"] = payload.get("max_pe")
        summary["rows"] = payload.get("rows")
    elif action == "data":
        summary["section"] = payload.get("section")
    elif action == "watch_add":
        summary["entry"] = payload.get("entry")
    elif action == "watch_remove":
        summary["remove"] = _compact_dict(
            payload,
            ["ticker", "display", "channel_id", "remove_all"],
        )
    elif action == "watch_list":
        entries = payload.get("entries")
        if isinstance(entries, list):
            summary["entries_preview"] = entries[:10]
            summary["entries_count"] = len(entries)

    return summary


async def _send_with_json(
    ctx: commands.Context,
    *,
    payload: dict[str, Any],
    meta: dict[str, Any],
    embed: discord.Embed | None = None,
    files: list[discord.File] | None = None,
    ephemeral: bool = False,
    error: bool = False,
) -> None:
    out_files = list(files or [])
    content = None
    if ctx is not None:
        summary = _summarize_finance_payload(payload, meta)
        compact_payload = _json_compact(payload, max_len=4000)
        record = {
            "meta": meta,
            "summary": summary,
            "payload_compact": compact_payload,
            "filename": None,
        }
        history = getattr(ctx, "finance_results", None)
        if not isinstance(history, list):
            history = []
        history.append(record)
        setattr(ctx, "finance_results", history)
    await safe_reply(
        ctx,
        content=content,
        embed=embed,
        files=out_files,
        ephemeral=ephemeral,
    )


def _parse_kv_query(q: str) -> tuple[dict[str, str | list[str]], list[str]]:
    parts = shlex.split(q)
    out: dict[str, str | list[str]] = {}
    extras: list[str] = []
    last_key: str | None = None
    for part in parts:
        if ":" in part or "=" in part:
            if ":" in part:
                key, value = part.split(":", 1)
            else:
                key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if "," in value:
                out[key] = [v.strip() for v in value.split(",") if v.strip()]
            else:
                out[key] = value
            last_key = key
        else:
            if last_key == "query" and isinstance(out.get("query"), str):
                out["query"] = f"{out['query']} {part}".strip()
            else:
                extras.append(part)
    return out, extras


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    series: pd.Series, *, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(
    series: pd.Series, *, window: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def _plot_candles(df: pd.DataFrame, title: str) -> BytesIO:
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        raise ValueError("empty_history")

    idx = df.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    except Exception:
        pass

    x = mdates.date2num(idx.to_pydatetime())
    candle_width, volume_width = _calc_bar_widths(x)

    fig = plt.figure(figsize=(10, 6), dpi=140)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    up_color = "#22c55e"
    down_color = "#ef4444"

    for xi, o, h, l, c in zip(x, df["Open"], df["High"], df["Low"], df["Close"]):
        color = up_color if c >= o else down_color
        ax.vlines(xi, l, h, linewidth=1.0, color=color, alpha=0.9)
        bottom = min(o, c)
        height = abs(c - o)
        if height == 0:
            height = max(h - l, 1e-9) * 0.02
        rect = plt.Rectangle(
            (xi - candle_width / 2.0, bottom),
            candle_width,
            height,
            fill=True,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.85,
        )
        ax.add_patch(rect)

    if "Volume" in df.columns:
        colors = [up_color if c >= o else down_color for o, c in zip(df["Open"], df["Close"])]
        axv.bar(
            x,
            df["Volume"].fillna(0).values,
            width=volume_width,
            color=colors,
            alpha=0.75,
        )

    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    axv.grid(True, linewidth=0.4, alpha=0.4)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.get_xticklabels(), visible=False)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_replay_command(symbol: str, action: str, **kwargs: Any) -> str:
    parts = [f"symbol:{symbol}", f"action:{action}"]
    for key, val in kwargs.items():
        if val is None:
            continue
        parts.append(f"{key}:{val}")
    joined = " ".join(parts)
    return f'{BOT_PREFIX}finance "{joined}"'


def _vwap(df: pd.DataFrame) -> pd.Series:
    if "Volume" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    denom = vol.cumsum().replace(0, np.nan)
    return (typical * vol).cumsum() / denom


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty or "Close" not in out.columns:
        return out
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()
    if "Volume" in out.columns:
        out["VOL_MA20"] = out["Volume"].rolling(20).mean()
    out["RSI"] = _rsi(out["Close"])
    macd_line, signal_line, hist = _macd(out["Close"])
    out["MACD"] = macd_line
    out["MACD_SIGNAL"] = signal_line
    out["MACD_HIST"] = hist
    out["VWAP"] = _vwap(out)
    return out


def _event_markers(df: pd.DataFrame, events: list[datetime]) -> list[datetime]:
    if df.empty or not events:
        return []
    idx = df.index
    markers: list[datetime] = []
    for dt in events:
        if dt is None:
            continue
        try:
            ts = pd.Timestamp(dt)
            if getattr(idx, "tz", None) is None and ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            markers.append(ts.to_pydatetime())
        except Exception:
            continue
    return markers


def _calc_bar_widths(x: np.ndarray) -> tuple[float, float]:
    if len(x) <= 1:
        return 0.8, 0.9
    diffs = np.diff(x)
    positive = diffs[diffs > 0]
    dx = float(np.median(positive)) if len(positive) > 0 else 1.0
    return max(dx * 0.8, 1e-6), max(dx * 0.9, 1e-6)


def _parse_mtf_spec(raw: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    defaults = [
        ("1D", "1d", "5m"),
        ("1M", "1mo", "1h"),
        ("6M", "6mo", "1d"),
        ("1Y", "1y", "1wk"),
    ]
    token_map = {
        "1d": ("1D", "1d", "5m"),
        "1w": ("1W", "5d", "1h"),
        "1mo": ("1M", "1mo", "1h"),
        "3mo": ("3M", "3mo", "1d"),
        "6mo": ("6M", "6mo", "1d"),
        "1y": ("1Y", "1y", "1wk"),
    }
    vals = [v.strip().lower() for v in (raw or "").split(",") if v.strip()]
    if not vals:
        return defaults, []
    out: list[tuple[str, str, str]] = []
    notes: list[str] = []
    for v in vals[:4]:
        token = v
        m = re.fullmatch(r"(\d+)m", token)
        if m:
            token = f"{m.group(1)}mo"
            notes.append(f"normalized mtf token '{v}' -> '{token}'")
        spec = token_map.get(token)
        if spec is None:
            spec = defaults[0]
            notes.append(f"unknown mtf token '{v}' -> using {spec[0]}")
        out.append(spec)
    while len(out) < 4:
        out.append(defaults[len(out)])
    return out[:4], notes


def _forecast_seed(
    *,
    ticker: str,
    close: pd.Series,
    horizon_days: int,
    paths: int,
    model: str,
) -> int:
    last_ts = str(close.index[-1]) if len(close.index) else ""
    last_price = float(close.iloc[-1]) if len(close) else 0.0
    material = f"{ticker}|{last_ts}|{last_price:.8f}|{horizon_days}|{paths}|{model}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def _ewma_vol(logret: pd.Series, lam: float = 0.94) -> pd.Series:
    if logret.empty:
        return pd.Series(dtype=float)
    var = np.zeros(len(logret), dtype=float)
    var[0] = float(logret.var(ddof=1) or 1e-8)
    for i in range(1, len(logret)):
        prev_r = float(logret.iloc[i - 1])
        var[i] = lam * var[i - 1] + (1 - lam) * (prev_r**2)
    return pd.Series(np.sqrt(np.maximum(var, 1e-12)), index=logret.index)


def _forecast_paths(
    close: pd.Series,
    *,
    horizon_days: int,
    paths: int,
    model: str,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    logret = np.log(close).diff().dropna()
    mu = float(logret.mean())
    sigma = float(logret.std(ddof=1))
    s0 = float(close.iloc[-1])
    rng = np.random.default_rng(seed)

    if model == "ewma_bootstrap":
        vol_hist = _ewma_vol(logret)
        valid = vol_hist.replace(0, np.nan)
        z_hist = (logret / valid).dropna().to_numpy()
        if z_hist.size == 0:
            z_hist = np.array([0.0], dtype=float)
        start_sigma = float(valid.dropna().iloc[-1]) if not valid.dropna().empty else max(sigma, 1e-6)
        path_prices = np.zeros((paths, horizon_days), dtype=float)
        for p in range(paths):
            sig2 = start_sigma**2
            level = s0
            for t in range(horizon_days):
                z = float(rng.choice(z_hist))
                r = mu + np.sqrt(max(sig2, 1e-12)) * z
                level = level * np.exp(r)
                path_prices[p, t] = level
                sig2 = 0.94 * sig2 + 0.06 * (r**2)
        return path_prices, mu, sigma

    shocks = rng.normal(loc=mu, scale=sigma, size=(paths, horizon_days))
    prices = s0 * np.exp(np.cumsum(shocks, axis=1))
    return prices, mu, sigma


def _plot_dash(
    df: pd.DataFrame,
    title: str,
    *,
    preset: str,
    theme: str = "light",
    event_dates: list[datetime] | None = None,
) -> BytesIO:
    src = _with_indicators(df).dropna(subset=["Open", "High", "Low", "Close"])
    if src.empty:
        raise ValueError("empty_history")
    idx = src.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    except Exception:
        pass
    src = src.copy()
    src.index = idx
    x = mdates.date2num(idx.to_pydatetime())
    candle_width, volume_width = _calc_bar_widths(x)
    up_color = "#22c55e"
    down_color = "#ef4444"

    show_rsi = preset in {"swing", "momentum"}
    show_macd = preset == "momentum"
    panel_count = 2 + int(show_rsi) + int(show_macd)
    ratios = [4, 1] + ([1] if show_rsi else []) + ([1] if show_macd else [])

    fig = plt.figure(figsize=(10, 7 if panel_count >= 4 else 6), dpi=140)
    palette = {
        "bg": "#0b1220" if theme == "dark" else "#ffffff",
        "fg": "#e5e7eb" if theme == "dark" else "#111827",
        "grid": 0.2 if theme == "dark" else 0.3,
    }
    fig.patch.set_facecolor(palette["bg"])
    gs = fig.add_gridspec(panel_count, 1, height_ratios=ratios, hspace=0.05)
    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    for ax_panel in (ax_price, ax_vol):
        ax_panel.set_facecolor(palette["bg"])
        ax_panel.tick_params(colors=palette["fg"])
        for spine in ax_panel.spines.values():
            spine.set_color(palette["fg"])
    cursor = 2
    ax_rsi = fig.add_subplot(gs[cursor], sharex=ax_price) if show_rsi else None
    if show_rsi:
        cursor += 1
    ax_macd = fig.add_subplot(gs[cursor], sharex=ax_price) if show_macd else None
    if ax_rsi is not None:
        ax_rsi.set_facecolor(palette["bg"])
        ax_rsi.tick_params(colors=palette["fg"])
        for spine in ax_rsi.spines.values():
            spine.set_color(palette["fg"])
    if ax_macd is not None:
        ax_macd.set_facecolor(palette["bg"])
        ax_macd.tick_params(colors=palette["fg"])
        for spine in ax_macd.spines.values():
            spine.set_color(palette["fg"])

    for xi, o, h, l, c in zip(x, src["Open"], src["High"], src["Low"], src["Close"]):
        color = up_color if c >= o else down_color
        ax_price.vlines(xi, l, h, linewidth=0.9, color=color, alpha=0.9)
        bottom = min(o, c)
        height = abs(c - o)
        if height == 0:
            height = max(h - l, 1e-9) * 0.02
        ax_price.add_patch(
            plt.Rectangle(
                (xi - candle_width / 2.0, bottom),
                candle_width,
                height,
                fill=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.85,
            )
        )

    for ma_name, color in (("MA20", "#2563eb"), ("MA50", "#f59e0b"), ("MA200", "#a855f7")):
        if ma_name == "MA200" and preset != "swing":
            continue
        if ma_name in src.columns and src[ma_name].notna().any():
            ax_price.plot(idx, src[ma_name], linewidth=1.0, label=ma_name, color=color)

    if preset == "intraday" and src["VWAP"].notna().any():
        ax_price.plot(idx, src["VWAP"], linewidth=1.2, label="VWAP", color="#0f766e")

    markers = _event_markers(src, event_dates or [])
    for dt in markers:
        ax_price.axvline(dt, color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.5)

    ax_price.set_title(title, color=palette["fg"])
    ax_price.grid(True, alpha=palette["grid"])
    if ax_price.get_legend_handles_labels()[0]:
        ax_price.legend(loc="upper left", fontsize=8)

    if "Volume" in src.columns:
        vol_colors = [up_color if c >= o else down_color for o, c in zip(src["Open"], src["Close"])]
        ax_vol.bar(
            x,
            src["Volume"].fillna(0).values,
            width=volume_width,
            color=vol_colors,
            alpha=0.75,
        )
        if "VOL_MA20" in src.columns and src["VOL_MA20"].notna().any():
            ax_vol.plot(idx, src["VOL_MA20"], color="#334155", linewidth=0.9, label="VOL MA20")
            ax_vol.legend(loc="upper left", fontsize=7)
    ax_vol.set_ylabel("Vol", color=palette["fg"])
    ax_vol.grid(True, alpha=palette["grid"])

    if ax_rsi is not None and "RSI" in src.columns:
        ax_rsi.plot(idx, src["RSI"], color="purple", linewidth=1.0)
        ax_rsi.axhline(70, color="red", linewidth=0.8, alpha=0.6)
        ax_rsi.axhline(30, color="green", linewidth=0.8, alpha=0.6)
        ax_rsi.set_ylabel("RSI", color=palette["fg"])
        ax_rsi.grid(True, alpha=palette["grid"])

    if ax_macd is not None:
        ax_macd.plot(idx, src["MACD"], linewidth=1.0, label="MACD")
        ax_macd.plot(idx, src["MACD_SIGNAL"], linewidth=1.0, label="Signal")
        ax_macd.bar(idx, src["MACD_HIST"], alpha=0.3, label="Hist")
        ax_macd.grid(True, alpha=palette["grid"])
        ax_macd.legend(loc="upper left", fontsize=7)

    hud = (
        f"O:{_fmt_num(src['Open'].iloc[-1])} H:{_fmt_num(src['High'].iloc[-1])} "
        f"L:{_fmt_num(src['Low'].iloc[-1])} C:{_fmt_num(src['Close'].iloc[-1])}\n"
        f"bars:{len(src)} preset:{preset} theme:{theme}"
    )
    ax_price.text(
        0.01,
        0.98,
        hud,
        transform=ax_price.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color=palette["fg"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": palette["bg"], "alpha": 0.35, "edgecolor": palette["fg"]},
    )

    ax_price.xaxis_date()
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax_price.get_xticklabels(), visible=False)
    if ax_rsi is not None:
        plt.setp(ax_rsi.get_xticklabels(), visible=False if ax_macd is not None else True)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_mtf(
    frames: list[tuple[str, pd.DataFrame]],
    title: str,
    *,
    theme: str = "light",
) -> BytesIO:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    up_color = "#22c55e"
    down_color = "#ef4444"
    palette = {
        "bg": "#0b1220" if theme == "dark" else "#ffffff",
        "fg": "#e5e7eb" if theme == "dark" else "#111827",
        "grid": 0.2 if theme == "dark" else 0.3,
    }
    fig.patch.set_facecolor(palette["bg"])

    for ax, (label, frame) in zip(axes.flat, frames):
        ax.set_facecolor(palette["bg"])
        ax.tick_params(colors=palette["fg"])
        for spine in ax.spines.values():
            spine.set_color(palette["fg"])
        df = _with_indicators(frame).dropna(subset=["Open", "High", "Low", "Close"]) if frame is not None else pd.DataFrame()
        if df.empty:
            ax.text(0.5, 0.5, f"{label}\nno data", ha="center", va="center", color=palette["fg"])
            ax.set_title(label, color=palette["fg"])
            ax.grid(True, alpha=palette["grid"])
            continue
        idx = df.index
        try:
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
        except Exception:
            pass
        df = df.copy()
        df.index = idx
        x = mdates.date2num(idx.to_pydatetime())
        candle_width, _volume_width = _calc_bar_widths(x)
        for xi, o, h, l, c in zip(x, df["Open"], df["High"], df["Low"], df["Close"]):
            color = up_color if c >= o else down_color
            ax.vlines(xi, l, h, linewidth=0.7, color=color, alpha=0.9)
            bottom = min(o, c)
            height = abs(c - o) or max(h - l, 1e-9) * 0.02
            ax.add_patch(
                plt.Rectangle(
                    (xi - candle_width / 2.0, bottom),
                    candle_width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.8,
                )
            )
        for ma_name, color in (("MA20", "#2563eb"), ("MA50", "#f59e0b")):
            if df[ma_name].notna().any():
                ax.plot(idx, df[ma_name], linewidth=0.8, color=color)
        if "Volume" in df.columns:
            ax2 = ax.twinx()
            ax2.fill_between(idx, 0, df["Volume"].fillna(0).values, color="#94a3b8", alpha=0.15)
            ax2.set_yticks([])
        ax.text(
            0.98,
            0.96,
            f"{label}\n{len(df)} bars",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=palette["fg"],
            bbox={"boxstyle": "round,pad=0.2", "facecolor": palette["bg"], "alpha": 0.35, "edgecolor": palette["fg"]},
        )
        ax.set_title(label, color=palette["fg"])
        ax.grid(True, alpha=palette["grid"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    fig.suptitle(title, color=palette["fg"])
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_compare(
    df_target: pd.DataFrame,
    df_bench: pd.DataFrame,
    title: str,
    *,
    base: float,
) -> tuple[BytesIO, float | None, float | None]:
    if df_target.empty or df_bench.empty:
        raise ValueError("empty_history")
    t = df_target[["Close"]].rename(columns={"Close": "target"}).dropna()
    b = df_bench[["Close"]].rename(columns={"Close": "bench"}).dropna()
    merged = t.join(b, how="inner").dropna()
    if merged.empty:
        raise ValueError("empty_history")
    base_value = float(base) if base not in (None, 0) else 100.0
    t_norm = merged["target"] / float(merged["target"].iloc[0]) * base_value
    b_norm = merged["bench"] / float(merged["bench"].iloc[0]) * base_value
    outperf = float(t_norm.iloc[-1] - b_norm.iloc[-1])

    fig = plt.figure(figsize=(10, 5), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(merged.index, t_norm, label="Target", linewidth=1.5)
    ax.plot(merged.index, b_norm, label="Benchmark", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(f"Relative (base={base_value:g})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf, float(t_norm.iloc[-1]), float(b_norm.iloc[-1])


def _normal_quantile(p: float) -> float:
    return NormalDist().inv_cdf(p)


def _weighted_quantile(values: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cw = np.cumsum(w)
    total = float(cw[-1]) if len(cw) else 0.0
    if total <= 0:
        return np.quantile(v, probs)
    targets = np.clip(probs, 0.0, 1.0) * total
    return np.interp(targets, cw, v)


def _deterministic_regime_forecast_core(close: pd.Series, *, horizon_days: int) -> dict[str, Any]:
    close = close.dropna()
    if len(close) < 60:
        return {"ok": False, "reason": "not_enough_history"}

    logret = np.log(close).diff().dropna()
    if len(logret) < 40:
        return {"ok": False, "reason": "not_enough_history"}

    ewma_sigma = _ewma_vol(logret).replace(0, np.nan)
    sigma_fallback = float(logret.std(ddof=1) or 1e-6)
    sigma_t = float(ewma_sigma.dropna().iloc[-1]) if not ewma_sigma.dropna().empty else sigma_fallback
    sigma_series = ewma_sigma.fillna(sigma_fallback)

    sigma_med = float(sigma_series.median())
    high_mask = sigma_series >= sigma_med
    low_mask = ~high_mask

    mu_low = float(logret[low_mask].mean()) if low_mask.any() else float(logret.mean())
    mu_high = float(logret[high_mask].mean()) if high_mask.any() else float(logret.mean())
    sigma_low = float(logret[low_mask].std(ddof=1)) if low_mask.sum() > 2 else sigma_fallback
    sigma_high = float(logret[high_mask].std(ddof=1)) if high_mask.sum() > 2 else max(sigma_fallback, sigma_t)

    if sigma_med > 1e-12:
        regime_prob_high = float(np.clip((sigma_t - sigma_med) / sigma_med, -2.0, 2.0))
    else:
        regime_prob_high = 0.0
    p_high = float(1.0 / (1.0 + np.exp(-2.5 * regime_prob_high)))
    p_low = 1.0 - p_high

    mu_mix = p_low * mu_low + p_high * mu_high
    sigma_mix = np.sqrt(max(p_low * sigma_low**2 + p_high * sigma_high**2, 1e-12))

    residual = (logret - mu_mix) / sigma_series
    residual = residual.replace([np.inf, -np.inf], np.nan).dropna()
    if len(residual) < 20:
        residual = ((logret - float(logret.mean())) / max(float(logret.std(ddof=1)), 1e-8)).replace([np.inf, -np.inf], np.nan).dropna()

    tail_cut = float(np.quantile(np.abs(logret), 0.92))
    jumps = logret[np.abs(logret) >= tail_cut]
    jump_prob = float(len(jumps) / max(len(logret), 1))
    jump_mu = float(jumps.mean()) if len(jumps) else 0.0
    jump_var = float(jumps.var(ddof=1)) if len(jumps) > 2 else 0.0

    probs = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=float)
    z_norm = np.array([_normal_quantile(p) for p in probs], dtype=float)
    if len(residual) > 0:
        recency = np.exp(np.linspace(-2.0, 0.0, len(residual), dtype=float))
        z_emp = _weighted_quantile(residual.to_numpy(dtype=float), probs, recency)
    else:
        z_emp = z_norm
    z_blend = 0.65 * z_norm + 0.35 * z_emp

    s0 = float(close.iloc[-1])
    x = np.arange(1, max(1, int(horizon_days)) + 1, dtype=float)
    drift = (mu_mix - 0.5 * sigma_mix**2 + jump_prob * jump_mu) * x
    scale = np.sqrt(np.maximum(sigma_mix**2 * x + jump_prob * jump_var * x, 1e-12))

    fan = np.exp(np.log(s0) + drift[None, :] + scale[None, :] * z_blend[:, None])
    fan = np.maximum(fan, 1e-8)

    labels = ("p05", "p25", "p50", "p75", "p95")
    end_quantiles = {k: float(fan[i, -1]) for i, k in enumerate(labels)}

    return {
        "ok": True,
        "s0": s0,
        "mu": mu_mix,
        "sigma": sigma_mix,
        "horizon_days": int(horizon_days),
        "end_quantiles": end_quantiles,
        "var95": float(max(0.0, s0 - end_quantiles["p05"])),
        "model": "gbm_analytic",
        "deterministic": True,
        "engine": "regime_ewma_jump_blend",
        "regime": {
            "p_low": p_low,
            "p_high": p_high,
            "mu_low": mu_low,
            "mu_high": mu_high,
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
        },
        "jump": {
            "prob": jump_prob,
            "mu": jump_mu,
            "var": jump_var,
        },
        "fan": {
            "x": x.tolist(),
            "p05": fan[0].tolist(),
            "p25": fan[1].tolist(),
            "p50": fan[2].tolist(),
            "p75": fan[3].tolist(),
            "p95": fan[4].tolist(),
        },
    }


def _deterministic_gbm_forecast(close: pd.Series, *, horizon_days: int) -> dict[str, Any]:
    return _deterministic_regime_forecast_core(close, horizon_days=horizon_days)


def _plot_forecast_fan_deterministic(close: pd.Series, *, horizon_days: int = 20) -> BytesIO:
    result = _deterministic_regime_forecast_core(close, horizon_days=horizon_days)
    if not result.get("ok"):
        raise ValueError(str(result.get("reason") or "forecast_error"))

    fan = result["fan"]
    x = np.array(fan["x"], dtype=float)
    q05 = np.array(fan["p05"], dtype=float)
    q25 = np.array(fan["p25"], dtype=float)
    q50 = np.array(fan["p50"], dtype=float)
    q75 = np.array(fan["p75"], dtype=float)
    q95 = np.array(fan["p95"], dtype=float)

    fig = plt.figure(figsize=(9, 4), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(x, q50, label="Median", linewidth=1.5)
    ax.fill_between(x, q05, q95, color="blue", alpha=0.1, label="p05-p95")
    ax.fill_between(x, q25, q75, color="blue", alpha=0.2, label="p25-p75")
    ax.set_title("Forecast fan chart (deterministic regime-EWMA)")
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_ta(df: pd.DataFrame, title: str) -> BytesIO:
    if df.empty:
        raise ValueError("empty_history")
    idx = df.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    except Exception:
        pass

    close = df["Close"]
    rsi = _rsi(close)
    macd_line, signal_line, hist = _macd(close)

    fig = plt.figure(figsize=(10, 7), dpi=140)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax)
    ax_macd = fig.add_subplot(gs[2], sharex=ax)

    ax.plot(idx, close, linewidth=1.4, label="Close")
    upper, mid, lower = _bollinger(close)
    ax.plot(idx, mid, linewidth=1.0, label="BB Mid")
    ax.plot(idx, upper, linewidth=0.8, label="BB Upper")
    ax.plot(idx, lower, linewidth=0.8, label="BB Lower")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    ax_rsi.plot(idx, rsi, linewidth=1.0, color="purple")
    ax_rsi.axhline(70, color="red", linewidth=0.8, alpha=0.6)
    ax_rsi.axhline(30, color="green", linewidth=0.8, alpha=0.6)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.grid(True, alpha=0.3)

    ax_macd.plot(idx, macd_line, label="MACD", linewidth=1.0)
    ax_macd.plot(idx, signal_line, label="Signal", linewidth=1.0)
    ax_macd.bar(idx, hist, label="Hist", alpha=0.3)
    ax_macd.grid(True, alpha=0.3)
    ax_macd.legend(loc="upper left", fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _mc_forecast(
    close: pd.Series,
    *,
    ticker: str,
    horizon_days: int = 20,
    paths: int = 2000,
    model: str = "gbm_seeded",
) -> dict[str, Any]:
    close = close.dropna()
    if len(close) < 40:
        return {"ok": False, "reason": "not_enough_history"}

    s0 = float(close.iloc[-1])
    seed = _forecast_seed(
        ticker=ticker,
        close=close,
        horizon_days=horizon_days,
        paths=paths,
        model=model,
    )
    prices, mu, sigma = _forecast_paths(
        close,
        horizon_days=horizon_days,
        paths=paths,
        model=model,
        seed=seed,
    )
    end = prices[:, -1]
    qs = np.quantile(prices, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    return {
        "ok": True,
        "s0": s0,
        "mu": mu,
        "sigma": sigma,
        "horizon_days": horizon_days,
        "paths": paths,
        "model": model,
        "seed": int(seed),
        "end_quantiles": {
            "p05": float(np.quantile(end, 0.05)),
            "p25": float(np.quantile(end, 0.25)),
            "p50": float(np.quantile(end, 0.50)),
            "p75": float(np.quantile(end, 0.75)),
            "p95": float(np.quantile(end, 0.95)),
        },
        "var95": float(np.quantile(s0 - end, 0.95)),
        "fan": {
            "x": list(range(1, horizon_days + 1)),
            "p05": qs[0].tolist(),
            "p25": qs[1].tolist(),
            "p50": qs[2].tolist(),
            "p75": qs[3].tolist(),
            "p95": qs[4].tolist(),
        },
    }


def _plot_forecast_fan_from_quantiles(fan: dict[str, Any], *, title: str = "Forecast fan chart") -> BytesIO:
    x = np.array(fan.get("x") or [], dtype=float)
    q05 = np.array(fan.get("p05") or [], dtype=float)
    q25 = np.array(fan.get("p25") or [], dtype=float)
    q50 = np.array(fan.get("p50") or [], dtype=float)
    q75 = np.array(fan.get("p75") or [], dtype=float)
    q95 = np.array(fan.get("p95") or [], dtype=float)
    if not len(x):
        raise ValueError("empty_fan")

    fig = plt.figure(figsize=(9, 4), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(x, q50, label="Median", linewidth=1.5)
    ax.fill_between(x, q05, q95, color="blue", alpha=0.1, label="p05-p95")
    ax.fill_between(x, q25, q75, color="blue", alpha=0.2, label="p25-p75")
    ax.set_title(title)
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_forecast_fan(
    close: pd.Series,
    *,
    ticker: str,
    horizon_days: int = 20,
    paths: int = 2000,
    model: str = "gbm_seeded",
) -> BytesIO:
    result = _mc_forecast(
        close,
        ticker=ticker,
        horizon_days=horizon_days,
        paths=paths,
        model=model,
    )
    if not result.get("ok"):
        raise ValueError(str(result.get("reason") or "forecast_error"))
    return _plot_forecast_fan_from_quantiles(result["fan"], title="Forecast fan chart")


class Finance(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.reg = SymbolRegistry.load()
        self.yf = YFinanceClient()

        self._watch_lock = asyncio.Lock()
        self._watch: dict[str, Any] = self._load_watch()
        self._monitor_task: asyncio.Task | None = asyncio.create_task(
            self._monitor_loop()
        )

    async def _yahoo_search_http(
        self,
        query: str,
        *,
        region: str = "JP",
        lang: str = "ja-JP",
        quotes_count: int = 8,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        params = {
            "q": query,
            "quotesCount": str(quotes_count),
            "newsCount": "0",
            "listsCount": "0",
            "enableFuzzyQuery": "true",
            "region": region,
            "lang": lang,
        }
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as resp:
                    data = await resp.json(content_type=None)
        except Exception:
            log.exception("Yahoo search HTTP failed query=%s params=%s", query, params)
            return []
        quotes = data.get("quotes") or []
        return [q for q in quotes if isinstance(q, dict)]

    def _load_watch(self) -> dict[str, Any]:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not WATCH_FILE.exists():
            WATCH_FILE.write_text(
                json.dumps({"entries": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        try:
            return json.loads(WATCH_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"entries": []}

    def _save_watch(self) -> None:
        WATCH_FILE.write_text(
            json.dumps(self._watch, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def _resolve_display_name(self, ticker: str, display: str) -> str:
        if display != ticker:
            return display
        try:
            quote = await self.yf.get_quote(ticker)
        except Exception:
            return display
        return _display_name_from_quote(ticker, display, quote)

    @staticmethod
    def _normalize_symbol_input(raw: str) -> str:
        s = (raw or "").strip()
        if not s:
            return s
        return s.rstrip(" ,，、")

    async def _resolve_symbol(
        self, raw: str, *, region: str, lookup_kind: str
    ) -> tuple[str, str]:
        s = self._normalize_symbol_input(raw)
        if not s:
            raise ValueError("symbol_empty")

        if CODE_ONLY_RE.match(s) or JPX_CODE_RE.match(s):
            log.info(
                "resolve_symbol: code-only search query=%s region=%s kind=%s",
                s,
                region,
                lookup_kind,
            )
            return await self._resolve_via_search(
                s, prefer_code=True, region=region, lookup_kind=lookup_kind
            )

        if TICKER_RE.match(s):
            s_norm = s.strip()
            s_cmp = s_norm.upper()
            results = await self._search_candidates(s_norm, limit=10, region=region)
            if results:
                for row in results:
                    symbol = str(row.get("symbol") or "").strip()
                    if symbol and symbol.upper() == s_cmp:
                        return symbol, symbol
                mismatches = [
                    str(row.get("symbol") or "").strip()
                    for row in results
                    if str(row.get("symbol") or "").strip()
                ]
                raise LookupError("symbol_mismatch:" + ", ".join(mismatches))
            log.info(
                "resolve_symbol: ticker regex matched, no search results symbol=%s",
                s_norm,
            )
            return s_norm, s_norm

        hit = self.reg.find_by_name(s)
        if hit:
            log.info("resolve_symbol: registry hit symbol=%s", s)
            return hit.ticker, hit.name

        log.info(
            "resolve_symbol: fallback search query=%s region=%s kind=%s",
            s,
            region,
            lookup_kind,
        )
        return await self._resolve_via_search(
            s, prefer_code=False, region=region, lookup_kind=lookup_kind
        )

    @staticmethod
    def _pick_search_candidate(
        results: list[dict[str, Any]],
        query: str,
        *,
        prefer_code: bool,
        lookup_kind: str,
        region: str,
    ) -> tuple[str | None, str | None, list[dict[str, Any]]]:
        candidates = [r for r in results if r.get("symbol")]
        if not candidates:
            return None, None, []

        if prefer_code:
            for row in candidates:
                symbol = str(row.get("symbol") or "")
                if symbol.startswith(f"{query}."):
                    name = row.get("name") or row.get("shortname") or row.get("longname")
                    return symbol, str(name) if name else symbol, []
            for row in candidates:
                symbol = str(row.get("symbol") or "")
                if symbol == query:
                    name = row.get("name") or row.get("shortname") or row.get("longname")
                    return symbol, str(name) if name else symbol, []

        normalized_kind = (lookup_kind or "all").strip().lower()
        allowed_types = {
            "equity": {"EQUITY"},
            "stock": {"EQUITY"},
            "etf": {"ETF"},
            "index": {"INDEX"},
            "fx": {"CURRENCY", "FX"},
            "future": {"FUTURE"},
            "crypto": {"CRYPTO", "CRYPTOCURRENCY"},
        }.get(normalized_kind)

        def _score(row: dict[str, Any]) -> int:
            symbol = str(row.get("symbol") or "")
            quote_type = str(row.get("type") or row.get("quoteType") or "")
            score = 0
            if allowed_types:
                if quote_type in allowed_types:
                    score += 6
                elif quote_type:
                    score -= 4
            else:
                if quote_type in {"EQUITY", "ETF"}:
                    score += 2
            if symbol.endswith("=F"):
                score += 3 if normalized_kind == "future" else -2
            if symbol.endswith("=X"):
                score += 3 if normalized_kind == "fx" else -2
            if region == "JP" and symbol.endswith(".T"):
                score += 2
            if CODE_ONLY_RE.match(query) and symbol.startswith(f"{query}."):
                score += 5
            if symbol and symbol.upper() == query.upper():
                score += 6
            return score

        scored = [(row, _score(row)) for row in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored:
            return None, None, []
        best_row, best_score = scored[0]
        if len(scored) > 1 and (best_score - scored[1][1]) < 2:
            return None, None, [row for row, _score in scored[:5]]
        symbol = str(best_row.get("symbol") or "")
        name = best_row.get("name") or best_row.get("shortname") or best_row.get("longname")
        return symbol or None, str(name) if name else symbol or None, []

    async def _resolve_via_search(
        self, query: str, *, prefer_code: bool, region: str, lookup_kind: str
    ) -> tuple[str, str]:
        results = await self._search_candidates(query, limit=10, region=region)
        if not results:
            raise LookupError("symbol_unknown")

        ticker, display, candidates = self._pick_search_candidate(
            results,
            query,
            prefer_code=prefer_code,
            lookup_kind=lookup_kind,
            region=region,
        )
        if not ticker:
            if candidates:
                symbols = [
                    str(row.get("symbol") or "").strip()
                    for row in candidates
                    if row.get("symbol")
                ]
                if symbols:
                    raise LookupError("symbol_ambiguous:" + ", ".join(symbols))
            raise LookupError("symbol_unknown")

        return ticker, display or ticker

    async def cog_unload(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()

    @commands.command(
        name="finance",
        description="Stocks: quote/chart/news/watchlist/data (Yahoo Finance via yfinance).",
        help=(
            "Examples:\n"
            "  /finance symbol:7203.T action:summary\n"
            "  /finance symbol:Toyota Motor Corporation action:summary\n"
            "  /finance symbol:7203.T action:chart period:6mo interval:1d\n"
            "  /finance symbol:7203.T action:candle period:3mo interval:1d\n"
            "  /finance symbol:7203.T action:ta period:6mo interval:1d\n"
            "  /finance symbol:7203.T action:mtf\n"
            "  /finance symbol:7203.T action:compare bench:1306.T period:6mo interval:1d\n"
            "  /finance symbol:7203.T action:forecast horizon_days:20 forecast_model:gbm_analytic\n"
            "  /finance action:symbols\n"
            "  /finance action:search query:7203\n"
            "  /finance action:search query:MicroAd\n"
            "  /finance action:lookup query:Toyota kind:stock\n"
            "  /finance action:screener_local min_mcap:1e11 max_pe:20\n"
            "  /finance symbol:7203.T action:news limit:5\n"
            "  /finance symbol:7203.T action:watch_add threshold_pct:2 check_every_s:300\n"
            "  /finance symbol:7203.T action:data section:financials\n"
            f"  {BOT_PREFIX}finance 7203.T summary\n"
            f"  {BOT_PREFIX}finance \"symbol:7203.T action:candle period:6mo interval:1d\""
        ),
        extras={
            "category": "Tools",
            "destination": "Analyze market data with quote, indicators, forecast, and chart actions.",
            "plus": "Use key:value arguments (e.g., symbol:, action:, period:, interval:); action:search helps when you only know a company name.",
            "pro": (
                "Flexible symbol matching (tickers or Yahoo search resolution) plus "
                "best-effort data pulls from Yahoo Finance via yfinance, "
                "including chart output. Use action:search if you only know the company name."),
        },
    )
    async def finance(self, ctx: commands.Context, *, raw: str = "") -> None:
        raw_args = (raw or "").strip()
        symbol: str | None = None
        action = "summary"
        period = "1mo"
        interval = "1d"
        auto_adjust = True
        limit = 5
        channel_id: int | None = None
        threshold_pct = 2.0
        check_every_s = 300
        section = "fast_info"
        max_rows = 30
        query: str | None = None
        region = "JP"
        remove_all = False
        horizon_days = 20
        paths = 2000
        lookup_kind = "all"
        screener_min_mcap: float | None = None
        screener_max_pe: float | None = None
        preset = "basic"
        events = "auto"
        ui = "auto"
        bench: str | None = None
        base = 100.0
        mtf = "1d,1mo,6mo,1y"
        theme = "light"
        forecast_model = "gbm_analytic"

        has_kv = bool(raw_args) and any(
            f"{key}:" in raw_args or f"{key}=" in raw_args for key in FINANCE_KV_KEYS
        )
        def _as_int(val: object, default: int) -> int:
            try:
                return int(val)
            except Exception:
                return default

        def _as_float(val: object) -> float | None:
            try:
                return float(val)
            except Exception:
                return None

        def _as_bool(val: object) -> bool | None:
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered in {"true", "1", "yes", "y", "on"}:
                    return True
                if lowered in {"false", "0", "no", "n", "off"}:
                    return False
            return None

        if raw_args:
            if has_kv:
                kv, extras = _parse_kv_query(raw_args)
                if extras:
                    log.warning(
                        "finance kv parse extras=%s raw_args=%s", extras, raw_args
                    )
                    await safe_reply(
                        ctx,
                        content=tag_error_text(
                            "Unrecognized tokens in key/value input: "
                            + " ".join(extras)
                            + ". Use key:value pairs; wrap multi-word query values in quotes."
                        ),
                    )
                    return
                if "symbol" not in kv and "ticker" in kv:
                    kv["symbol"] = kv.get("ticker")
                symbol = str(kv.get("symbol") or "").strip() or None
                action = str(kv.get("action") or "summary").strip().lower()
                period = str(kv.get("period") or "1mo").strip()
                interval = str(kv.get("interval") or "1d").strip()
                query = str(kv.get("query") or "").strip() or None
                region = str(kv.get("region") or "JP").strip()
                lookup_kind = (
                    str(kv.get("lookup_kind") or kv.get("kind") or "all")
                    .strip()
                    .lower()
                )
                section = str(kv.get("section") or "fast_info").strip()
                preset = str(kv.get("preset") or "basic").strip().lower()
                theme = str(kv.get("theme") or "light").strip().lower()
                events = str(kv.get("events") or "auto").strip().lower()
                ui = str(kv.get("ui") or "auto").strip().lower()
                bench = str(kv.get("bench") or kv.get("compare") or "").strip() or None
                mtf = str(kv.get("mtf") or "1d,1mo,6mo,1y").strip()
                forecast_model = (
                    str(kv.get("forecast_model") or "gbm_analytic").strip().lower()
                )
                limit = _as_int(kv.get("limit"), limit)
                horizon_days = _as_int(kv.get("horizon_days"), horizon_days)
                paths = _as_int(kv.get("paths"), paths)
                max_rows = _as_int(kv.get("max_rows"), max_rows)
                if kv.get("channel_id") is not None:
                    try:
                        channel_id = int(kv.get("channel_id"))
                    except Exception:
                        channel_id = None
                threshold_pct_value = _as_float(kv.get("threshold_pct"))
                if threshold_pct_value is not None:
                    threshold_pct = threshold_pct_value
                check_every_s = _as_int(kv.get("check_every_s"), check_every_s)
                screener_min_mcap_raw = kv.get("min_market_cap", kv.get("min_mcap"))
                screener_max_pe_raw = kv.get("max_pe")
                screener_min_mcap_value = _as_float(screener_min_mcap_raw)
                screener_max_pe_value = _as_float(screener_max_pe_raw)
                if screener_min_mcap_value is not None:
                    screener_min_mcap = screener_min_mcap_value
                if screener_max_pe_value is not None:
                    screener_max_pe = screener_max_pe_value
                auto_adjust_value = _as_bool(kv.get("auto_adjust"))
                if auto_adjust_value is not None:
                    auto_adjust = auto_adjust_value
                remove_all_value = _as_bool(kv.get("remove_all"))
                if remove_all_value is not None:
                    remove_all = remove_all_value
                base_value = _as_float(kv.get("base"))
                if base_value is not None:
                    base = base_value
                log.info(
                    "finance parsed kv action=%s symbol=%s query=%s period=%s interval=%s region=%s kind=%s section=%s",
                    action,
                    symbol,
                    query,
                    period,
                    interval,
                    region,
                    lookup_kind,
                    section,
                )
            else:
                try:
                    tokens = shlex.split(raw_args)
                except ValueError:
                    await safe_reply(
                        ctx,
                        content=tag_error_text(
                            "Failed to parse arguments. Check quotes in the input."
                        ),
                    )
                    return
                positional_fields: list[tuple[str, str]] = [
                    ("symbol", "str"),
                    ("action", "str"),
                    ("period", "str"),
                    ("interval", "str"),
                    ("auto_adjust", "bool"),
                    ("limit", "int"),
                    ("channel_id", "int"),
                    ("threshold_pct", "float"),
                    ("check_every_s", "int"),
                    ("section", "str"),
                    ("max_rows", "int"),
                    ("query", "str"),
                    ("region", "str"),
                    ("remove_all", "bool"),
                    ("horizon_days", "int"),
                    ("paths", "int"),
                    ("lookup_kind", "str"),
                    ("screener_min_mcap", "float"),
                    ("screener_max_pe", "float"),
                    ("preset", "str"),
                    ("events", "str"),
                    ("ui", "str"),
                    ("bench", "str"),
                    ("base", "float"),
                    ("mtf", "str"),
                ]
                if len(tokens) > len(positional_fields):
                    await safe_reply(
                        ctx,
                        content=tag_error_text(
                            "Too many positional arguments. Use key:value pairs instead."
                        ),
                    )
                    return
                for token, (field, kind) in zip(tokens, positional_fields):
                    if kind == "str":
                        value: object = token
                    elif kind == "int":
                        try:
                            value = int(token)
                        except ValueError:
                            await safe_reply(
                                ctx,
                                content=tag_error_text(
                                    f"Invalid integer for {field}: {token}"
                                ),
                            )
                            return
                    elif kind == "float":
                        try:
                            value = float(token)
                        except ValueError:
                            await safe_reply(
                                ctx,
                                content=tag_error_text(
                                    f"Invalid number for {field}: {token}"
                                ),
                            )
                            return
                    elif kind == "bool":
                        parsed = _as_bool(token)
                        if parsed is None:
                            await safe_reply(
                                ctx,
                                content=tag_error_text(
                                    f"Invalid boolean for {field}: {token}"
                                ),
                            )
                            return
                        value = parsed
                    else:
                        value = token
                    if field == "symbol":
                        symbol = str(value).strip() or None
                    elif field == "action":
                        action = str(value).strip()
                    elif field == "period":
                        period = str(value).strip()
                    elif field == "interval":
                        interval = str(value).strip()
                    elif field == "auto_adjust":
                        auto_adjust = bool(value)
                    elif field == "limit":
                        limit = int(value)
                    elif field == "channel_id":
                        channel_id = int(value)
                    elif field == "threshold_pct":
                        threshold_pct = float(value)
                    elif field == "check_every_s":
                        check_every_s = int(value)
                    elif field == "section":
                        section = str(value).strip()
                    elif field == "max_rows":
                        max_rows = int(value)
                    elif field == "query":
                        query = str(value).strip() or None
                    elif field == "region":
                        region = str(value).strip()
                    elif field == "remove_all":
                        remove_all = bool(value)
                    elif field == "horizon_days":
                        horizon_days = int(value)
                    elif field == "paths":
                        paths = int(value)
                    elif field == "lookup_kind":
                        lookup_kind = str(value).strip()
                    elif field == "screener_min_mcap":
                        screener_min_mcap = float(value)
                    elif field == "screener_max_pe":
                        screener_max_pe = float(value)
                    elif field == "preset":
                        preset = str(value).strip()
                    elif field == "events":
                        events = str(value).strip()
                    elif field == "ui":
                        ui = str(value).strip()
                    elif field == "bench":
                        bench = str(value).strip() or None
                    elif field == "base":
                        base = float(value)
                    elif field == "mtf":
                        mtf = str(value).strip()
        await self._finance_impl(
            ctx,
            symbol=symbol,
            action=action,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            limit=limit,
            channel_id=channel_id,
            threshold_pct=threshold_pct,
            check_every_s=check_every_s,
            section=section,
            max_rows=max_rows,
            query=query,
            region=region,
            remove_all=remove_all,
            horizon_days=horizon_days,
            paths=paths,
            lookup_kind=lookup_kind,
            screener_min_mcap=screener_min_mcap,
            screener_max_pe=screener_max_pe,
            preset=preset,
            events=events,
            ui=ui,
            bench=bench,
            base=base,
            mtf=mtf,
            theme=theme,
            forecast_model=forecast_model,
            raw_args=raw_args,
            has_kv=has_kv,
        )

    @app_commands.command(
        name="finance",
        description="Stocks: quote/chart/news/watchlist/data (Yahoo Finance via yfinance).",
    )
    @app_commands.choices(action=FINANCE_ACTION_CHOICES)
    async def finance_slash(
        self,
        interaction: discord.Interaction,
        symbol: str | None = None,
        action: str = "summary",
        period: str = "1mo",
        interval: str = "1d",
        auto_adjust: bool = True,
        limit: int = 5,
        channel_id: int | None = None,
        threshold_pct: float = 2.0,
        check_every_s: int = 300,
        section: str = "fast_info",
        max_rows: int = 30,
        query: str | None = None,
        region: str = "JP",
        remove_all: bool = False,
        horizon_days: int = 20,
        paths: int = 2000,
        lookup_kind: str = "all",
        screener_min_mcap: float | None = None,
        screener_max_pe: float | None = None,
        preset: str = "basic",
        events: str = "auto",
        ui: str = "auto",
        bench: str | None = None,
        base: float = 100.0,
        mtf: str = "1d,1mo,6mo,1y",
    ) -> None:
        ctx_factory = getattr(commands.Context, "from_interaction", None)
        if ctx_factory is None:
            await interaction.response.send_message(
                "This command isn't available right now. Please try again later.",
                ephemeral=True,
            )
            return
        ctx_candidate = ctx_factory(interaction)
        ctx = await ctx_candidate if inspect.isawaitable(ctx_candidate) else ctx_candidate
        await self._finance_impl(
            ctx,
            symbol=symbol,
            action=action,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            limit=limit,
            channel_id=channel_id,
            threshold_pct=threshold_pct,
            check_every_s=check_every_s,
            section=section,
            max_rows=max_rows,
            query=query,
            region=region,
            remove_all=remove_all,
            horizon_days=horizon_days,
            paths=paths,
            lookup_kind=lookup_kind,
            screener_min_mcap=screener_min_mcap,
            screener_max_pe=screener_max_pe,
            preset=preset,
            events=events,
            ui=ui,
            bench=bench,
            base=base,
            mtf=mtf,
            theme="light",
            forecast_model="gbm_analytic",
            raw_args="",
            has_kv=False,
        )

    async def _finance_impl(
        self,
        ctx: commands.Context,
        *,
        symbol: str | None = None,
        action: str = "summary",
        period: str = "1mo",
        interval: str = "1d",
        auto_adjust: bool = True,
        limit: int = 5,
        channel_id: int | None = None,
        threshold_pct: float = 2.0,
        check_every_s: int = 300,
        section: str = "fast_info",
        max_rows: int = 30,
        query: str | None = None,
        region: str = "JP",
        remove_all: bool = False,
        horizon_days: int = 20,
        paths: int = 2000,
        lookup_kind: str = "all",
        screener_min_mcap: float | None = None,
        screener_max_pe: float | None = None,
        preset: str = "basic",
        events: str = "auto",
        ui: str = "auto",
        bench: str | None = None,
        base: float = 100.0,
        mtf: str = "1d,1mo,6mo,1y",
        theme: str = "light",
        forecast_model: str = "gbm_analytic",
        raw_args: str = "",
        has_kv: bool = False,
    ) -> None:
        is_prefix = isinstance(ctx, commands.Context) and ctx.interaction is None
        log.info(
            "finance invocation type=%s raw_args=%s has_kv=%s",
            "prefix" if is_prefix else "slash",
            raw_args if is_prefix else "",
            has_kv if is_prefix else False,
        )
        region = (region or "JP").strip().upper()
        action = action.strip().lower() or "summary"
        preset = (preset or "basic").strip().lower()
        if preset not in {"basic", "swing", "momentum", "intraday"}:
            preset = "basic"
        if theme not in {"light", "dark"}:
            theme = "light"
        if forecast_model not in {"gbm_analytic", "gbm_seeded", "ewma_bootstrap"}:
            forecast_model = "gbm_analytic"
        if ui not in {"auto", "buttons", "none"}:
            ui = "none"
        if is_prefix:
            ui = "none"
        if getattr(ctx, "author", None) and getattr(ctx.author, "bot", False):
            ui = "none"

        if symbol and action == "summary" and symbol in FINANCE_ACTION_SET:
            action = symbol
            symbol = None

        if action not in FINANCE_ACTION_SET:
            action_list = ", ".join(FINANCE_ACTIONS)
            await safe_reply(
                ctx,
                content=tag_error_text(
                    f"Unknown action: {action}. Choose one of: {action_list}."
                ),
            )
            return

        if query and action not in {"search", "lookup"}:
            await safe_reply(
                ctx,
                content=tag_error_text(
                    "query is only valid with action:search or action:lookup. Use symbol: for other actions."
                ),
            )
            return

        if action == "symbols":
            await self._send_symbols(ctx)
            return

        if action == "search":
            await self._send_search(ctx, query=query or symbol, region=region, limit=limit)
            return

        if symbol is None:
            await safe_reply(
                ctx, content=tag_error_text("symbol is required for this action.")
            )
            return

        if action == "quote":
            await self._send_quote_only(
                ctx, symbol, region=region, lookup_kind=lookup_kind
            )
            return
        if action in {"summary", "chart"}:
            await self._send_summary(
                ctx,
                symbol=symbol,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                region=region,
                lookup_kind=lookup_kind,
                preset=preset,
                events_mode=events,
                ui_mode=ui,
                theme=theme,
            )
            return
        if action == "news":
            await self._send_news(
                ctx, symbol, limit=limit, region=region, lookup_kind=lookup_kind
            )
            return
        if action == "candle":
            await self._send_candle(
                ctx,
                symbol=symbol,
                period=period,
                interval=interval,
                region=region,
                lookup_kind=lookup_kind,
            )
            return
        if action == "mtf":
            await self._send_mtf(
                ctx,
                symbol=symbol,
                mtf=mtf,
                theme=theme,
                region=region,
                lookup_kind=lookup_kind,
            )
            return
        if action == "compare":
            await self._send_compare(
                ctx,
                symbol=symbol,
                period=period,
                interval=interval,
                bench=bench,
                base=base,
                region=region,
                lookup_kind=lookup_kind,
            )
            return
        if action == "ta":
            await self._send_ta(
                ctx,
                symbol=symbol,
                period=period,
                interval=interval,
                region=region,
                lookup_kind=lookup_kind,
            )
            return
        if action == "forecast":
            await self._send_forecast(
                ctx,
                symbol=symbol,
                period=period,
                interval=interval,
                horizon_days=horizon_days,
                paths=paths,
                region=region,
                lookup_kind=lookup_kind,
            )
            return
        if action == "lookup":
            await self._send_lookup(ctx, query=query or symbol, kind=lookup_kind, limit=limit)
            return
        if action == "screener_local":
            await self._send_screener_local(
                ctx,
                min_market_cap=screener_min_mcap,
                max_pe=screener_max_pe,
                limit=limit,
            )
            return
        if action == "data":
            await self._send_data(
                ctx,
                symbol,
                section=section,
                max_rows=max_rows,
                region=region,
                lookup_kind=lookup_kind,
            )
            return
        if action == "watch_add":
            await self._watch_add(
                ctx,
                symbol,
                channel_id=channel_id,
                threshold_pct=threshold_pct,
                check_every_s=check_every_s,
                region=region,
                lookup_kind=lookup_kind,
            )
            return
        if action == "watch_remove":
            await self._watch_remove(
                ctx,
                symbol,
                channel_id=channel_id,
                remove_all=remove_all,
                region=region,
                lookup_kind=lookup_kind,
            )
            return
        if action == "watch_list":
            await self._watch_list(ctx)
            return

        await safe_reply(ctx, content=tag_error_text(f"Unknown action: {action}"))

    async def _send_symbol_error(
        self, ctx: commands.Context, raw: str, exc: Exception
    ) -> None:
        msg = str(exc)
        suggestions: list[dict[str, Any]] = []
        err_code = "symbol_error"
        if isinstance(exc, LookupError) and msg.startswith("symbol_unknown:"):
            err_code = "symbol_unknown"
            names = [n.strip() for n in msg.split(":", 1)[1].split(",") if n.strip()]
            suggestions = [{"name": n} for n in names]
            desc = (
                "Unknown symbol. Try a ticker (e.g. 7203.T) or "
                "use /finance action:search query:<name>.\n"
                "If search returns no results, try the English name or ticker, "
                "or check via browser/web search.\nSuggestions:\n"
                + "\n".join(names)
            )
        elif isinstance(exc, LookupError) and msg.startswith("symbol_ambiguous:"):
            err_code = "symbol_ambiguous"
            raw_symbols = msg.split(":", 1)[1].strip()
            symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]
            suggestions = [{"symbol": s} for s in symbols[:10]]
            desc_lines = [s["symbol"] for s in suggestions]
            desc = (
                "Symbol is ambiguous. Please specify the exact ticker.\n"
                "Candidates:\n"
                + ("\n".join(desc_lines) if desc_lines else "(no candidates)")
            )
        elif isinstance(exc, LookupError) and msg.startswith("ticker_missing_suffix:"):
            err_code = "ticker_missing_suffix"
            raw_cands = msg.split(":", 1)[1].strip().split()
            for c in raw_cands[:10]:
                if ":" in c:
                    t, n = c.split(":", 1)
                    suggestions.append({"ticker": t, "name": n})
            desc_lines = [f"{s['ticker']} -> {s['name']}" for s in suggestions]
            desc = (
                "Ticker needs exchange suffix.\nCandidates:\n"
                + ("\n".join(desc_lines) if desc_lines else "(no registry matches)")
            )
        elif isinstance(exc, LookupError) and msg.startswith("symbol_mismatch:"):
            err_code = "symbol_mismatch"
            raw_symbols = msg.split(":", 1)[1].strip()
            symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]
            suggestions = [{"symbol": s} for s in symbols[:10]]
            desc_lines = [s["symbol"] for s in suggestions]
            desc = (
                "Symbol mismatch from Yahoo search results. "
                "Check the ticker or run /finance action:search query:<code>.\n"
                "Candidates:\n"
                + ("\n".join(desc_lines) if desc_lines else "(no candidates)")
            )
        elif isinstance(exc, LookupError) and msg == "ticker_missing_suffix":
            err_code = "ticker_missing_suffix"
            desc = (
                "Ticker needs exchange suffix (e.g. 7203.T). "
                "Use /finance action:search query:7203 to list candidates."
            )
        elif isinstance(exc, LookupError):
            err_code = "symbol_unknown"
            desc = (
                "Unknown symbol. Use a ticker (e.g. 7203.T) or "
                "run /finance action:search query:<name/code> to find one. "
                "If search fails, try the English name or use browser/web search."
            )
        elif isinstance(exc, ValueError) and msg == "symbol_empty":
            err_code = "symbol_empty"
            desc = "Symbol is required."
        else:
            desc = f"Symbol error: {repr(exc)}"

        if err_code == "ticker_missing_suffix" and not suggestions:
            search_results = await self._search_candidates(raw, limit=10)
            if search_results:
                suggestions = search_results
                desc_lines = [
                    f"{item.get('symbol')} -> {item.get('name') or ''} ({item.get('exchange') or ''})"
                    for item in search_results[:10]
                ]
                desc = "Ticker needs exchange suffix.\nCandidates:\n" + "\n".join(
                    desc_lines
                )

        payload = {
            "ok": False,
            "error": err_code,
            "input": raw,
            "suggestions": suggestions,
        }
        embed = discord.Embed(title="Finance error", description=desc, color=0xFF0000)
        embed = tag_error_embed(embed)
        await _send_with_json(
            ctx,
            payload=payload,
            meta={"ok": False, "error": err_code, "input": raw},
            embed=embed,
            ephemeral=True,
            error=True,
        )

    async def _search_candidates(
        self, query: str, *, limit: int = 10, region: str = "JP"
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        quotes = await self._yahoo_search_http(
            query, region=region, lang="ja-JP", quotes_count=limit
        )
        results: list[dict[str, Any]] = []
        for q in quotes[:limit]:
            results.append(
                {
                    "symbol": q.get("symbol"),
                    "name": q.get("shortname") or q.get("longname"),
                    "exchange": q.get("exchDisp") or q.get("exchange"),
                    "type": q.get("quoteType"),
                }
            )
        if results:
            return results
        log.info(
            "Yahoo HTTP search returned no results, falling back to yfinance.Search query=%s region=%s",
            query,
            region,
        )

        def _sync():
            s = Search(query, max_results=limit, news_count=0, enable_fuzzy_query=True)
            return list(s.quotes or [])

        try:
            quotes = await _to_thread(_sync)
        except Exception:
            log.exception("yfinance.Search failed query=%s", query)
            return []
        for q in quotes[:limit]:
            if isinstance(q, dict):
                results.append(
                    {
                        "symbol": q.get("symbol"),
                        "name": q.get("shortname") or q.get("longname"),
                        "exchange": q.get("exchDisp") or q.get("exchange"),
                        "type": q.get("quoteType"),
                    }
                )
        return results

    async def _send_symbols(self, ctx: commands.Context) -> None:
        lines = [f"{s.name} -> {s.ticker}" for s in self.reg.symbols]
        payload = {
            "symbols": [{"name": s.name, "ticker": s.ticker} for s in self.reg.symbols],
            "count": len(self.reg.symbols),
        }
        embed = discord.Embed(
            title="Registered symbols",
            description="\n".join(lines[:120]),
            color=0x2B90D9,
        )
        await _send_with_json(
            ctx,
            payload=payload,
            meta={"ok": True, "action": "symbols", "count": payload["count"]},
            embed=embed,
        )

    async def _send_search(
        self,
        ctx: commands.Context,
        *,
        query: str | None,
        region: str,
        limit: int,
    ) -> None:
        query = (query or "").strip()
        if not query:
            payload = {
                "ok": False,
                "error": "query_required",
                "input": query,
                "suggestions": [],
            }
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": False, "error": "query_required", "action": "search"},
                embed=tag_error_embed(
                    discord.Embed(
                        title="Finance error",
                        description="query is required for search.",
                        color=0xFF0000,
                    )
                ),
                ephemeral=True,
                error=True,
            )
            return

        await defer_interaction(ctx)
        limit = max(1, min(int(limit), 20))
        items = await self._search_candidates(query, limit=limit, region=region)
        payload = {"ok": True, "query": query, "region": region, "results": items}
        lines = [
            f"{r.get('symbol')}:{r.get('name') or ''} ({r.get('exchange') or ''})"
            for r in items
            if r.get("symbol")
        ]
        message = "\n".join(lines[:20]) if lines else "(no results)"
        embed = discord.Embed(
            title=f"Search results: {query}",
            description=message,
            color=0x2B90D9,
        )
        await _send_with_json(
            ctx,
            payload=payload,
            meta={"ok": True, "action": "search", "query": query},
            embed=embed,
        )

    async def _send_quote_only(
        self, ctx: commands.Context, symbol: str, *, region: str, lookup_kind: str
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        await defer_interaction(ctx)
        try:
            data = await self.yf.get_quote(ticker)
        except Exception as e:
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return
        display = _display_name_from_quote(ticker, display, data)
        view = _build_quote_view(data)

        now_jst = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))
        desc = (
            f"As of {now_jst.strftime('%Y-%m-%d %H:%M:%S JST')}\n"
            f"Change: {_fmt_num(view['change'])} ({_fmt_pct(view['change_pct'])})"
        )

        embed = discord.Embed(
            title=f"{display} ({ticker})", description=desc, color=0x2B90D9
        )
        embed.add_field(
            name="Price",
            value=f"{_fmt_num(view['price'])} {view['currency']}",
            inline=True,
        )
        embed.add_field(name="Prev close", value=_fmt_num(view["prev_close"]), inline=True)
        embed.add_field(name="Open", value=_fmt_num(view["open"]), inline=True)
        embed.add_field(
            name="Day low/high",
            value=f"{_fmt_num(view['day_low'])} / {_fmt_num(view['day_high'])}",
            inline=True,
        )
        embed.add_field(
            name="52w low/high",
            value=f"{_fmt_num(view['52w_low'])} / {_fmt_num(view['52w_high'])}",
            inline=True,
        )
        embed.add_field(
            name="Volume", value=_fmt_num(view["volume"], nd=0), inline=True
        )

        payload = {
            "symbol_input": symbol,
            "ticker": ticker,
            "yahoo_symbol": ticker,
            "display": display,
            "asof_jst": now_jst.isoformat(),
            "quote": {
                k: view[k]
                for k in (
                    "price",
                    "prev_close",
                    "open",
                    "day_low",
                    "day_high",
                    "52w_low",
                    "52w_high",
                    "volume",
                    "market_cap",
                    "currency",
                    "change",
                    "change_pct",
                )
            },
        }
        await _send_with_json(
            ctx,
            payload=payload,
            meta={"ok": True, "action": "quote", "ticker": ticker},
            embed=embed,
        )

    async def _send_summary(
        self,
        ctx: commands.Context,
        *,
        symbol: str,
        period: str,
        interval: str,
        auto_adjust: bool,
        region: str,
        lookup_kind: str,
        preset: str,
        events_mode: str,
        ui_mode: str,
        theme: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        await defer_interaction(ctx)

        try:
            quote = await self.yf.get_quote(ticker)
            display = _display_name_from_quote(ticker, display, quote)
            view = _build_quote_view(quote)
            hist = await self.yf.get_history(
                ticker, period=period, interval=interval, auto_adjust=auto_adjust
            )
        except ValueError as e:
            if str(e) == "intraday_limit_60d":
                await safe_reply(
                    ctx,
                    content=tag_error_text(
                        "Intraday intervals are limited to the last 60 days. "
                        "Use a shorter period or a >=1d interval."
                    ),
                )
                return
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return
        except Exception as e:
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return

        hist_i = _with_indicators(hist)
        event_dates: list[datetime] = []
        upcoming: list[str] = []
        intraday = interval in INTRADAY_INTERVALS
        effective_events = (events_mode or "auto").lower()
        if effective_events == "auto":
            effective_events = "0" if intraday else "1"
        if effective_events in {"1", "true", "yes", "on"}:
            event_dates, upcoming = await self._collect_event_info(ticker)

        file = None
        try:
            buf = _plot_dash(
                hist,
                title=f"{display} ({ticker}) {period}/{interval}",
                preset=preset,
                theme=theme,
                event_dates=event_dates,
            )
            file = discord.File(buf, filename="dash.png")
        except Exception:
            file = None

        now_jst = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))
        desc = (
            f"As of {now_jst.strftime('%Y-%m-%d %H:%M:%S JST')} | period={period} "
            f"interval={interval}\n"
            f"Change: {_fmt_num(view['change'])} ({_fmt_pct(view['change_pct'])})"
        )
        embed = discord.Embed(
            title=f"{display} ({ticker})", description=desc, color=0x2B90D9
        )
        embed.add_field(
            name="Price",
            value=f"{_fmt_num(view['price'])} {view['currency']}",
            inline=True,
        )
        embed.add_field(name="Prev close", value=_fmt_num(view["prev_close"]), inline=True)
        embed.add_field(
            name="Day low/high",
            value=f"{_fmt_num(view['day_low'])} / {_fmt_num(view['day_high'])}",
            inline=True,
        )
        embed.add_field(
            name="52w low/high",
            value=f"{_fmt_num(view['52w_low'])} / {_fmt_num(view['52w_high'])}",
            inline=True,
        )
        embed.add_field(
            name="Volume", value=_fmt_num(view["volume"], nd=0), inline=True
        )
        latest = hist_i.iloc[-1] if not hist_i.empty else None
        vol_ma20 = latest.get("VOL_MA20") if latest is not None and "VOL_MA20" in hist_i.columns else None
        vol_ratio = None
        try:
            if latest is not None and vol_ma20 not in (None, 0) and "Volume" in hist_i.columns:
                vol_ratio = float(latest.get("Volume")) / float(vol_ma20)
        except Exception:
            vol_ratio = None
        embed.add_field(
            name="Vol context",
            value=f"vol/vol_ma20: {_fmt_num(vol_ratio)}",
            inline=True,
        )
        momentum_parts = []
        if preset in {"swing", "momentum"} and latest is not None:
            momentum_parts.append(f"RSI14: {_fmt_num(latest.get('RSI'))}")
        if preset == "momentum" and latest is not None:
            momentum_parts.append(f"MACD hist: {_fmt_signed_num(latest.get('MACD_HIST'))}")
        if momentum_parts:
            embed.add_field(name="Momentum", value=" | ".join(momentum_parts), inline=False)
        if upcoming:
            embed.add_field(name="Upcoming", value="\n".join(upcoming[:3]), inline=False)
        replay = _build_replay_command(
            ticker,
            "summary",
            period=period,
            interval=interval,
            preset=preset,
            events=effective_events,
            ui="none" if ui_mode == "none" else "auto",
            theme=theme,
        )
        embed.add_field(name="Replay", value=f"```txt\n{replay}\n```", inline=False)

        payload = {
            "symbol_input": symbol,
            "ticker": ticker,
            "yahoo_symbol": ticker,
            "display": display,
            "asof_jst": now_jst.isoformat(),
            "params": {
                "period": period,
                "interval": interval,
                "auto_adjust": auto_adjust,
                "preset": preset,
                "events": effective_events,
                "ui": ui_mode,
                "theme": theme,
            },
            "upcoming": upcoming[:3],
            "replay": replay,
            "quote": {
                k: view[k]
                for k in (
                    "price",
                    "prev_close",
                    "open",
                    "day_low",
                    "day_high",
                    "52w_low",
                    "52w_high",
                    "volume",
                    "market_cap",
                    "currency",
                    "change",
                    "change_pct",
                )
            },
        }

        if file:
            embed.set_image(url="attachment://dash.png")
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "summary", "ticker": ticker},
                embed=embed,
                files=[file],
            )
        else:
            embed.add_field(name="Chart", value="(plot unavailable)", inline=False)
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "summary", "ticker": ticker},
                embed=embed,
            )

    async def _collect_event_info(self, ticker: str) -> tuple[list[datetime], list[str]]:
        def _sync() -> tuple[list[datetime], list[str]]:
            t = yf.Ticker(ticker)
            dates: list[datetime] = []
            upcoming: list[tuple[str, datetime]] = []
            today = datetime.now(timezone.utc).date()

            def _as_dt(value: Any) -> datetime | None:
                if value is None:
                    return None
                try:
                    ts = pd.Timestamp(value)
                    if ts.tzinfo is not None:
                        ts = ts.tz_convert("UTC")
                    else:
                        ts = ts.tz_localize("UTC")
                    return ts.to_pydatetime()
                except Exception:
                    return None

            try:
                cal_raw = None
                try:
                    cal_raw = t.get_calendar()
                except Exception:
                    cal_raw = getattr(t, "calendar", None)

                if isinstance(cal_raw, pd.DataFrame):
                    for val in cal_raw.values.flatten().tolist():
                        dt = _as_dt(val)
                        if dt is None:
                            continue
                        dates.append(dt)
                        if dt.date() >= today:
                            upcoming.append(("earnings", dt))
                elif isinstance(cal_raw, dict):
                    for key in ("Earnings Date", "Earnings Average"):
                        val = cal_raw.get(key)
                        vals = val if isinstance(val, (list, tuple)) else [val]
                        for item in vals:
                            dt = _as_dt(item)
                            if dt is None:
                                continue
                            dates.append(dt)
                            if dt.date() >= today:
                                upcoming.append(("earnings", dt))
            except Exception:
                pass

            try:
                div = getattr(t, "dividends", None)
                if isinstance(div, pd.Series) and not div.empty:
                    last_div = div.index[-1].to_pydatetime()
                    dates.append(last_div)
                    if last_div.date() >= today:
                        upcoming.append(("dividend", last_div))
            except Exception:
                pass

            try:
                splits = getattr(t, "splits", None)
                if isinstance(splits, pd.Series) and not splits.empty:
                    last_split = splits.index[-1].to_pydatetime()
                    dates.append(last_split)
                    if last_split.date() >= today:
                        upcoming.append(("split", last_split))
            except Exception:
                pass

            uniq = sorted({d.date() for d in dates})
            uniq_dt = [datetime.combine(d, datetime.min.time()) for d in uniq]
            priority = {"earnings": 0, "dividend": 1, "split": 2}
            upcoming.sort(key=lambda item: (priority.get(item[0], 9), item[1]))
            lines = [f"{kind}: {dt.strftime('%Y-%m-%d')}" for kind, dt in upcoming[:3]]
            return uniq_dt, lines

        try:
            return await _to_thread(_sync)
        except Exception:
            return [], []

    async def _send_candle(
        self,
        ctx: commands.Context,
        *,
        symbol: str,
        period: str,
        interval: str,
        region: str,
        lookup_kind: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        await defer_interaction(ctx)

        try:
            hist = await self.yf.get_history(ticker, period=period, interval=interval)
        except ValueError as e:
            if str(e) == "intraday_limit_60d":
                await safe_reply(
                    ctx,
                    content=tag_error_text(
                        "Intraday intervals are limited to the last 60 days. "
                        "Use a shorter period or a >=1d interval."
                    ),
                )
                return
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return
        except Exception as e:
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return

        display = await self._resolve_display_name(ticker, display)

        try:
            buf = _plot_candles(hist, title=f"{display} ({ticker}) candle {period}/{interval}")
            file = discord.File(buf, filename="candle.png")
        except Exception:
            await safe_reply(ctx, content=tag_error_text("No data for candle chart."))
            return

        embed = discord.Embed(
            title=f"{display} ({ticker}) candle",
            description=f"period={period} interval={interval}",
            color=0x2B90D9,
        )
        replay = _build_replay_command(
            ticker,
            "candle",
            period=period,
            interval=interval,
            ui="none",
        )
        embed.add_field(name="Replay", value=f"```txt\n{replay}\n```", inline=False)
        embed.set_image(url="attachment://candle.png")
        payload = {
            "ok": True,
            "action": "candle",
            "symbol_input": symbol,
            "ticker": ticker,
            "display": display,
            "params": {"period": period, "interval": interval},
            "replay": replay,
        }
        await _send_with_json(
            ctx,
            payload=payload,
            meta={"ok": True, "action": "candle", "ticker": ticker},
            embed=embed,
            files=[file],
        )

    async def _send_mtf(
        self,
        ctx: commands.Context,
        *,
        symbol: str,
        mtf: str,
        region: str,
        lookup_kind: str,
        theme: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(symbol, region=region, lookup_kind=lookup_kind)
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return
        await defer_interaction(ctx)
        display = await self._resolve_display_name(ticker, display)

        pairs, parse_notes = _parse_mtf_spec(mtf)
        frames: list[tuple[str, pd.DataFrame]] = []
        mtf_notes: list[str] = list(parse_notes)
        for label, p, iv in pairs:
            try:
                frame = await self.yf.get_history(ticker, period=p, interval=iv)
            except Exception:
                if label == "1D":
                    frame = await self.yf.get_history(ticker, period="1d", interval="15m")
                    mtf_notes.append(f"{label}: fallback interval=15m")
                else:
                    frame = pd.DataFrame()
                    mtf_notes.append(f"{label}: no data ({p}/{iv})")
            frames.append((label, frame))
        try:
            buf = _plot_mtf(frames, f"{display} ({ticker}) MTF", theme=theme)
            file = discord.File(buf, filename="mtf.png")
        except Exception:
            await safe_reply(ctx, content=tag_error_text("No data for MTF chart."))
            return

        replay = _build_replay_command(ticker, "mtf", mtf=mtf or "1d,1mo,6mo,1y", ui="none", theme=theme)
        embed = discord.Embed(
            title=f"{display} ({ticker}) MTF",
            description="1D/1M/6M/1Y overview",
            color=0x2B90D9,
        )
        if mtf_notes:
            embed.add_field(name="Notes", value="\n".join(mtf_notes[:4]), inline=False)
        embed.add_field(name="Replay", value=f"```txt\n{replay}\n```", inline=False)
        embed.set_image(url="attachment://mtf.png")
        await _send_with_json(
            ctx,
            payload={"ok": True, "action": "mtf", "ticker": ticker, "display": display, "replay": replay, "notes": mtf_notes},
            meta={"ok": True, "action": "mtf", "ticker": ticker},
            embed=embed,
            files=[file],
        )

    async def _send_compare(
        self,
        ctx: commands.Context,
        *,
        symbol: str,
        period: str,
        interval: str,
        bench: str | None,
        base: float,
        region: str,
        lookup_kind: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(symbol, region=region, lookup_kind=lookup_kind)
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return
        bench_input = (bench or "").strip()
        bench_explicit = bench_input or None
        bench_symbol = bench_explicit or ("1306.T" if region == "JP" else "SPY")
        await defer_interaction(ctx)
        display = await self._resolve_display_name(ticker, display)
        if (not math.isfinite(base)) or base <= 0:
            await safe_reply(
                ctx,
                content=tag_error_text("compare error: base must be a finite number > 0."),
            )
            return

        try:
            hist_target = await self.yf.get_history(ticker, period=period, interval=interval)
            hist_bench = await self.yf.get_history(bench_symbol, period=period, interval=interval)
            if hist_bench.empty and bench_explicit is None and region == "JP":
                bench_symbol = "^N225"
                hist_bench = await self.yf.get_history(bench_symbol, period=period, interval=interval)
            if hist_bench.empty and bench_explicit is None and bench_symbol != "SPY":
                bench_symbol = "SPY"
                hist_bench = await self.yf.get_history(bench_symbol, period=period, interval=interval)
            if hist_bench.empty and bench_explicit is not None:
                await safe_reply(
                    ctx,
                    content=tag_error_text(
                        f"compare error: benchmark returned no data ({bench_symbol})"
                    ),
                )
                return
            buf, last_target, last_bench = _plot_compare(
                hist_target,
                hist_bench,
                title=f"{display} ({ticker}) vs {bench_symbol} ({period}/{interval})",
                base=base,
            )
            file = discord.File(buf, filename="compare.png")
        except Exception as e:
            await safe_reply(ctx, content=tag_error_text(f"compare error: {repr(e)}"))
            return

        outperf = None
        try:
            if last_target is not None and last_bench is not None:
                outperf = float(last_target) - float(last_bench)
        except Exception:
            outperf = None
        replay = _build_replay_command(
            ticker,
            "compare",
            period=period,
            interval=interval,
            bench=bench_symbol,
            base=base,
            ui="none",
        )
        embed = discord.Embed(
            title=f"{display} ({ticker}) compare",
            description=f"period={period} interval={interval} bench={bench_symbol} base={base}",
            color=0x2B90D9,
        )
        embed.add_field(name="Target(last)", value=_fmt_num(last_target), inline=True)
        embed.add_field(name="Bench(last)", value=_fmt_num(last_bench), inline=True)
        embed.add_field(name="Outperformance", value=_fmt_signed_num(outperf), inline=True)
        embed.add_field(name="Replay", value=f"```txt\n{replay}\n```", inline=False)
        embed.set_image(url="attachment://compare.png")
        await _send_with_json(
            ctx,
            payload={
                "ok": True,
                "action": "compare",
                "ticker": ticker,
                "display": display,
                "bench": bench_symbol,
                "base": base,
                "last_target": last_target,
                "last_bench": last_bench,
                "outperformance": outperf,
                "replay": replay,
            },
            meta={"ok": True, "action": "compare", "ticker": ticker},
            embed=embed,
            files=[file],
        )

    async def _send_ta(
        self,
        ctx: commands.Context,
        *,
        symbol: str,
        period: str,
        interval: str,
        region: str,
        lookup_kind: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        await defer_interaction(ctx)

        try:
            hist = await self.yf.get_history(ticker, period=period, interval=interval)
        except ValueError as e:
            if str(e) == "intraday_limit_60d":
                await safe_reply(
                    ctx,
                    content=tag_error_text(
                        "Intraday intervals are limited to the last 60 days. "
                        "Use a shorter period or a >=1d interval."
                    ),
                )
                return
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return
        except Exception as e:
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return

        display = await self._resolve_display_name(ticker, display)

        if hist.empty or "Close" not in hist.columns:
            await safe_reply(ctx, content=tag_error_text("No data for indicators."))
            return

        close = hist["Close"]
        hist = hist.copy()
        hist["RSI"] = _rsi(close)
        macd_line, signal_line, macd_hist = _macd(close)
        hist["MACD"] = macd_line
        hist["MACD_signal"] = signal_line
        hist["MACD_hist"] = macd_hist
        upper, mid, lower = _bollinger(close)
        hist["BB_upper"] = upper
        hist["BB_mid"] = mid
        hist["BB_lower"] = lower

        try:
            buf = _plot_ta(hist, title=f"{display} ({ticker}) TA")
            file = discord.File(buf, filename="ta.png")
        except Exception:
            file = None

        latest = hist.dropna().iloc[-1] if not hist.dropna().empty else hist.iloc[-1]
        embed = discord.Embed(
            title=f"{display} ({ticker}) indicators",
            description=f"period={period} interval={interval}",
            color=0x2B90D9,
        )
        embed.add_field(name="RSI", value=_fmt_num(latest.get("RSI")), inline=True)
        embed.add_field(name="MACD", value=_fmt_num(latest.get("MACD")), inline=True)
        embed.add_field(
            name="BB Mid", value=_fmt_num(latest.get("BB_mid")), inline=True
        )
        if file:
            embed.set_image(url="attachment://ta.png")

        csv_bytes = _to_csv_bytes(hist, max_rows=200)
        csv_file = discord.File(BytesIO(csv_bytes), filename=f"{ticker}.ta.csv")
        payload = {
            "ok": True,
            "action": "ta",
            "symbol_input": symbol,
            "ticker": ticker,
            "display": display,
            "params": {"period": period, "interval": interval},
            "latest": {
                "rsi": latest.get("RSI"),
                "macd": latest.get("MACD"),
                "macd_signal": latest.get("MACD_signal"),
                "macd_hist": latest.get("MACD_hist"),
                "bb_mid": latest.get("BB_mid"),
                "bb_upper": latest.get("BB_upper"),
                "bb_lower": latest.get("BB_lower"),
            },
        }
        files = [csv_file]
        if file:
            files.append(file)
        await _send_with_json(
            ctx,
            payload=payload,
            meta={"ok": True, "action": "ta", "ticker": ticker},
            embed=embed,
            files=files,
        )

    async def _send_forecast(
        self,
        ctx: commands.Context,
        *,
        symbol: str,
        period: str,
        interval: str,
        horizon_days: int,
        paths: int,
        region: str,
        lookup_kind: str,
        forecast_model: str = "gbm_analytic",
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        await defer_interaction(ctx)

        try:
            hist = await self.yf.get_history(ticker, period=period, interval=interval)
        except ValueError as e:
            if str(e) == "intraday_limit_60d":
                await safe_reply(
                    ctx,
                    content=tag_error_text(
                        "Intraday intervals are limited to the last 60 days. "
                        "Use a shorter period or a >=1d interval."
                    ),
                )
                return
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return
        except Exception as e:
            await safe_reply(ctx, content=tag_error_text(f"Finance error: {repr(e)}"))
            return

        display = await self._resolve_display_name(ticker, display)

        if hist.empty or "Close" not in hist.columns:
            await safe_reply(ctx, content=tag_error_text("No data for forecast."))
            return

        model = (forecast_model or "gbm_analytic").strip().lower()
        if model not in {"gbm_analytic", "gbm_seeded", "ewma_bootstrap"}:
            model = "gbm_analytic"

        if model == "gbm_analytic":
            result = _deterministic_gbm_forecast(
                hist["Close"],
                horizon_days=max(1, int(horizon_days)),
            )
        else:
            result = _mc_forecast(
                hist["Close"],
                ticker=ticker,
                horizon_days=max(1, int(horizon_days)),
                paths=max(100, int(paths)),
                model=model,
            )
        if not result.get("ok"):
            await safe_reply(
                ctx, content=tag_error_text("Not enough history for forecast.")
            )
            return

        replay = _build_replay_command(
            ticker,
            "forecast",
            period=period,
            interval=interval,
            horizon_days=max(1, int(horizon_days)),
            paths=max(100, int(paths)),
            forecast_model=model,
            ui="none",
        )

        try:
            if model == "gbm_analytic":
                buf = _plot_forecast_fan_deterministic(
                    hist["Close"],
                    horizon_days=max(1, int(horizon_days)),
                )
            else:
                fan = result.get("fan")
                if isinstance(fan, dict):
                    buf = _plot_forecast_fan_from_quantiles(
                        fan,
                        title=f"Forecast fan chart ({model})",
                    )
                else:
                    buf = _plot_forecast_fan(
                        hist["Close"],
                        ticker=ticker,
                        horizon_days=max(1, int(horizon_days)),
                        paths=max(100, int(paths)),
                        model=model,
                    )
            file = discord.File(buf, filename="forecast.png")
        except Exception:
            file = None

        embed = discord.Embed(
            title=f"{display} ({ticker}) forecast",
            description=f"horizon={horizon_days} days paths={paths} model={model}",
            color=0x2B90D9,
        )
        embed.add_field(
            name="End p50", value=_fmt_num(result["end_quantiles"]["p50"]), inline=True
        )
        embed.add_field(
            name="End p05/p95",
            value=f"{_fmt_num(result['end_quantiles']['p05'])} / {_fmt_num(result['end_quantiles']['p95'])}",
            inline=True,
        )
        embed.add_field(name="VaR95", value=_fmt_num(result["var95"]), inline=True)
        if file:
            embed.set_image(url="attachment://forecast.png")
        embed.add_field(name="Replay", value=f"```txt\n{replay}\n```", inline=False)

        payload = {
            "ok": True,
            "action": "forecast",
            "symbol_input": symbol,
            "ticker": ticker,
            "display": display,
            "params": {"period": period, "interval": interval, "horizon_days": horizon_days, "paths": paths, "model": model},
            "seed": result.get("seed"),
            "replay": replay,
            "forecast": result,
        }
        if file:
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "forecast", "ticker": ticker},
                embed=embed,
                files=[file],
            )
        else:
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "forecast", "ticker": ticker},
                embed=embed,
            )

    async def _send_lookup(
        self, ctx: commands.Context, *, query: str | None, kind: str, limit: int
    ) -> None:
        query = (query or "").strip()
        if not query:
            payload = {"ok": False, "error": "query_required", "input": query}
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": False, "error": "query_required", "action": "lookup"},
                embed=tag_error_embed(
                    discord.Embed(
                        title="Finance error",
                        description="query is required for lookup.",
                        color=0xFF0000,
                    )
                ),
                ephemeral=True,
                error=True,
            )
            return

        await defer_interaction(ctx)

        def _sync():
            lu = yf.Lookup(query=query)
            getters = {
                "all": lu.get_all,
                "stock": lu.get_stock,
                "etf": lu.get_etf,
                "index": lu.get_index,
                "future": lu.get_future,
                "currency": lu.get_currency,
                "crypto": lu.get_cryptocurrency,
                "mutualfund": lu.get_mutualfund,
            }
            fn = getters.get(kind, lu.get_all)
            return fn(count=max(1, min(int(limit), 50)))

        try:
            df = await _to_thread(_sync)
        except Exception as e:
            await safe_reply(ctx, content=tag_error_text(f"lookup error: {repr(e)}"))
            return

        file = None
        preview = ""
        try:
            b = _to_csv_bytes(df, max_rows=200)
            file = discord.File(BytesIO(b), filename=f"lookup.{kind}.csv")
            preview = str(df.head(10))
        except Exception:
            preview = str(df)[:900]

        embed = discord.Embed(
            title=f"Lookup: {query}",
            description=f"kind={kind}\n\npreview:\n{preview}",
            color=0x2B90D9,
        )
        payload = {
            "ok": True,
            "action": "lookup",
            "query": query,
            "kind": kind,
            "rows": int(getattr(df, "shape", [0])[0]) if df is not None else 0,
        }
        if file:
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "lookup", "query": query},
                embed=embed,
                files=[file],
            )
        else:
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "lookup", "query": query},
                embed=embed,
            )

    async def _send_screener_local(
        self,
        ctx: commands.Context,
        *,
        min_market_cap: float | None,
        max_pe: float | None,
        limit: int,
    ) -> None:
        await defer_interaction(ctx)
        rows: list[dict[str, Any]] = []

        async def _fetch_one(sym: ListedSymbol) -> dict[str, Any]:
            def _sync():
                t = yf.Ticker(sym.ticker)
                info = {}
                try:
                    info = dict(t.fast_info or {})
                except Exception:
                    info = {}
                if not info:
                    try:
                        info = dict(t.info or {})
                    except Exception:
                        info = {}
                return info

            info = await _to_thread(_sync)
            mcap = info.get("market_cap") or info.get("marketCap")
            pe = info.get("trailingPE") or info.get("trailing_pe")
            return {
                "name": sym.name,
                "ticker": sym.ticker,
                "market_cap": mcap,
                "pe": pe,
            }

        sem = asyncio.Semaphore(5)

        async def _guard(sym: ListedSymbol) -> dict[str, Any]:
            async with sem:
                return await _fetch_one(sym)

        results = await asyncio.gather(
            *(_guard(sym) for sym in self.reg.symbols), return_exceptions=True
        )
        for result in results:
            if isinstance(result, dict):
                rows.append(result)

        df = pd.DataFrame(rows)
        if "market_cap" in df.columns:
            df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        if "pe" in df.columns:
            df["pe"] = pd.to_numeric(df["pe"], errors="coerce")
        if min_market_cap is not None:
            df = df[df["market_cap"] >= float(min_market_cap)]
        if max_pe is not None:
            df = df[df["pe"] <= float(max_pe)]
        df = df.sort_values(by="market_cap", ascending=False).head(max(1, min(int(limit), 50)))

        file = None
        preview = ""
        try:
            b = _to_csv_bytes(df, max_rows=200)
            file = discord.File(BytesIO(b), filename="screener_local.csv")
            preview = str(df.head(10))
        except Exception:
            preview = str(df)[:900]

        embed = discord.Embed(
            title="Screener (local registry)",
            description=f"min_market_cap={min_market_cap} max_pe={max_pe}\n\npreview:\n{preview}",
            color=0x2B90D9,
        )
        payload = {
            "ok": True,
            "action": "screener_local",
            "min_market_cap": min_market_cap,
            "max_pe": max_pe,
            "rows": int(df.shape[0]),
        }
        if file:
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "screener_local"},
                embed=embed,
                files=[file],
            )
        else:
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "screener_local"},
                embed=embed,
            )

    async def _send_news(
        self,
        ctx: commands.Context,
        symbol: str,
        limit: int,
        *,
        region: str,
        lookup_kind: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        await defer_interaction(ctx)

        def _sync():
            t = yf.Ticker(ticker)
            try:
                return list(t.news or [])
            except Exception as ex:
                return [{"_error": repr(ex)}]

        items = await _to_thread(_sync)
        limit = max(1, min(int(limit), 10))
        display = await self._resolve_display_name(ticker, display)

        lines: list[str] = []
        for it in items[:limit]:
            if "_error" in it:
                lines.append(f"news error: {it['_error']}")
                continue
            title = str(it.get("title") or "").strip()
            link = str(it.get("link") or "").strip()
            pub = it.get("providerPublishTime")
            if pub:
                try:
                    dt = datetime.fromtimestamp(int(pub), tz=timezone.utc).astimezone(
                        timezone(timedelta(hours=9))
                    )
                    pub_s = dt.strftime("%Y-%m-%d %H:%M JST")
                except Exception:
                    pub_s = "N/A"
            else:
                pub_s = "N/A"
            if title and link:
                lines.append(f"- {title}\n  {pub_s}\n  {link}")

        if not lines:
            lines = ["No news items were returned."]

        embed = discord.Embed(
            title=f"{display} ({ticker}) news",
            description="\n".join(lines),
            color=0x2B90D9,
        )
        payload = {
            "ticker": ticker,
            "display": display,
            "limit": limit,
            "items": [
                {
                    "title": str(it.get("title") or "").strip(),
                    "link": str(it.get("link") or "").strip(),
                    "provider_publish_time": it.get("providerPublishTime"),
                }
                for it in items[:limit]
                if "_error" not in it
            ],
        }
        await _send_with_json(
            ctx,
            payload=payload,
            meta={"ok": True, "action": "news", "ticker": ticker},
            embed=embed,
        )

    async def _send_data(
        self,
        ctx: commands.Context,
        symbol: str,
        *,
        section: str,
        max_rows: int,
        region: str,
        lookup_kind: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        section = (section or "").strip()
        if section not in DATA_SECTIONS:
            msg = "bad section. allowed:\n" + "\n".join(sorted(DATA_SECTIONS.keys()))
            await safe_reply(ctx, content=tag_error_text(msg))
            return

        await defer_interaction(ctx)

        try:
            val = await self.yf.get_section(ticker, section)
        except Exception as e:
            await safe_reply(ctx, content=tag_error_text(f"data error: {repr(e)}"))
            return

        display = await self._resolve_display_name(ticker, display)

        preview = ""
        file = None

        try:
            if isinstance(val, (pd.DataFrame, pd.Series)):
                b = _to_csv_bytes(val, max_rows=max(1, min(int(max_rows), 200)))
                file = discord.File(BytesIO(b), filename=f"{ticker}.{section}.csv")
                preview = str(val.head(5))
            else:
                b = json.dumps(val, ensure_ascii=False, indent=2, default=str).encode(
                    "utf-8"
                )
                file = discord.File(BytesIO(b), filename=f"{ticker}.{section}.json")
                preview = json.dumps(val, ensure_ascii=False, default=str)[:900] + "..."
        except Exception:
            preview = str(val)[:900]

        embed = discord.Embed(
            title=f"{display} ({ticker}) data: {section}",
            description=f"section={section}\n{DATA_SECTIONS[section]}\n\npreview:\n{preview}",
            color=0x2B90D9,
        )

        payload = {"ticker": ticker, "display": display, "section": section}
        if file:
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "data", "ticker": ticker, "section": section},
                embed=embed,
                files=[file],
            )
        else:
            await _send_with_json(
                ctx,
                payload=payload,
                meta={"ok": True, "action": "data", "ticker": ticker, "section": section},
                embed=embed,
            )

    async def _watch_add(
        self,
        ctx: commands.Context,
        symbol: str,
        *,
        channel_id: int | None,
        threshold_pct: float,
        check_every_s: int,
        region: str,
        lookup_kind: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        if channel_id is None:
            channel_id = ctx.channel.id if getattr(ctx, "channel", None) else None
        if channel_id is None:
            await safe_reply(ctx, content=tag_error_text("Could not determine channel_id."))
            return

        await defer_interaction(ctx)
        display = await self._resolve_display_name(ticker, display)

        entry = {
            "ticker": ticker,
            "display": display,
            "channel_id": int(channel_id),
            "threshold_pct": float(threshold_pct),
            "check_every_s": int(check_every_s),
            "enabled": True,
            "last_price": None,
            "last_ts": None,
        }

        async with self._watch_lock:
            entries: list[dict[str, Any]] = list(self._watch.get("entries", []))
            entries = [
                e
                for e in entries
                if not (
                    e.get("ticker") == ticker
                    and int(e.get("channel_id")) == int(channel_id)
                )
            ]
            entries.append(entry)
            self._watch["entries"] = entries
            self._save_watch()

        embed = discord.Embed(
            title="Watch added",
            description=(
                f"{display} ({ticker}) -> channel {channel_id}\n"
                f"threshold {threshold_pct:.2f}% | every {check_every_s}s"
            ),
            color=0x2B90D9,
        )
        await _send_with_json(
            ctx,
            payload={"action": "watch_add", "entry": entry},
            meta={"ok": True, "action": "watch_add", "ticker": ticker},
            embed=embed,
        )

    async def _watch_remove(
        self,
        ctx: commands.Context,
        symbol: str,
        *,
        channel_id: int | None,
        remove_all: bool,
        region: str,
        lookup_kind: str,
    ) -> None:
        try:
            ticker, display = await self._resolve_symbol(
                symbol, region=region, lookup_kind=lookup_kind
            )
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        if channel_id is None:
            channel_id = ctx.channel.id if getattr(ctx, "channel", None) else None

        await defer_interaction(ctx)
        display = await self._resolve_display_name(ticker, display)

        async with self._watch_lock:
            entries: list[dict[str, Any]] = list(self._watch.get("entries", []))
            before = len(entries)
            if remove_all or channel_id is None:
                entries = [e for e in entries if e.get("ticker") != ticker]
            else:
                entries = [
                    e
                    for e in entries
                    if not (
                        e.get("ticker") == ticker
                        and int(e.get("channel_id")) == int(channel_id)
                    )
                ]
            self._watch["entries"] = entries
            self._save_watch()

        embed = discord.Embed(
            title="Watch removed",
            description=f"{display} ({ticker}) ({before} -> {len(entries)})",
            color=0x2B90D9,
        )
        await _send_with_json(
            ctx,
            payload={
                "action": "watch_remove",
                "ticker": ticker,
                "display": display,
                "channel_id": channel_id,
                "remove_all": remove_all,
            },
            meta={"ok": True, "action": "watch_remove", "ticker": ticker},
            embed=embed,
        )

    async def _watch_list(self, ctx: commands.Context) -> None:
        async with self._watch_lock:
            entries: list[dict[str, Any]] = list(self._watch.get("entries", []))

        if not entries:
            embed = discord.Embed(
                title="Watchlist",
                description="watchlist is empty.",
                color=0x2B90D9,
            )
            await _send_with_json(
                ctx,
                payload={"action": "watch_list", "entries": []},
                meta={"ok": True, "action": "watch_list"},
                embed=embed,
            )
            return

        lines = []
        for e in entries[:120]:
            lines.append(
                f"- {e.get('display')} ({e.get('ticker')}) "
                f"ch={e.get('channel_id')} thr={e.get('threshold_pct')}% "
                f"every={e.get('check_every_s')}s enabled={e.get('enabled')}"
            )
        embed = discord.Embed(
            title="Watchlist",
            description="\n".join(lines),
            color=0x2B90D9,
        )
        await _send_with_json(
            ctx,
            payload={"action": "watch_list", "entries": entries},
            meta={"ok": True, "action": "watch_list"},
            embed=embed,
        )

    async def _monitor_loop(self) -> None:
        await self.bot.wait_until_ready()
        while not self.bot.is_closed():
            try:
                await self._monitor_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("finance monitor tick failed")
            await asyncio.sleep(5.0)

    async def _monitor_tick(self) -> None:
        async with self._watch_lock:
            entries: list[dict[str, Any]] = list(self._watch.get("entries", []))

        if not entries:
            return

        now = datetime.now(timezone.utc)
        due = []
        for e in entries:
            if not e.get("enabled", True):
                continue
            every = int(e.get("check_every_s") or 300)
            last_ts = e.get("last_ts")
            if not last_ts:
                due.append(e)
                continue
            try:
                last_dt = datetime.fromisoformat(last_ts)
            except Exception:
                due.append(e)
                continue
            if (now - last_dt).total_seconds() >= every:
                due.append(e)

        if not due:
            return

        tickers = sorted({str(e.get("ticker")) for e in due if e.get("ticker")})
        if not tickers:
            return

        def _sync_download():
            try:
                df = yf.download(
                    tickers=tickers,
                    period="1d",
                    interval="1m",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass

            return yf.download(
                tickers=tickers,
                period="5d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )

        df_all = await _to_thread(_sync_download)

        async with self._watch_lock:
            for e in self._watch.get("entries", []):
                t = e.get("ticker")
                if t not in tickers:
                    continue

                thr = float(e.get("threshold_pct") or 2.0)

                last_price = None
                try:
                    if isinstance(df_all.columns, pd.MultiIndex):
                        close = df_all[(t, "Close")].dropna()
                    else:
                        close = df_all["Close"].dropna()
                    if not close.empty:
                        last_price = float(close.iloc[-1])
                except Exception:
                    last_price = None

                prev_price = e.get("last_price")
                fire = False
                change_pct = None
                if last_price is not None and prev_price not in (None, 0):
                    try:
                        change_pct = (
                            (last_price - float(prev_price)) / float(prev_price) * 100.0
                        )
                        if abs(change_pct) >= thr:
                            fire = True
                    except Exception:
                        pass

                e["last_price"] = last_price
                e["last_ts"] = now.isoformat()

                if fire:
                    ch = self.bot.get_channel(int(e.get("channel_id")))
                    if ch:
                        try:
                            msg = (
                                f"Finance alert: {e.get('display')} ({t}) "
                                f"{_fmt_num(prev_price)} -> {_fmt_num(last_price)} "
                                f"({change_pct:.2f}%)"
                            )
                            await ch.send(msg)
                        except Exception:
                            pass

            self._save_watch()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Finance(bot))
