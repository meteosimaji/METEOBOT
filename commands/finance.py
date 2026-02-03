from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import aiohttp
import discord
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
            starter = {"symbols": [{"name": "トヨタ自動車(株)", "ticker": "7203.T"}]}
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

    def resolve_strict(self, raw: str) -> tuple[str, str]:
        """
        仕様:
        - raw が ticker っぽいなら ticker としてそのまま通す
        - それ以外は registry の name と完全一致のみ通す
        - /finance トヨタ みたいな曖昧入力は必ずエラー
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
    yfinance は同期I/Oなので to_thread で包む。
    .info は壊れやすいので fast_info 優先、info は best-effort。
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
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(s) <= max_len:
        return s
    truncated = s[: max_len - 3] + "..."
    return json.dumps(
        {"truncated": True, "preview": truncated},
        ensure_ascii=False,
        separators=(",", ":"),
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


def _make_json_file(obj: Any, filename: str) -> discord.File:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return discord.File(fp=BytesIO(data), filename=filename)


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
    filename = "finance_error.json" if error else "finance.json"
    json_file = _make_json_file(payload, filename=filename)
    out_files = list(files or [])
    out_files.append(json_file)
    prefix = "FINANCE_ERROR_JSON_FILE=" if error else "FINANCE_JSON_FILE="
    content = (
        f"{prefix}attachment://{filename}\nFINANCE_META="
        + _json_compact(meta, max_len=800)
    )
    await safe_reply(
        ctx,
        content=content,
        embed=embed,
        files=out_files,
        ephemeral=ephemeral,
    )


def _parse_kv_query(q: str) -> dict[str, str | list[str]]:
    parts = shlex.split(q)
    out: dict[str, str | list[str]] = {}
    symbol: str | None = None
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
        else:
            if symbol is None:
                symbol = part
            else:
                symbol = symbol + " " + part
    if symbol is not None:
        out["symbol"] = symbol
    return out


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

    fig = plt.figure(figsize=(10, 6), dpi=140)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    for xi, o, h, l, c in zip(x, df["Open"], df["High"], df["Low"], df["Close"]):
        ax.vlines(xi, l, h, linewidth=1.0)
        bottom = min(o, c)
        height = abs(c - o)
        if height == 0:
            height = max(h - l, 1e-9) * 0.02
        rect = plt.Rectangle((xi - 0.3, bottom), 0.6, height, fill=False, linewidth=1.0)
        ax.add_patch(rect)

    if "Volume" in df.columns:
        axv.bar(x, df["Volume"].fillna(0).values, width=0.6)

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
    close: pd.Series, *, horizon_days: int = 20, paths: int = 2000
) -> dict[str, Any]:
    close = close.dropna()
    if len(close) < 40:
        return {"ok": False, "reason": "not_enough_history"}

    logret = np.log(close).diff().dropna()
    mu = float(logret.mean())
    sigma = float(logret.std(ddof=1))
    s0 = float(close.iloc[-1])

    shocks = np.random.normal(loc=mu, scale=sigma, size=(paths, horizon_days))
    prices = s0 * np.exp(np.cumsum(shocks, axis=1))
    end = prices[:, -1]
    return {
        "ok": True,
        "s0": s0,
        "mu": mu,
        "sigma": sigma,
        "horizon_days": horizon_days,
        "paths": paths,
        "end_quantiles": {
            "p05": float(np.quantile(end, 0.05)),
            "p25": float(np.quantile(end, 0.25)),
            "p50": float(np.quantile(end, 0.50)),
            "p75": float(np.quantile(end, 0.75)),
            "p95": float(np.quantile(end, 0.95)),
        },
        "var95": float(np.quantile(s0 - end, 0.95)),
    }


def _plot_forecast_fan(
    close: pd.Series, *, horizon_days: int = 20, paths: int = 2000
) -> BytesIO:
    close = close.dropna()
    if len(close) < 40:
        raise ValueError("not_enough_history")

    logret = np.log(close).diff().dropna()
    mu = float(logret.mean())
    sigma = float(logret.std(ddof=1))
    s0 = float(close.iloc[-1])
    shocks = np.random.normal(loc=mu, scale=sigma, size=(paths, horizon_days))
    prices = s0 * np.exp(np.cumsum(shocks, axis=1))
    qs = np.quantile(prices, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

    fig = plt.figure(figsize=(9, 4), dpi=140)
    ax = fig.add_subplot(111)
    x = np.arange(1, horizon_days + 1)
    ax.plot(x, qs[2], label="Median", linewidth=1.5)
    ax.fill_between(x, qs[0], qs[4], color="blue", alpha=0.1, label="p05-p95")
    ax.fill_between(x, qs[1], qs[3], color="blue", alpha=0.2, label="p25-p75")
    ax.set_title("Forecast fan chart")
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


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

    def cog_unload(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()

    @commands.hybrid_command(
        name="finance",
        description="Stocks: quote/chart/news/watchlist/data (Yahoo Finance via yfinance).",
        help=(
            "Examples:\n"
            "  /finance symbol:7203.T action:summary\n"
            "  /finance symbol:トヨタ自動車(株) action:summary\n"
            "  /finance symbol:7203.T action:chart period:6mo interval:1d\n"
            "  /finance symbol:7203.T action:candle period:3mo interval:1d\n"
            "  /finance symbol:7203.T action:ta period:6mo interval:1d\n"
            "  /finance symbol:7203.T action:forecast horizon_days:20\n"
            "  /finance action:symbols\n"
            "  /finance action:search query:7203\n"
            "  /finance action:lookup query:Toyota kind:stock\n"
            "  /finance action:screener_local min_mcap:1e11 max_pe:20\n"
            "  /finance symbol:7203.T action:news limit:5\n"
            "  /finance symbol:7203.T action:watch_add threshold_pct:2 check_every_s:300\n"
            "  /finance symbol:7203.T action:data section:financials\n"
            f"  {BOT_PREFIX}finance 7203.T summary"
        ),
        extras={
            "category": "Tools",
            "pro": (
                "Strict symbol matching (exact registry name or ticker) plus "
                "best-effort data pulls from Yahoo Finance via yfinance, "
                "including chart output and machine-readable JSON."),
        },
    )
    async def finance(
        self,
        ctx: commands.Context,
        symbol: str | None = None,
        action: Literal[
            "summary",
            "quote",
            "chart",
            "candle",
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
        ] = "summary",
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
    ) -> None:
        if symbol and action == "summary" and symbol in {
            "summary",
            "quote",
            "chart",
            "candle",
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
        }:
            action = symbol  # type: ignore[assignment]
            symbol = None

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
            await self._send_quote_only(ctx, symbol)
            return
        if action in {"summary", "chart"}:
            await self._send_summary(
                ctx, symbol=symbol, period=period, interval=interval, auto_adjust=auto_adjust
            )
            return
        if action == "news":
            await self._send_news(ctx, symbol, limit=limit)
            return
        if action == "candle":
            await self._send_candle(ctx, symbol=symbol, period=period, interval=interval)
            return
        if action == "ta":
            await self._send_ta(ctx, symbol=symbol, period=period, interval=interval)
            return
        if action == "forecast":
            await self._send_forecast(
                ctx, symbol=symbol, period=period, interval=interval, horizon_days=horizon_days, paths=paths
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
            await self._send_data(ctx, symbol, section=section, max_rows=max_rows)
            return
        if action == "watch_add":
            await self._watch_add(
                ctx,
                symbol,
                channel_id=channel_id,
                threshold_pct=threshold_pct,
                check_every_s=check_every_s,
            )
            return
        if action == "watch_remove":
            await self._watch_remove(
                ctx, symbol, channel_id=channel_id, remove_all=remove_all
            )
            return
        if action == "watch_list":
            await self._watch_list(ctx)
            return

        await safe_reply(ctx, content=tag_error_text(f"Unknown action: {action}"))

    @commands.hybrid_command(
        name="financeq",
        description="Finance query (single-arg, tool-friendly).",
        help=(
            "Examples:\n"
            "  /financeq 7203.T action:candle period:6mo interval:1d\n"
            "  /financeq トヨタ自動車(株) action:ta\n"
            "  /financeq 7203 action:search\n"
            f"  {BOT_PREFIX}financeq 7203.T action:forecast horizon_days:20\n"
        ),
        extras={
            "category": "Tools",
            "pro": (
                "Single-argument finance query for bot_invoke: "
                "use key:value tokens to control action, period, interval, and more."
            ),
        },
    )
    async def financeq(self, ctx: commands.Context, *, q: str) -> None:
        kv = _parse_kv_query(q)
        symbol = str(kv.get("symbol") or "").strip() or None
        action = str(kv.get("action") or "summary").strip().lower()
        period = str(kv.get("period") or "6mo").strip()
        interval = str(kv.get("interval") or "1d").strip()
        query = str(kv.get("query") or "").strip() or None
        region = str(kv.get("region") or "JP").strip()

        def _as_int(val: Any, default: int) -> int:
            try:
                return int(val)
            except Exception:
                return default

        def _as_float(val: Any) -> float | None:
            try:
                return float(val)
            except Exception:
                return None

        limit = _as_int(kv.get("limit"), 5)
        horizon_days = _as_int(kv.get("horizon_days"), 20)
        paths = _as_int(kv.get("paths"), 2000)
        lookup_kind = str(kv.get("kind") or "all").strip().lower()
        min_mcap = _as_float(kv.get("min_mcap"))
        max_pe = _as_float(kv.get("max_pe"))

        await self.finance(
            ctx,
            symbol=symbol,
            action=action,  # type: ignore[arg-type]
            period=period,
            interval=interval,
            limit=limit,
            query=query,
            region=region,
            horizon_days=horizon_days,
            paths=paths,
            lookup_kind=lookup_kind,
            screener_min_mcap=min_mcap,
            screener_max_pe=max_pe,
        )

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
            desc = "Unknown symbol. Exact-match name required.\nSuggestions:\n" + "\n".join(
                names
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
        elif isinstance(exc, LookupError) and msg == "ticker_missing_suffix":
            err_code = "ticker_missing_suffix"
            desc = (
                "Ticker needs exchange suffix (e.g. 7203.T). "
                "Use /finance action:search query:7203 to list candidates."
            )
        elif isinstance(exc, LookupError):
            err_code = "symbol_unknown"
            desc = (
                "Unknown symbol. Use ticker (e.g. 7203.T) or exact registry "
                "name (e.g. トヨタ自動車(株))."
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

        def _sync():
            s = Search(query, max_results=limit, news_count=0, enable_fuzzy_query=True)
            return list(s.quotes or [])

        try:
            quotes = await _to_thread(_sync)
        except Exception:
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

    async def _send_quote_only(self, ctx: commands.Context, symbol: str) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        await defer_interaction(ctx)
        data = await self.yf.get_quote(ticker)
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
            "ticker": ticker,
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
    ) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        await defer_interaction(ctx)

        try:
            quote = await self.yf.get_quote(ticker)
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

        file = None
        try:
            buf = _plot_line_chart(hist, title=f"{display} ({ticker}) {period} / {interval}")
            file = discord.File(buf, filename="chart.png")
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
        embed.add_field(name="Auto adjust", value=str(bool(auto_adjust)), inline=True)

        payload = {
            "ticker": ticker,
            "display": display,
            "asof_jst": now_jst.isoformat(),
            "params": {
                "period": period,
                "interval": interval,
                "auto_adjust": auto_adjust,
            },
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
            embed.set_image(url="attachment://chart.png")
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

    async def _send_candle(
        self, ctx: commands.Context, *, symbol: str, period: str, interval: str
    ) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
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
        embed.set_image(url="attachment://candle.png")
        payload = {
            "ok": True,
            "action": "candle",
            "symbol_input": symbol,
            "ticker": ticker,
            "display": display,
            "params": {"period": period, "interval": interval},
        }
        await _send_with_json(
            ctx,
            payload=payload,
            meta={"ok": True, "action": "candle", "ticker": ticker},
            embed=embed,
            files=[file],
        )

    async def _send_ta(
        self, ctx: commands.Context, *, symbol: str, period: str, interval: str
    ) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
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
    ) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
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

        if hist.empty or "Close" not in hist.columns:
            await safe_reply(ctx, content=tag_error_text("No data for forecast."))
            return

        result = _mc_forecast(
            hist["Close"], horizon_days=max(1, int(horizon_days)), paths=max(100, int(paths))
        )
        if not result.get("ok"):
            await safe_reply(
                ctx, content=tag_error_text("Not enough history for forecast.")
            )
            return

        try:
            buf = _plot_forecast_fan(
                hist["Close"], horizon_days=max(1, int(horizon_days)), paths=max(100, int(paths))
            )
            file = discord.File(buf, filename="forecast.png")
        except Exception:
            file = None

        embed = discord.Embed(
            title=f"{display} ({ticker}) forecast",
            description=f"horizon={horizon_days} days paths={paths}",
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

        payload = {
            "ok": True,
            "action": "forecast",
            "symbol_input": symbol,
            "ticker": ticker,
            "display": display,
            "params": {"period": period, "interval": interval, "horizon_days": horizon_days, "paths": paths},
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

    async def _send_news(self, ctx: commands.Context, symbol: str, limit: int) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
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
        self, ctx: commands.Context, symbol: str, *, section: str, max_rows: int
    ) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
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
    ) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        if channel_id is None:
            channel_id = ctx.channel.id if getattr(ctx, "channel", None) else None
        if channel_id is None:
            await safe_reply(ctx, content=tag_error_text("Could not determine channel_id."))
            return

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
    ) -> None:
        try:
            ticker, display = self.reg.resolve_strict(symbol)
        except Exception as e:
            await self._send_symbol_error(ctx, symbol, e)
            return

        if channel_id is None:
            channel_id = ctx.channel.id if getattr(ctx, "channel", None) else None

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
