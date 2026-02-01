from __future__ import annotations

import asyncio
import contextlib
import html
from html.parser import HTMLParser
import inspect
import json
import logging
import os
import random
import re
import shlex
import shutil
import socket
import tempfile
import uuid
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse
from urllib.parse import urljoin
from urllib.request import HTTPRedirectHandler, Request, build_opener

import discord
from discord import app_commands
from discord.ext import commands
import yt_dlp

from utils import BOT_PREFIX, defer_interaction, error_embed, humanize_delta, safe_reply

log = logging.getLogger(__name__)

# ---- defaults (tuned for Discord attachment limits) ----

# Safety cap. Actual per-guild cap is ctx.guild.filesize_limit.
DEFAULT_MAX_BYTES = 200 * 1024 * 1024  # 200MiB
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_LOG_TAIL = 60
DEFAULT_WORK_ROOT = "data/savevideo"
DEFAULT_DNS_TIMEOUT_S = 2
DEFAULT_ERROR_TEXT_MAX = 1900
DISCORD_CONTENT_LIMIT = 4000
DEFAULT_HTML_FALLBACK_MAX_BYTES = 2_000_000
DEFAULT_X_SYNDICATION_ATTEMPTS = 3
DEFAULT_EXTERNAL_LINK_LIMIT_BYTES = 10 * 1024 * 1024
DEFAULT_EXTERNAL_LINK_TTL_S = 30 * 60
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
DEFAULT_AUDIO_EXTS = ("m4a", "mp4", "webm", "opus", "ogg")
DEFAULT_VIDEO_EXTS = ("mp4", "m4v")
SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "flac", "m4a", "opus", "ogg")
SUPPORTED_AUDIO_CODECS = {
    "wav": "wav",
    "mp3": "mp3",
    "flac": "flac",
    "m4a": "m4a",
    "opus": "opus",
    "ogg": "vorbis",
}
SUPPORTED_AUDIO_EXTS = {
    "wav": ("wav",),
    "mp3": ("mp3",),
    "flac": ("flac",),
    "m4a": ("m4a", "mp4"),
    "opus": ("opus", "ogg"),
    "ogg": ("ogg",),
}
DEFAULT_COOKIES_AUTO_BROWSER = "chrome"
DEFAULT_COOKIES_PATH = Path(__file__).resolve().parents[1] / "cookie" / "cookies.txt"

ERROR_PATTERNS: dict[str, re.Pattern[str]] = {
    "unsupported_url": re.compile(
        r"(unsupported\s+url|no\s+video\s+formats|no\s+compatible\s+formats|mp4[-\s]?compatible\s+formats)",
        re.I,
    ),
    "forbidden": re.compile(r"(403|forbidden|access\s+denied)", re.I),
    "rate_limited": re.compile(r"(429|rate\s*limit|too\s+many\s+requests)", re.I),
    "private": re.compile(r"(private|unavailable|not\s+available|deleted)", re.I),
    "login_required": re.compile(r"(login\s*required|sign\s*in|confirm\s+your\s+age)", re.I),
    "geo_restricted": re.compile(r"(not\s+available\s+in\s+your\s+country|geo\s*restricted)", re.I),
    "ffmpeg_missing": re.compile(r"(ffmpeg|avconv).*(not\s+found|not\s+installed)", re.I),
    "max_filesize": re.compile(r"(max(filesize)?|file\s+is\s+too\s+large|max\-filesize)", re.I),
}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_str(name: str) -> str | None:
    raw = os.getenv(name)
    raw = raw.strip() if isinstance(raw, str) else None
    return raw or None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _default_cookie_file() -> str | None:
    if DEFAULT_COOKIES_PATH.exists():
        return str(DEFAULT_COOKIES_PATH)
    return None


def _strip_ansi(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _normalize_audio_format(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError("Unsupported audio format.")
    return normalized


def _safe_filename(value: str, *, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^\w\s\-]+", "_", value, flags=re.UNICODE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "download"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned


def _parse_cookies_from_browser(
    spec: str | None,
) -> tuple[str] | tuple[str, str | None, str | None, str | None] | None:
    """Parse CLI-like cookies-from-browser into the tuple expected by yt-dlp."""
    spec = (spec or "").strip()
    if not spec:
        return None

    parts = spec.split(":")
    head = parts[0].strip()
    profile = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else None
    container = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None

    if "+" in head:
        browser, keyring = head.split("+", 1)
        browser = browser.strip()
        keyring = keyring.strip() or None
    else:
        browser, keyring = head.strip(), None

    if profile is None and keyring is None and container is None:
        return (browser,)

    if container is not None and keyring is None:
        return (browser, profile, None, container)

    if keyring is not None and container is None:
        return (browser, profile, keyring)

    if keyring is not None and container is not None:
        return (browser, profile, keyring, container)

    return (browser, profile, None, None)


def _is_blocked_ip_literal(host: str) -> bool:
    """Block obvious SSRF targets (localhost, RFC1918, link-local, etc.) for IP literals."""

    try:
        ip = ip_address(host)
    except ValueError:
        return host.lower() in {"localhost", "ip6-localhost"}

    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


async def _resolve_host_ips_async(host: str) -> set[str]:
    addrs: set[str] = set()
    infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    for family, _type, _proto, _canon, sockaddr in infos:
        if family == socket.AF_INET:
            addrs.add(sockaddr[0])
        elif family == socket.AF_INET6:
            addrs.add(sockaddr[0])
    return addrs


def _resolve_host_ips(host: str) -> set[str]:
    addrs: set[str] = set()
    for family, _type, _proto, _canon, sockaddr in socket.getaddrinfo(host, None):
        if family == socket.AF_INET:
            addrs.add(sockaddr[0])
        elif family == socket.AF_INET6:
            addrs.add(sockaddr[0])
    return addrs


def _browser_cookie_db_exists(browser: str) -> bool:
    name = browser.strip().lower()
    if not name:
        return False
    home = Path.home()
    if name in {"chrome", "google-chrome"}:
        bases = [home / ".config/google-chrome"]
    elif name == "chromium":
        bases = [home / ".config/chromium"]
    elif name in {"brave", "brave-browser"}:
        bases = [home / ".config/BraveSoftware/Brave-Browser"]
    elif name in {"edge", "microsoft-edge"}:
        bases = [home / ".config/microsoft-edge"]
    elif name == "firefox":
        profile_root = home / ".mozilla/firefox"
        if not profile_root.exists():
            return False
        return any(profile_root.rglob("cookies.sqlite"))
    else:
        return False

    for base in bases:
        if base.exists() and any(base.rglob("Cookies")):
            return True
    return False


def _is_safe_redirect_target(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    if _is_blocked_ip_literal(host):
        return False
    try:
        ips = _resolve_host_ips(host)
    except Exception:
        return False
    for ip_s in ips:
        if _is_blocked_ip_literal(ip_s):
            return False
    return True


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urljoin(req.full_url, newurl)
        if not _is_safe_redirect_target(target):
            raise RuntimeError(f"HTML fallback redirect blocked: {target}")
        return super().redirect_request(req, fp, code, msg, headers, target)


async def _is_safe_direct_url_async(url: str, *, dns_timeout_s: int) -> bool:
    """Allow only public http(s) targets."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False

    host = parsed.hostname
    if not host:
        return False

    if _is_blocked_ip_literal(host):
        return False

    try:
        ips = await asyncio.wait_for(_resolve_host_ips_async(host), timeout=float(dns_timeout_s))
    except Exception:
        return False

    for ip_s in ips:
        if _is_blocked_ip_literal(ip_s):
            return False

    return True


def _first_http_url(values: Sequence[object]) -> str | None:
    for value in values:
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def _parse_save_cli_args(
    raw: str,
    *,
    default_max_height: int | None,
    default_audio_only: bool,
    default_audio_focus: bool,
    default_item: int,
    default_force_url: bool,
    default_audio_format: str | None,
) -> tuple[SaveRequest | None, str | None]:
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return None, "Failed to parse arguments. Check quotes in the URL or flags."

    url: str | None = None
    max_height = default_max_height
    audio_only = default_audio_only
    audio_focus = default_audio_focus
    item = default_item
    force_url = default_force_url
    audio_format = default_audio_format
    unknown: list[str] = []

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--"):
            name, value = token[2:].split("=", 1) if "=" in token else (token[2:], None)
            name = name.strip().lower()

            def _consume_value() -> str | None:
                nonlocal idx
                if value is not None:
                    return value
                if idx + 1 >= len(tokens):
                    return None
                idx += 1
                return tokens[idx]

            if name in {"audio", "audio-only"}:
                audio_only = True if value is None else str(value).lower() not in {"0", "false", "no"}
            elif name in {"audio-focus", "audio-priority", "audio-quality"}:
                audio_focus = True if value is None else str(value).lower() not in {"0", "false", "no"}
            elif name in {"wav", "mp3", "flac", "m4a", "opus", "ogg"}:
                if audio_format and audio_format != name:
                    return None, "Only one audio format flag can be set."
                audio_only = True
                audio_format = name
            elif name in {"max-height", "height", "resolution"}:
                raw_value = _consume_value()
                if raw_value is None:
                    return None, "Missing value for --max-height/--height."
                try:
                    max_height = int(raw_value)
                except ValueError:
                    return None, "Height must be a number."
                if max_height <= 0:
                    return None, "Height must be a positive number."
            elif name in {"item", "index"}:
                raw_value = _consume_value()
                if raw_value is None:
                    return None, "Missing value for --item/--index."
                try:
                    item = int(raw_value)
                except ValueError:
                    return None, "Item index must be a number."
                if item <= 0:
                    return None, "Item index must be 1 or higher."
            elif name in {"url", "link", "external"}:
                force_url = True if value is None else str(value).lower() not in {
                    "0",
                    "false",
                    "no",
                }
            else:
                unknown.append(token)
        else:
            if url is None:
                url = token
            else:
                unknown.append(token)
        idx += 1

    if unknown:
        return None, f"Unknown arguments: {' '.join(unknown)}"

    if not url:
        return None, "Provide a video URL to save."
    if audio_format and not audio_only:
        return None, "Audio format flags require --audio."

    return SaveRequest(
        url=url,
        max_height=max_height,
        audio_only=audio_only,
        audio_focus=audio_focus,
        item=item,
        force_url=force_url,
        audio_format=audio_format,
    ), None


def _pick_downloaded_file(
    download_dir: Path,
    *,
    prefer_exts: Iterable[str] | None = None,
) -> Path | None:
    files = [path for path in download_dir.iterdir() if path.is_file()]
    if not files:
        return None
    prefer_set = {ext.lower().lstrip(".") for ext in (prefer_exts or []) if ext}
    preferred = [path for path in files if path.suffix.lower().lstrip(".") in prefer_set]
    # prefer preferred extensions if present, else the largest file.
    candidates = preferred or files
    return max(candidates, key=lambda path: path.stat().st_size)


class YTDLLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def debug(self, message: str) -> None:
        self._append(message)

    def info(self, message: str) -> None:
        self._append(message)

    def warning(self, message: str) -> None:
        self._append(message)

    def error(self, message: str) -> None:
        self._append(message)

    def _append(self, message: str) -> None:
        if message:
            self.lines.append(str(message))

    def tail(self, count: int) -> str:
        if count <= 0:
            return ""
        return "\n".join(self.lines[-count:])


@dataclass(frozen=True)
class SaveConfig:
    max_bytes: int
    timeout_s: int
    max_concurrent: int
    log_tail_lines: int
    work_root: Path
    dns_timeout_s: int
    user_agent: str
    cookies_file: str | None
    cookies_from_browser: tuple[str] | tuple[str, str | None, str | None, str | None] | None
    impersonate: str | None
    cookies_auto: bool
    cookies_auto_browser: str

    @staticmethod
    def from_env() -> "SaveConfig":
        return SaveConfig(
            max_bytes=_env_int("SAVEVIDEO_MAX_BYTES", DEFAULT_MAX_BYTES),
            timeout_s=_env_int("SAVEVIDEO_TIMEOUT_S", DEFAULT_TIMEOUT_S),
            max_concurrent=_env_int("SAVEVIDEO_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT),
            log_tail_lines=_env_int("SAVEVIDEO_LOG_TAIL_LINES", DEFAULT_LOG_TAIL),
            work_root=Path(os.getenv("SAVEVIDEO_WORK_DIR", DEFAULT_WORK_ROOT)),
            dns_timeout_s=_env_int("SAVEVIDEO_DNS_TIMEOUT_S", DEFAULT_DNS_TIMEOUT_S),
            user_agent=os.getenv("SAVEVIDEO_USER_AGENT", DEFAULT_USER_AGENT),
            cookies_file=_env_str("SAVEVIDEO_COOKIES_FILE") or _default_cookie_file(),
            cookies_from_browser=_parse_cookies_from_browser(
                _env_str("SAVEVIDEO_COOKIES_FROM_BROWSER")
            ),
            impersonate=_env_str("SAVEVIDEO_IMPERSONATE"),
            cookies_auto=False,
            cookies_auto_browser=_env_str("SAVEVIDEO_COOKIES_AUTO_BROWSER")
            or DEFAULT_COOKIES_AUTO_BROWSER,
        )


@dataclass(frozen=True)
class SaveRequest:
    url: str
    max_height: int | None
    audio_only: bool
    audio_focus: bool
    item: int
    force_url: bool
    audio_format: str | None


def _human_size(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"


def _truncate_text(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    suffix = "...(truncated)"
    keep = max(max_len - len(suffix), 0)
    return f"{text[:keep]}{suffix}"


def _escape_markdown_link_label(text: str) -> str:
    return text.replace("\\", "＼").replace("[", "［").replace("]", "］")


def _assemble_error_message(*, base: str, tail: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    tail = tail.strip()
    if not tail:
        return _truncate_text(base, max_len)

    tail_prefix = "\n```text\n"
    tail_suffix = "\n```"
    trunc_suffix = "\n...(truncated)"

    headroom = max_len - len(base) - len(tail_prefix) - len(tail_suffix)
    if headroom <= 0:
        return _truncate_text(base, max_len)

    if len(tail) <= headroom:
        return f"{base}{tail_prefix}{tail}{tail_suffix}"

    trimmed_tail = tail[: max(headroom - len(trunc_suffix), 0)]
    return f"{base}{tail_prefix}{trimmed_tail}{trunc_suffix}{tail_suffix}"


def _cap_error_message(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""

    return _truncate_text(text, max_len)


def _normalize_impersonate_opt(value: object, logger: "YTDLLogger") -> object | None:
    """Normalize yt-dlp 'impersonate' option for Python API compatibility."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            from yt_dlp.networking.impersonate import ImpersonateTarget  # type: ignore
        except Exception:
            return raw
        try:
            import curl_cffi  # type: ignore  # noqa: F401
        except Exception as exc:
            logger.debug("impersonate disabled (curl-cffi missing): %r", exc)
            return None
        try:
            return ImpersonateTarget.from_str(raw.lower())
        except Exception as exc:
            logger.debug("impersonate disabled (invalid target %r): %r", raw, exc)
            return None
    return value


def _extra_headers(url: str, user_agent: str) -> dict[str, str]:
    headers = {"User-Agent": user_agent}
    host = urlparse(url).hostname or ""
    if "tiktok.com" in host:
        # cobaltもここを強めにやってる。Refererが無いと弾かれることがある。
        headers["Referer"] = "https://www.tiktok.com/"
    if host.endswith("x.com") or host.endswith("twitter.com"):
        headers["Referer"] = "https://x.com/"
    return headers


def _fetch_url_text(url: str, headers: dict[str, str], *, timeout_s: int) -> tuple[str, str]:
    opener = build_opener(_SafeRedirectHandler())
    req = Request(url, headers=headers)
    with opener.open(req, timeout=timeout_s) as resp:
        final_url = resp.geturl()
        content_type = resp.headers.get("Content-Type", "")
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip() or "utf-8"
        data = resp.read(DEFAULT_HTML_FALLBACK_MAX_BYTES + 1)
    if len(data) > DEFAULT_HTML_FALLBACK_MAX_BYTES:
        data = data[:DEFAULT_HTML_FALLBACK_MAX_BYTES]
    return data.decode(charset, errors="replace"), final_url


def _extract_meta_urls(html_text: str) -> list[str]:
    html_text = html.unescape(html_text)
    patterns = [
        r'<meta[^>]+property=["\']og:video(?::secure_url)?["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+name=["\']twitter:player:stream["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+property=["\']twitter:player:stream["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+property=["\']og:video:url["\'][^>]+content=["\']([^"\']+)',
        r'<video[^>]+src=["\']([^"\']+)',
    ]
    urls: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, html_text, re.IGNORECASE):
            url = match.group(1).strip()
            if url.startswith("//"):
                url = f"https:{url}"
            if url.startswith(("http://", "https://")):
                urls.append(url)
    for match in re.finditer(r'https?://[^"\'\s>]+\.(?:mp4|m3u8)(?:\?[^"\'\s>]*)?', html_text):
        urls.append(match.group(0))
    return urls


def _extract_x_syndication_urls(text: str) -> list[tuple[str, int, int]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    media = data.get("mediaDetails") or data.get("media")
    if not isinstance(media, list):
        return []
    results: list[tuple[str, int, int]] = []
    for item in media:
        if not isinstance(item, dict):
            continue
        video_info = item.get("video_info")
        if not isinstance(video_info, dict):
            continue
        variants = video_info.get("variants")
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            url = variant.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            bitrate = variant.get("bitrate") or 0
            try:
                bitrate_val = int(bitrate)
            except (ValueError, TypeError):
                bitrate_val = 0
            ext_priority = 0
            lowered = url.lower()
            if ".mp4" in lowered:
                ext_priority = 2
            elif ".m3u8" in lowered:
                ext_priority = 1
            results.append((url, bitrate_val, ext_priority))
    return results


def _extract_x_fallback_urls(
    url: str,
    headers: dict[str, str],
    *,
    timeout_s: int,
) -> list[str]:
    match = re.search(r"/status/(\d+)", url)
    if not match:
        return []
    tweet_id = match.group(1)
    variants: list[tuple[str, int, int]] = []
    for _ in range(DEFAULT_X_SYNDICATION_ATTEMPTS):
        token = str(random.randint(10_000_000, 99_999_999))
        syndication_url = (
            f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&lang=en&token={token}"
        )
        try:
            text, _ = _fetch_url_text(syndication_url, headers, timeout_s=timeout_s)
        except Exception:
            continue
        variants = _extract_x_syndication_urls(text)
        if variants:
            break
    variants.sort(key=lambda item: (item[2], item[1]), reverse=True)
    return [url for url, _, _ in variants]


def _collect_html_fallback_urls(
    url: str,
    headers: dict[str, str],
    *,
    timeout_s: int,
) -> list[str]:
    html_text, _ = _fetch_url_text(url, headers, timeout_s=timeout_s)
    urls = _extract_meta_urls(html_text)
    seen: set[str] = set()
    unique = []
    for candidate in urls:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


class _MetaRefreshParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refresh_content: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.refresh_content is not None or tag.lower() != "meta":
            return
        attributes = {key.lower(): value for key, value in attrs if key}
        http_equiv = attributes.get("http-equiv")
        if not http_equiv or http_equiv.lower() != "refresh":
            return
        content = attributes.get("content")
        if content:
            self.refresh_content = content


def _extract_meta_refresh_url(html_text: str) -> str | None:
    parser = _MetaRefreshParser()
    parser.feed(html_text)
    content = parser.refresh_content
    if not content:
        return None
    content = html.unescape(content)
    for part in content.split(";"):
        match = re.search(r"url\s*=\s*(.+)", part, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("'\"")
    return None


def _extract_js_redirect_url(html_text: str) -> str | None:
    html_text = html.unescape(html_text)
    patterns = [
        r"location\\.replace\\(['\\\"]([^'\\\"]+)['\\\"]\\)",
        r"location\\.href\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


async def _resolve_url_for_policy(
    url: str,
    headers: dict[str, str],
    *,
    timeout_s: int,
    dns_timeout_s: int,
    max_hops: int = 5,
) -> str:
    current_url = url
    for _ in range(max_hops):
        html_text, final_url = await asyncio.to_thread(
            _fetch_url_text,
            current_url,
            headers,
            timeout_s=timeout_s,
        )
        if final_url != current_url:
            current_url = final_url
        meta_url = _extract_meta_refresh_url(html_text)
        js_url = _extract_js_redirect_url(html_text)
        next_url = meta_url or js_url
        if not next_url:
            return current_url
        resolved = urljoin(current_url, next_url)
        if not await _is_safe_direct_url_async(resolved, dns_timeout_s=dns_timeout_s):
            return current_url
        if resolved == current_url:
            return current_url
        current_url = resolved
    return current_url


def _base_ydl_opts(
    *,
    logger: YTDLLogger,
    work_dir: Path,
    max_bytes: int,
    headers: dict[str, str],
    cookies_file: str | None,
    cookies_from_browser: tuple[str] | tuple[str, str | None, str | None, str | None] | None,
    socket_timeout: int,
    impersonate: object | None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "noplaylist": True,
        "restrictfilenames": True,
        "logger": logger,
        "max_filesize": max_bytes,
        "merge_output_format": "mp4",
        "http_headers": headers,
        "outtmpl": {"default": str(work_dir / "%(title).200B-%(id)s.%(ext)s")},
        # ffmpegがあるなら勝手に良い感じにマルチスレ化されるが、断片DLの並列数を少し上げる。
        "concurrent_fragment_downloads": 4,
        "socket_timeout": socket_timeout,
        "retries": 3,
        "fragment_retries": 3,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        # e.g. "chrome", "firefox" (yt-dlpの仕様)
        opts["cookiesfrombrowser"] = cookies_from_browser
    norm_impersonate = _normalize_impersonate_opt(impersonate, logger)
    if norm_impersonate is not None:
        opts["impersonate"] = norm_impersonate
    return opts


def _probe_info(
    url: str,
    logger: YTDLLogger,
    headers: dict[str, str],
    cookies_file: str | None,
    cookies_from_browser: tuple[str] | tuple[str, str | None, str | None, str | None] | None,
    impersonate: object | None,
    extra_opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "noplaylist": True,
        "logger": logger,
        "skip_download": True,
        "http_headers": headers,
    }
    if extra_opts:
        opts.update(extra_opts)
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = cookies_from_browser
    norm_impersonate = _normalize_impersonate_opt(impersonate, logger)
    if norm_impersonate is not None:
        opts["impersonate"] = norm_impersonate
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise RuntimeError("yt-dlp returned unexpected info")
    return info


def _estimate_format_size(fmt: dict[str, Any], duration_s: float | None) -> int | None:
    size = fmt.get("filesize") or fmt.get("filesize_approx")
    if isinstance(size, (int, float)) and size > 0:
        return int(size)

    # bitrate (tbr) is in Kbps
    tbr = fmt.get("tbr")
    if duration_s and isinstance(tbr, (int, float)) and tbr > 0:
        return int(float(duration_s) * float(tbr) * 1000.0 / 8.0)

    # fallback: vbr/abr
    vbr = fmt.get("vbr")
    abr = fmt.get("abr")
    if duration_s and isinstance(vbr, (int, float)) and isinstance(abr, (int, float)):
        br = float(vbr) + float(abr)
        if br > 0:
            return int(float(duration_s) * br * 1000.0 / 8.0)

    return None


def _is_mp4_video(fmt: dict[str, Any]) -> bool:
    if (fmt.get("vcodec") or "none") == "none":
        return False
    ext = (fmt.get("ext") or "").lower()
    return ext in {"mp4", "m4v"}


def _is_mp4_h264ish_video(fmt: dict[str, Any]) -> bool:
    if (fmt.get("vcodec") or "none") == "none":
        return False
    ext = (fmt.get("ext") or "").lower()
    if ext != "mp4":
        return False
    vcodec = (fmt.get("vcodec") or "").lower()
    # avc1 (H.264) が最強に互換性高い。hevc (hvc1/hev1) も一応OK。
    return (
        vcodec.startswith("avc1")
        or ("h264" in vcodec)
        or vcodec.startswith("hvc1")
        or vcodec.startswith("hev1")
    )


def _is_hls_video(fmt: dict[str, Any]) -> bool:
    vcodec = (fmt.get("vcodec") or "none").lower()
    if vcodec == "none":
        return False
    ext = (fmt.get("ext") or "").lower()
    protocol = (fmt.get("protocol") or "").lower()
    # ts を雑に拾うと関係ない ts を掴む可能性があるので、m3u8 系に寄せる
    return ext == "m3u8" or protocol.startswith("m3u8")


def _is_hls_audio(fmt: dict[str, Any]) -> bool:
    if (fmt.get("vcodec") or "none").lower() != "none":
        return False
    if (fmt.get("acodec") or "none").lower() == "none":
        return False
    ext = (fmt.get("ext") or "").lower()
    protocol = (fmt.get("protocol") or "").lower()
    return ext == "m3u8" or protocol.startswith("m3u8")


def _is_m4a_aacish_audio(fmt: dict[str, Any]) -> bool:
    if (fmt.get("acodec") or "none") == "none":
        return False
    if (fmt.get("vcodec") or "none") != "none":
        return False
    ext = (fmt.get("ext") or "").lower()
    if ext not in {"m4a", "mp4"}:
        return False
    acodec = (fmt.get("acodec") or "").lower()
    return acodec.startswith("mp4a") or ("aac" in acodec)


def _is_audio_only(fmt: dict[str, Any]) -> bool:
    return (fmt.get("vcodec") or "none") == "none" and (fmt.get("acodec") or "none") != "none"


def _iter_mp4_candidates(
    info: dict[str, Any],
    *,
    max_bytes: int,
    max_height: int | None,
    allow_any_mp4: bool = False,
    allow_hls_fallback: bool = False,
) -> Iterable[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]]:
    """Yield candidates as (format_spec, vfmt, afmt, est_size).

    - format_spec is either "id" (muxed) or "vid+aud".
    - vfmt is the chosen video format dict (for muxed it is the whole format).
    - afmt is None for muxed, otherwise audio format dict.
    - est_size may be None if we can't estimate.
    """

    duration = info.get("duration")
    duration_s = float(duration) if isinstance(duration, (int, float)) and duration > 0 else None

    formats: list[dict[str, Any]] = [f for f in (info.get("formats") or []) if isinstance(f, dict)]

    # Partition formats.
    muxed: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    audios: list[dict[str, Any]] = []

    for f in formats:
        # Ignore story/unknown formats.
        if not f.get("format_id"):
            continue

        if _is_mp4_h264ish_video(f) and (f.get("acodec") or "none") != "none":
            # muxed mp4 with audio
            muxed.append(f)
        elif _is_mp4_h264ish_video(f):
            videos.append(f)
        elif _is_m4a_aacish_audio(f):
            audios.append(f)
        elif allow_hls_fallback and _is_hls_video(f):
            if (f.get("acodec") or "none") != "none":
                muxed.append(f)
            else:
                videos.append(f)
        elif allow_hls_fallback and _is_hls_audio(f):
            audios.append(f)
        elif allow_any_mp4 and _is_mp4_video(f) and (f.get("acodec") or "none") != "none":
            muxed.append(f)
        elif allow_any_mp4 and _is_mp4_video(f):
            videos.append(f)

    # Apply max_height filter.
    if max_height is not None:
        muxed = [f for f in muxed if (f.get("height") or 0) <= max_height]
        videos = [f for f in videos if (f.get("height") or 0) <= max_height]

    # Sort with a bias toward higher res / higher bitrate.
    def _vkey(f: dict[str, Any]) -> tuple[int, float]:
        return (int(f.get("height") or 0), float(f.get("tbr") or 0.0))

    def _akey(f: dict[str, Any]) -> float:
        return float(f.get("abr") or f.get("tbr") or 0.0)

    muxed.sort(key=_vkey, reverse=True)
    videos.sort(key=_vkey, reverse=True)
    audios.sort(key=_akey, reverse=True)

    # Limit combinatorics.
    videos = videos[:30]
    audios = audios[:10]

    # 1) muxed candidates
    for f in muxed:
        est = _estimate_format_size(f, duration_s)
        yield (str(f["format_id"]), f, None, est)

    # 2) video+audio candidates
    for v in videos:
        for a in audios:
            est_v = _estimate_format_size(v, duration_s)
            est_a = _estimate_format_size(a, duration_s)
            est = est_v + est_a if isinstance(est_v, int) and isinstance(est_a, int) else None
            yield (f"{v['format_id']}+{a['format_id']}", v, a, est)


def _rank_candidate(
    vfmt: dict[str, Any],
    afmt: dict[str, Any] | None,
    est_size: int | None,
    *,
    audio_focus: bool,
) -> tuple:
    """Higher is better."""
    height = int(vfmt.get("height") or 0)
    fps = float(vfmt.get("fps") or 0.0)
    tbr = float(vfmt.get("tbr") or 0.0)

    # codec preference: avc1 > h264-other > hevc
    vcodec = (vfmt.get("vcodec") or "").lower()
    codec_score = 0
    if vcodec.startswith("avc1"):
        codec_score = 3
    elif "h264" in vcodec:
        codec_score = 2
    elif vcodec.startswith("hvc1") or vcodec.startswith("hev1"):
        codec_score = 1

    audio_score = 0
    if afmt is None:
        acodec = (vfmt.get("acodec") or "").lower()
    else:
        acodec = (afmt.get("acodec") or "").lower()
    if acodec.startswith("mp4a") or "aac" in acodec:
        audio_score = 1

    audio_br = 0.0
    if afmt is not None:
        audio_br = float(afmt.get("abr") or afmt.get("tbr") or 0.0)
    else:
        audio_br = float(vfmt.get("abr") or vfmt.get("tbr") or 0.0)

    # Prefer known-under-limit sizes; unknown sizes get a penalty.
    size_known = 1 if isinstance(est_size, int) else 0

    # Bigger est_size is usually better quality *if* still under limit, but we don't want to hug the limit.
    # We'll use a small nudge rather than a main axis.
    size_nudge = int(est_size or 0)

    if audio_focus:
        return (audio_score, audio_br, codec_score, size_known, height, fps, tbr, size_nudge)
    return (codec_score, audio_score, size_known, height, fps, tbr, size_nudge)


def _pick_best_formats(
    info: dict[str, Any],
    *,
    max_bytes: int,
    max_height: int | None,
    audio_focus: bool,
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]]:
    """Return candidates in best-first order.

    We only keep candidates whose *estimated* size is under max_bytes (with a small safety margin).
    Unknown-size candidates are kept as a last resort.
    """

    safety = int(max_bytes * 0.97)

    def _collect_candidates(allow_any_mp4: bool, allow_hls_fallback: bool) -> list[
        tuple[str, dict[str, Any], dict[str, Any] | None, int | None]
    ]:
        in_limit: list[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]] = []
        unknown: list[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]] = []
        seen: set[str] = set()
        for spec, vfmt, afmt, est in _iter_mp4_candidates(
            info,
            max_bytes=max_bytes,
            max_height=max_height,
            allow_any_mp4=allow_any_mp4,
            allow_hls_fallback=allow_hls_fallback,
        ):
            if spec in seen:
                continue
            seen.add(spec)
            if isinstance(est, int):
                if est <= safety:
                    in_limit.append((spec, vfmt, afmt, est))
            else:
                unknown.append((spec, vfmt, afmt, est))

        in_limit.sort(
            key=lambda t: _rank_candidate(t[1], t[2], t[3], audio_focus=audio_focus),
            reverse=True,
        )
        unknown.sort(
            key=lambda t: _rank_candidate(t[1], t[2], t[3], audio_focus=audio_focus),
            reverse=True,
        )

        # Try known-good candidates first, then unknown.
        return in_limit + unknown

    primary = _collect_candidates(False, False)
    if primary:
        return primary
    secondary = _collect_candidates(True, False)
    if secondary:
        return secondary
    return _collect_candidates(True, True)


def _iter_audio_candidates(
    info: dict[str, Any],
    *,
    max_bytes: int,
) -> Iterable[tuple[str, dict[str, Any], int | None]]:
    duration = info.get("duration")
    duration_s = float(duration) if isinstance(duration, (int, float)) and duration > 0 else None

    formats: list[dict[str, Any]] = [f for f in (info.get("formats") or []) if isinstance(f, dict)]
    audio_formats = [f for f in formats if f.get("format_id") and _is_audio_only(f)]

    def _audio_key(fmt: dict[str, Any]) -> tuple[int, float]:
        ext = (fmt.get("ext") or "").lower()
        ext_score = 2 if ext in {"m4a", "mp4"} else 1 if ext in {"webm", "opus", "ogg"} else 0
        abr = float(fmt.get("abr") or fmt.get("tbr") or 0.0)
        return (ext_score, abr)

    audio_formats.sort(key=_audio_key, reverse=True)

    for fmt in audio_formats:
        est = _estimate_format_size(fmt, duration_s)
        yield (str(fmt["format_id"]), fmt, est)


def _pick_best_audio_formats(
    info: dict[str, Any],
    *,
    max_bytes: int,
) -> list[tuple[str, dict[str, Any], int | None]]:
    safety = int(max_bytes * 0.97)
    in_limit: list[tuple[str, dict[str, Any], int | None]] = []
    unknown: list[tuple[str, dict[str, Any], int | None]] = []
    seen: set[str] = set()

    for spec, fmt, est in _iter_audio_candidates(info, max_bytes=max_bytes):
        if spec in seen:
            continue
        seen.add(spec)
        if isinstance(est, int):
            if est <= safety:
                in_limit.append((spec, fmt, est))
        else:
            unknown.append((spec, fmt, est))

    def _rank(fmt: dict[str, Any], est: int | None) -> tuple:
        ext = (fmt.get("ext") or "").lower()
        ext_score = 2 if ext in {"m4a", "mp4"} else 1 if ext in {"webm", "opus", "ogg"} else 0
        abr = float(fmt.get("abr") or fmt.get("tbr") or 0.0)
        size_known = 1 if isinstance(est, int) else 0
        return (ext_score, abr, size_known, int(est or 0))

    in_limit.sort(key=lambda t: _rank(t[1], t[2]), reverse=True)
    unknown.sort(key=lambda t: _rank(t[1], t[2]), reverse=True)

    return in_limit + unknown


def _download_with_spec(
    url: str,
    *,
    format_spec: str,
    max_bytes: int,
    logger: YTDLLogger,
    work_dir: Path,
    headers: dict[str, str],
    cookies_file: str | None,
    cookies_from_browser: tuple[str] | tuple[str, str | None, str | None, str | None] | None,
    socket_timeout: int,
    impersonate: object | None,
    prefer_exts: Sequence[str] | None = None,
    extra_opts: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    opts = _base_ydl_opts(
        logger=logger,
        work_dir=work_dir,
        max_bytes=max_bytes,
        headers=headers,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
        socket_timeout=socket_timeout,
        impersonate=impersonate,
    )
    opts["format"] = format_spec
    if extra_opts:
        opts.update(extra_opts)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if not isinstance(info, dict):
        raise RuntimeError("yt-dlp returned unexpected info")

    chosen = _pick_downloaded_file(work_dir, prefer_exts=prefer_exts)
    if chosen is None:
        raise RuntimeError("Download completed but no files were written")

    return info, chosen


def _classify_error(message: str) -> str:
    for key, pattern in ERROR_PATTERNS.items():
        if pattern.search(message):
            return key
    return "unknown"


class Save(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.cfg = SaveConfig.from_env()
        self._sem = asyncio.Semaphore(self.cfg.max_concurrent)

    async def _create_external_download_link(
        self,
        ctx: commands.Context,
        download_path: Path,
        *,
        filename: str,
    ) -> str | None:
        ask_cog = self.bot.get_cog("Ask")
        if not ask_cog or not hasattr(ask_cog, "register_download"):
            return None
        target_dir = self.cfg.work_root / "downloads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"{uuid.uuid4().hex}_{filename}"
        target_path = target_dir / target_name
        try:
            shutil.move(str(download_path), target_path)
        except Exception:
            return None
        link = await ask_cog.register_download(
            target_path,
            filename=filename,
            expires_s=DEFAULT_EXTERNAL_LINK_TTL_S,
        )
        if not link:
            with contextlib.suppress(Exception):
                target_path.unlink()
            return None
        return link

    @commands.command(
        name="save",
        description="Save a video or audio from a public URL and attach it.",
        help=(
            "Download a video from a public URL (TikTok, YouTube, etc.) with yt-dlp "
            "and attach it to Discord. The bot auto-selects an mp4-compatible format "
            "and can honor a max-height cap you provide.\n\n"
            "**Usage**: `/save <url>`\n"
            "**Optional**: `/save <url> max_height:<int> audio_only:<bool> audio_focus:<bool> item:<int> "
            "force_url:<bool> audio_format:<str>`\n"
            "**Prefix flags**: `--audio`, `--audio-focus`, `--max-height 720`, `--item 2`, `--url`, "
            "`--wav`, `--mp3`, `--flac`, `--m4a`, `--opus`, `--ogg`\n\n"
            "Env knobs: SAVEVIDEO_MAX_BYTES, SAVEVIDEO_TIMEOUT_S, SAVEVIDEO_MAX_CONCURRENT, "
            "SAVEVIDEO_LOG_TAIL_LINES, SAVEVIDEO_WORK_DIR, SAVEVIDEO_DNS_TIMEOUT_S, "
            "SAVEVIDEO_USER_AGENT, SAVEVIDEO_COOKIES_FILE, SAVEVIDEO_COOKIES_FROM_BROWSER, "
            "SAVEVIDEO_COOKIES_AUTO, SAVEVIDEO_COOKIES_AUTO_BROWSER, SAVEVIDEO_IMPERSONATE\n\n"
            f"Prefix: `{BOT_PREFIX}save <url> [--audio] [--audio-focus] [--max-height <h>] "
            f"[--item <n>] [--url] [--wav|--mp3|--flac|--m4a|--opus|--ogg]`"
        ),
        extras={
            "category": "Tools",
            "pro": (
                "Fetches a single video (or audio-only) with yt-dlp, blocks private/localhost "
                "targets, and honors optional max-height caps."
            ),
            "destination": "Download a public video or audio-only URL and attach it to Discord.",
            "plus": (
                "Supports audio-only mode, audio-focused ranking, max-height caps, and item "
                "selection for carousel/playlist-style URLs. Use --url to return a download link."
            ),
        },
    )
    async def save(self, ctx: commands.Context, *, args: str) -> None:
        parsed, error = _parse_save_cli_args(
            args,
            default_max_height=None,
            default_audio_only=False,
            default_audio_focus=False,
            default_item=1,
            default_force_url=False,
            default_audio_format=None,
        )
        if error:
            await safe_reply(
                ctx,
                embed=error_embed(desc=error),
                mention_author=False,
                ephemeral=True,
            )
            return
        if parsed is None:
            await safe_reply(
                ctx,
                embed=error_embed(desc="Provide a video URL to save."),
                mention_author=False,
                ephemeral=True,
            )
            return
        await self._save_impl(ctx, parsed)

    @app_commands.command(
        name="save",
        description="Save a video or audio from a public URL and attach it.",
    )
    @app_commands.describe(
        url="Public video URL to download.",
        max_height="Optional max height cap.",
        audio_only="Download audio-only instead of video.",
        audio_focus="Prefer higher audio bitrate when picking formats.",
        item="Select which item to download from a carousel/playlist-like URL.",
        force_url="Always return a download link instead of uploading to Discord.",
        audio_format="Audio format for audio-only (wav, mp3, flac, m4a, opus, ogg).",
    )
    async def save_slash(
        self,
        interaction: discord.Interaction,
        url: str,
        max_height: int | None = None,
        audio_only: bool = False,
        audio_focus: bool = False,
        item: int = 1,
        force_url: bool = False,
        audio_format: str | None = None,
    ) -> None:
        ctx_factory = getattr(commands.Context, "from_interaction", None)
        if ctx_factory is None:
            await interaction.response.send_message(
                "This command isn't available right now. Please try again later.", ephemeral=True
            )
            return
        ctx_candidate = ctx_factory(interaction)
        ctx = await ctx_candidate if inspect.isawaitable(ctx_candidate) else ctx_candidate
        try:
            normalized_audio_format = _normalize_audio_format(audio_format)
        except ValueError:
            await safe_reply(
                ctx,
                embed=error_embed(
                    desc=(
                        "Unsupported audio format. Choose from: "
                        f"{', '.join(SUPPORTED_AUDIO_FORMATS)}."
                    )
                ),
                mention_author=False,
                ephemeral=True,
            )
            return
        if normalized_audio_format and not audio_only:
            await safe_reply(
                ctx,
                embed=error_embed(desc="Audio format options require audio-only mode."),
                mention_author=False,
                ephemeral=True,
            )
            return
        await self._save_impl(
            ctx,
            SaveRequest(
                url=url,
                max_height=max_height,
                audio_only=audio_only,
                audio_focus=audio_focus,
                item=item,
                force_url=force_url,
                audio_format=normalized_audio_format,
            ),
        )

    async def _save_impl(self, ctx: commands.Context, request: SaveRequest) -> None:
        url = (request.url or "").strip()
        max_height = request.max_height
        audio_only = request.audio_only
        audio_focus = request.audio_focus
        item = request.item
        force_url = request.force_url
        audio_format = request.audio_format

        if not url:
            await safe_reply(
                ctx,
                embed=error_embed(desc="Provide a video URL to save."),
                mention_author=False,
                ephemeral=True,
            )
            return
        if audio_format and not audio_only:
            await safe_reply(
                ctx,
                embed=error_embed(desc="Audio format options require audio-only mode."),
                mention_author=False,
                ephemeral=True,
            )
            return

        # SSRF hardening: resolve hostnames and block private ranges.
        ok = await _is_safe_direct_url_async(url, dns_timeout_s=self.cfg.dns_timeout_s)

        if not ok:
            await safe_reply(
                ctx,
                embed=error_embed(desc="That URL is not allowed. Use a public http/https link."),
                mention_author=False,
                ephemeral=True,
            )
            return

        # slash/interaction response
        await defer_interaction(ctx)

        guild = ctx.guild
        # discord.py exposes the *effective* cap based on boost tier.
        max_upload_bytes = int(getattr(ctx, "filesize_limit", 0) or 0)
        if max_upload_bytes <= 0:
            max_upload_bytes = self.cfg.max_bytes

        max_bytes = self.cfg.max_bytes

        async with self._sem:
            logger = YTDLLogger()
            headers = _extra_headers(url, self.cfg.user_agent)
            resolved_url = await _resolve_url_for_policy(
                url,
                headers,
                timeout_s=min(10, self.cfg.timeout_s),
                dns_timeout_s=self.cfg.dns_timeout_s,
            )
            if resolved_url != url:
                url = resolved_url
                headers = _extra_headers(url, self.cfg.user_agent)
            host = urlparse(url).hostname or ""
            impersonate = self.cfg.impersonate
            if impersonate is None and (
                "tiktok.com" in host or host.endswith("x.com") or host.endswith("twitter.com")
            ):
                impersonate = "chrome"
            cookies_file = self.cfg.cookies_file
            cookies_from_browser = self.cfg.cookies_from_browser
            auto_cookie_eligible = (
                self.cfg.cookies_auto
                and not cookies_file
                and not cookies_from_browser
                and (host.endswith("x.com") or host.endswith("twitter.com") or "tiktok.com" in host)
            )
            cookie_db_exists = (
                _browser_cookie_db_exists(self.cfg.cookies_auto_browser)
                if auto_cookie_eligible
                else False
            )
            auto_cookie_active = auto_cookie_eligible and cookie_db_exists
            if auto_cookie_active:
                cookies_from_browser = (self.cfg.cookies_auto_browser,)
            log.info(
                "save cookies config: file=%s from_browser=%s auto=%s auto_browser=%s db_exists=%s active=%s",
                cookies_file or "-",
                cookies_from_browser or "-",
                self.cfg.cookies_auto,
                self.cfg.cookies_auto_browser,
                cookie_db_exists,
                auto_cookie_active,
            )
            used_generic_extractor = False

            progress_title = "⏳ Downloading audio..." if audio_only else "⏳ Downloading video..."
            progress_link_label = "Loading..."
            progress_embed = discord.Embed(
                title=progress_title,
                description=f"[{progress_link_label}]({url})",
                color=0xF1C40F,
            )
            progress_message: discord.Message | None = None
            try:
                progress_message = await safe_reply(
                    ctx, embed=progress_embed, mention_author=False
                )
            except Exception:
                progress_message = None

            try:
                self.cfg.work_root.mkdir(parents=True, exist_ok=True)

                with tempfile.TemporaryDirectory(prefix="savevideo_", dir=self.cfg.work_root) as tmp_dir:
                    tmp_path = Path(tmp_dir)

                    last_probe_error: Exception | None = None
                    probe_extra_opts: dict[str, Any] | None = None
                    try:
                        for attempt in range(3):
                            try:
                                info = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _probe_info,
                                        url,
                                        logger,
                                        headers,
                                        cookies_file,
                                        cookies_from_browser,
                                        impersonate,
                                        probe_extra_opts,
                                    ),
                                    timeout=float(self.cfg.timeout_s),
                                )
                                break
                            except yt_dlp.utils.DownloadError as exc:
                                last_probe_error = exc
                                raw_error = _strip_ansi(str(exc).strip() or repr(exc))
                                tail = _strip_ansi(logger.tail(self.cfg.log_tail_lines))
                                kind = _classify_error(raw_error)
                                if kind == "unknown" and tail:
                                    kind = _classify_error(tail)

                                if auto_cookie_active and cookies_from_browser is None and kind in {
                                    "login_required",
                                    "forbidden",
                                    "rate_limited",
                                    "private",
                                    "unknown",
                                }:
                                    cookies_from_browser = (self.cfg.cookies_auto_browser,)
                                    continue
                                if not used_generic_extractor:
                                    used_generic_extractor = True
                                    probe_extra_opts = {"force_generic_extractor": True}
                                    continue
                                raise
                        else:
                            raise last_probe_error or RuntimeError("Download failed")
                        title_text = str(info.get("title") or "Saved video")
                        title_text = _truncate_text(title_text, 256)
                        title_text = _escape_markdown_link_label(title_text)
                        if progress_message is not None:
                            updated_embed = discord.Embed(
                                title=progress_title,
                                description=f"[{title_text}]({url})",
                                color=0xF1C40F,
                            )
                            with contextlib.suppress(Exception):
                                await progress_message.edit(embed=updated_embed)
                    except yt_dlp.utils.DownloadError as exc:
                        fallback_urls: list[str] = []
                        if host.endswith("x.com") or host.endswith("twitter.com"):
                            fallback_urls.extend(
                                await asyncio.to_thread(
                                    _extract_x_fallback_urls,
                                    url,
                                    headers,
                                    timeout_s=self.cfg.timeout_s,
                                )
                            )
                        fallback_urls.extend(
                            await asyncio.to_thread(
                                _collect_html_fallback_urls,
                                url,
                                headers,
                                timeout_s=self.cfg.timeout_s,
                            )
                        )
                        candidate_url = None
                        for candidate in fallback_urls:
                            if await _is_safe_direct_url_async(
                                candidate, dns_timeout_s=self.cfg.dns_timeout_s
                            ):
                                candidate_url = candidate
                                break
                        if candidate_url is None:
                            raise exc
                        url = candidate_url
                        headers = _extra_headers(url, self.cfg.user_agent)
                        host = urlparse(url).hostname or ""
                        if impersonate is None and (
                            "tiktok.com" in host or host.endswith("x.com") or host.endswith("twitter.com")
                        ):
                            impersonate = "chrome"
                        if not (
                            host.endswith("x.com") or host.endswith("twitter.com") or "tiktok.com" in host
                        ):
                            cookies_from_browser = None
                        probe_extra_opts = {"force_generic_extractor": True}
                        info = await asyncio.wait_for(
                            asyncio.to_thread(
                                _probe_info,
                                url,
                                logger,
                                headers,
                                cookies_file,
                                cookies_from_browser,
                                impersonate,
                                probe_extra_opts,
                            ),
                            timeout=float(self.cfg.timeout_s),
                        )

                    # yt-dlp can still return a playlist-like object in some edge cases.
                    entries = info.get("entries")
                    if isinstance(entries, list):
                        if not entries:
                            raise RuntimeError("No entries found in this URL.")
                        if item < 1 or item > len(entries):
                            range_message = (
                                f"This URL contains {len(entries)} items. "
                                "Choose one with `item` (slash) or `--item` (prefix)."
                            )
                            log.info("save item out of range: %s", range_message)
                            range_embed = error_embed(desc=range_message)
                            if progress_message is not None:
                                try:
                                    await progress_message.edit(
                                        content=None, embed=range_embed, attachments=[]
                                    )
                                except Exception:
                                    await safe_reply(
                                        ctx,
                                        embed=range_embed,
                                        mention_author=False,
                                        ephemeral=True,
                                    )
                            else:
                                await safe_reply(
                                    ctx,
                                    embed=range_embed,
                                    mention_author=False,
                                    ephemeral=True,
                                )
                            return
                        entry = entries[item - 1]
                        if not isinstance(entry, dict):
                            raise RuntimeError("Selected entry metadata is unavailable.")
                        entry_url = _first_http_url(
                            [entry.get("webpage_url"), entry.get("original_url"), entry.get("url")]
                        )
                        if not entry_url:
                            raise RuntimeError(
                                "This entry does not expose a direct URL. Try a direct link instead."
                            )
                        if not await _is_safe_direct_url_async(
                            entry_url, dns_timeout_s=self.cfg.dns_timeout_s
                        ):
                            raise RuntimeError("That entry URL is not allowed.")
                        url = entry_url
                        headers = _extra_headers(url, self.cfg.user_agent)
                        host = urlparse(url).hostname or ""
                        if impersonate is None and (
                            "tiktok.com" in host or host.endswith("x.com") or host.endswith("twitter.com")
                        ):
                            impersonate = "chrome"
                        if auto_cookie_active:
                            cookies_from_browser = (self.cfg.cookies_auto_browser,)
                        else:
                            cookies_from_browser = self.cfg.cookies_from_browser
                        info = await asyncio.wait_for(
                            asyncio.to_thread(
                                _probe_info,
                                url,
                                logger,
                                headers,
                                cookies_file,
                                cookies_from_browser,
                                impersonate,
                                probe_extra_opts,
                            ),
                            timeout=float(self.cfg.timeout_s),
                        )

                    if audio_only:
                        candidates = _pick_best_audio_formats(info, max_bytes=max_bytes)
                        if not candidates:
                            raise RuntimeError("No audio-only formats found")
                    else:
                        candidates = _pick_best_formats(
                            info,
                            max_bytes=max_bytes,
                            max_height=max_height,
                            audio_focus=audio_focus,
                        )
                        if not candidates:
                            raise RuntimeError("No compatible formats found")

                    last_error: Exception | None = None
                    download_info: dict[str, Any] | None = None
                    download_path: Path | None = None
                    chosen_spec: str | None = None
                    chosen_est: int | None = None

                    # Try a few best candidates first.
                    if audio_only:
                        audio_prefer_exts = DEFAULT_AUDIO_EXTS
                        audio_extra_opts = dict(probe_extra_opts or {})
                        if audio_format:
                            codec = SUPPORTED_AUDIO_CODECS.get(audio_format)
                            if codec:
                                postprocessors = list(audio_extra_opts.get("postprocessors") or [])
                                postprocessors.append(
                                    {"key": "FFmpegExtractAudio", "preferredcodec": codec}
                                )
                                audio_extra_opts["postprocessors"] = postprocessors
                                audio_prefer_exts = SUPPORTED_AUDIO_EXTS.get(
                                    audio_format, (audio_format,)
                                )
                        for spec, _fmt, est in candidates[:8]:
                            try:
                                chosen_spec = spec
                                chosen_est = est if isinstance(est, int) else None
                                download_info, download_path = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _download_with_spec,
                                        url,
                                        format_spec=spec,
                                        max_bytes=max_bytes,
                                        logger=logger,
                                        work_dir=tmp_path,
                                        headers=headers,
                                        cookies_file=cookies_file,
                                        cookies_from_browser=cookies_from_browser,
                                        socket_timeout=max(10, int(self.cfg.timeout_s)),
                                        impersonate=impersonate,
                                        prefer_exts=audio_prefer_exts,
                                        extra_opts=audio_extra_opts or None,
                                    ),
                                    timeout=float(self.cfg.timeout_s),
                                )
                                break
                            except Exception as exc:
                                last_error = exc
                                log.debug("save download failed for %s: %s", spec, exc)
                    else:
                        for spec, vfmt, afmt, est in candidates[:8]:
                            try:
                                chosen_spec = spec
                                chosen_est = est if isinstance(est, int) else None
                                download_info, download_path = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _download_with_spec,
                                        url,
                                        format_spec=spec,
                                        max_bytes=max_bytes,
                                        logger=logger,
                                        work_dir=tmp_path,
                                        headers=headers,
                                        cookies_file=cookies_file,
                                        cookies_from_browser=cookies_from_browser,
                                        socket_timeout=max(10, int(self.cfg.timeout_s)),
                                        impersonate=impersonate,
                                        prefer_exts=DEFAULT_VIDEO_EXTS,
                                        extra_opts=probe_extra_opts,
                                    ),
                                    timeout=float(self.cfg.timeout_s),
                                )
                                break
                            except Exception as exc:
                                last_error = exc
                                log.debug("save download failed for %s: %s", spec, exc)

                    if download_path is None or download_info is None:
                        raise last_error or RuntimeError("Download failed")

                    file_size = download_path.stat().st_size

                    title = str(download_info.get("title") or info.get("title") or "Saved video")
                    duration = download_info.get("duration", info.get("duration"))
                    duration_text = humanize_delta(duration) if isinstance(duration, (int, float)) else "?"
                    res_width = download_info.get("width")
                    res_height = download_info.get("height")
                    resolution = (
                        f"{res_width}x{res_height}"
                        if isinstance(res_width, int) and isinstance(res_height, int)
                        else "unknown"
                    )
                    webpage_url = download_info.get("webpage_url") or info.get("webpage_url") or url

                    embed_title = "💾 Audio saved" if audio_only else "💾 Video saved"
                    embed = discord.Embed(
                        title=embed_title,
                        description=f"[{title}]({webpage_url})",
                        color=0x2ECC71,
                    )
                    embed.add_field(name="Duration", value=duration_text, inline=True)
                    if not audio_only:
                        embed.add_field(name="Resolution", value=resolution, inline=True)
                    embed.add_field(name="Size", value=_human_size(file_size), inline=True)
                    if chosen_spec:
                        embed.add_field(name="Format", value=chosen_spec, inline=False)
                    if chosen_est is not None:
                        embed.add_field(name="Est. size", value=_human_size(chosen_est), inline=True)

                    if audio_only:
                        abr = download_info.get("abr") or download_info.get("tbr")
                        if isinstance(abr, (int, float)) and abr > 0:
                            embed.add_field(name="Bitrate", value=f"{abr:.0f}kbps", inline=True)

                    ext = download_path.suffix or ".mp4"
                    base_name = _safe_filename(title)
                    filename = f"{base_name}{ext}"
                    if (
                        force_url
                        or file_size > max_upload_bytes
                        or file_size > DEFAULT_EXTERNAL_LINK_LIMIT_BYTES
                    ):
                        link = await self._create_external_download_link(
                            ctx,
                            download_path,
                            filename=filename,
                        )
                        if not link:
                            if force_url:
                                raise RuntimeError("Failed to create an external download link")
                            raise RuntimeError("max-filesize: exceeds Discord upload limit")
                        embed.add_field(
                            name="Download link (30 min)",
                            value=link,
                            inline=False,
                        )
                        if progress_message is not None:
                            await self._cleanup_progress_message(progress_message)
                        await safe_reply(ctx, embed=embed, mention_author=False)
                        return
                    file_obj = discord.File(download_path, filename=filename)
                    if progress_message is not None:
                        await self._cleanup_progress_message(progress_message)
                    await safe_reply(ctx, embed=embed, file=file_obj, mention_author=False)

            except yt_dlp.utils.DownloadError as exc:
                await self._handle_error(
                    ctx, url, exc, logger, max_upload_bytes, progress_message=progress_message
                )
            except asyncio.TimeoutError:
                timeout_message = (
                    f"Download timed out after {self.cfg.timeout_s}s. "
                    "Try a shorter clip, or increase SAVEVIDEO_TIMEOUT_S."
                )
                log.info("save timed out: %s", timeout_message)
                timeout_embed = error_embed(desc=timeout_message)
                if progress_message is not None:
                    await self._cleanup_progress_message(progress_message)
                await safe_reply(
                    ctx,
                    embed=timeout_embed,
                    mention_author=False,
                    ephemeral=True,
                )
            except Exception as exc:
                await self._handle_error(
                    ctx, url, exc, logger, max_upload_bytes, progress_message=progress_message
                )

    async def _handle_error(
        self,
        ctx: commands.Context,
        url: str,
        exc: Exception,
        logger: YTDLLogger,
        max_upload_bytes: int,
        *,
        progress_message: discord.Message | None = None,
    ) -> None:
        raw_error = str(exc).strip()
        if not raw_error:
            raw_error = str(getattr(exc, "msg", "") or "").strip()
        if not raw_error and getattr(exc, "args", None):
            raw_error = str(exc.args[0]).strip()
        if not raw_error:
            raw_error = repr(exc)

        raw_error = _strip_ansi(raw_error)
        tail = _strip_ansi(logger.tail(self.cfg.log_tail_lines))
        kind = _classify_error(raw_error)
        if kind == "unknown" and tail:
            kind = _classify_error(tail)

        reason = {
            "unsupported_url": "Unsupported URL or no compatible formats.",
            "forbidden": "Access denied (403). The site may block bots.",
            "rate_limited": "Rate limited (429). Wait a bit and retry.",
            "private": "The video looks private, deleted, or unavailable.",
            "login_required": "Login is required (or age/anti-bot verification).",
            "geo_restricted": "Geo-restricted. The content is not available from this region.",
            "ffmpeg_missing": "ffmpeg is missing on the host. Install ffmpeg to merge streams.",
            "max_filesize": f"The video exceeds the Discord upload limit ({_human_size(max_upload_bytes)}).",
            "unknown": "Extraction failed. The site may require cookies or a newer yt-dlp.",
        }.get(kind, "Extraction failed.")

        host = urlparse(url).hostname or ""
        cookies_hint = ""
        if (
            kind in {"unknown", "login_required", "forbidden", "rate_limited", "private"}
            and not self.cfg.cookies_file
            and not self.cfg.cookies_from_browser
            and (host.endswith("x.com") or host.endswith("twitter.com") or "tiktok.com" in host)
        ):
            cookies_hint = (
                "\nThis site often requires login cookies. "
                "Set SAVEVIDEO_COOKIES_FILE or SAVEVIDEO_COOKIES_FROM_BROWSER."
            )

        base = f"{reason}{cookies_hint}\nURL: {url}\nRaw error: {_truncate_text(raw_error, 500)}"
        message = _assemble_error_message(
            base=base,
            tail=tail,
            max_len=DEFAULT_ERROR_TEXT_MAX,
        )

        content = _cap_error_message(message, DISCORD_CONTENT_LIMIT)
        log.info("save error: %s", content)
        error = error_embed(desc=content)
        if progress_message is not None:
            await self._cleanup_progress_message(progress_message)
        await safe_reply(ctx, embed=error, mention_author=False, ephemeral=True)

    @staticmethod
    async def _cleanup_progress_message(message: discord.Message) -> None:
        with contextlib.suppress(Exception):
            await message.delete()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Save(bot))
