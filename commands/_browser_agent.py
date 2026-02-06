from __future__ import annotations

# mypy: ignore-errors
from dataclasses import dataclass, replace
from collections import deque
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import asyncio
import contextlib
import importlib.metadata
import logging
import os
import time
from typing import Any, Awaitable, Callable, Literal
import uuid
import re

import playwright

from playwright.async_api import (
    Browser,
    BrowserContext,
    Frame,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

BrowserMode = Literal["launch", "cdp"]
RefMode = Literal["aria", "role", "css"]

log = logging.getLogger(__name__)

INTERACTIVE_ROLES: set[str] = {
    "button",
    "checkbox",
    "combobox",
    "link",
    "listbox",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
}

CONTEXT_ROLES: set[str] = {
    "article",
    "cell",
    "columnheader",
    "group",
    "heading",
    "list",
    "listitem",
    "region",
    "row",
    "rowheader",
    "section",
}

ARIA_REF_LINE_RE = re.compile(
    r'^\s*-\s*(?P<role>[\w-]+)(?:\s+"(?P<name>[^"]*)")?.*?\[ref=(?P<ref>e\d+)\]',
    re.IGNORECASE,
)

ARIA_TREE_LINE_RE = re.compile(
    r'^(?P<indent>\s*)-\s*(?P<role>[\w-]+)(?:\s+"(?P<name>[^"]*)")?(?:.*?\[ref=(?P<ref>e\d+)\])?',
    re.IGNORECASE,
)

MAX_REF_ENTRIES = 200
MIN_BBOX_ENTRIES = 5
MAX_SELECTOR_CHARS = 200
REF_FALLBACK_MAX_ITEMS = int(os.getenv("ASK_BROWSER_REF_FALLBACK_MAX_ITEMS", "30"))
REF_PRIMARY_MAX_ITEMS = int(os.getenv("ASK_BROWSER_REF_PRIMARY_MAX_ITEMS", "60"))


def _env_timeout_seconds(name: str, default: float, *, min_value: float = 0.0, max_value: float = 30.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return min(max(parsed, min_value), max_value)


OBSERVE_READ_TIMEOUT_S = _env_timeout_seconds("ASK_BROWSER_OBSERVE_READ_TIMEOUT_S", 2.5)
POST_ACTION_LOAD_TIMEOUT_S = _env_timeout_seconds("ASK_BROWSER_POST_ACTION_LOAD_TIMEOUT_S", 1.0)
REF_PRIMARY_READ_TIMEOUT_S = _env_timeout_seconds("ASK_BROWSER_REF_PRIMARY_TIMEOUT_S", 6.0)
REF_FALLBACK_READ_TIMEOUT_S = _env_timeout_seconds("ASK_BROWSER_REF_FALLBACK_TIMEOUT_S", 4.5)

CSS_PATH_SCRIPT = """
el => {
  const escape =
    window.CSS && window.CSS.escape
      ? window.CSS.escape
      : (value) => value.replace(/([\\s#.:>+~\\[\\](),=])/g, "\\\\$1");
  if (!(el instanceof Element)) return "";
  const path = [];
  let element = el;
  while (element && element.nodeType === Node.ELEMENT_NODE) {
    let selector = element.nodeName.toLowerCase();
    if (element.id) {
      selector += "#" + escape(element.id);
      path.unshift(selector);
      break;
    }
    let sibling = element;
    let nth = 1;
    while ((sibling = sibling.previousElementSibling)) {
      if (sibling.nodeName.toLowerCase() === selector) nth += 1;
    }
    if (nth > 1) {
      selector += `:nth-of-type(${nth})`;
    }
    path.unshift(selector);
    element = element.parentElement;
  }
  return path.join(" > ");
}
"""


@dataclass
class RefEntry:
    ref: str
    role: str | None
    name: str | None
    nth: int | None
    mode: RefMode
    selector: str | None
    bbox: dict[str, float] | None
    frame_name: str | None
    frame_url: str | None

    def to_dict(self) -> dict[str, Any]:
        selector = self.selector
        if selector:
            selector = selector[:MAX_SELECTOR_CHARS]
        return {
            "ref": self.ref,
            "role": self.role,
            "name": self.name,
            "nth": self.nth,
            "mode": self.mode,
            "selector": selector,
            "bbox": self.bbox,
            "frame_name": self.frame_name,
            "frame_url": self.frame_url,
        }


@dataclass
class BrowserObservation:
    url: str
    title: str
    aria: str
    ref_generation: int
    ref_snapshot: str
    refs: list[dict[str, Any]]
    ok: bool = True
    title_error: str | None = None
    aria_error: str | None = None
    ref_error: str | None = None
    ref_error_raw: str | None = None
    ref_degraded: bool = False
    nav_race: bool = False
    last_good_used: bool = False
    retry_count: int = 0
    error: str | None = None
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "aria": self.aria,
            "ref_generation": self.ref_generation,
            "ref_snapshot": self.ref_snapshot,
            "refs": self.refs,
            "ok": self.ok,
            "title_error": self.title_error,
            "aria_error": self.aria_error,
            "ref_error": self.ref_error,
            "ref_error_raw": self.ref_error_raw,
            "ref_degraded": self.ref_degraded,
            "nav_race": self.nav_race,
            "last_good_used": self.last_good_used,
            "retry_count": self.retry_count,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class BrowserAgent:
    """Async browser controller for LLM-driven actions."""

    _NAVIGATION_ERROR_MARKERS = (
        "Execution context was destroyed",
        "most likely because of a navigation",
        "Frame was detached",
        "Target closed",
        "has been closed",
    )
    _OBSERVE_READ_TIMEOUT_S = OBSERVE_READ_TIMEOUT_S
    _POST_ACTION_LOAD_TIMEOUT_S = POST_ACTION_LOAD_TIMEOUT_S

    def __init__(
        self,
        *,
        default_timeout_ms: int = 15_000,
        max_aria_chars: int = 10_000,
        max_action_history: int = 20,
    ) -> None:
        self.default_timeout_ms = default_timeout_ms
        self.max_aria_chars = max_aria_chars
        self.max_action_history = max_action_history

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._pages: dict[str, Page] = {}
        self._page_ids: dict[Page, str] = {}
        self._active_tab_id: str | None = None
        self._owns_browser = False
        self._owns_context = False
        self._tab_actions: dict[str, deque[dict[str, Any]]] = {}
        self._refs_by_tab: dict[str, dict[str, RefEntry]] = {}
        self._ref_generation_by_tab: dict[str, int] = {}
        self._aria_snapshot_ref_supported: bool | None = None
        self._last_good_observation_by_tab: dict[str, BrowserObservation] = {}

    @staticmethod
    def _detect_playwright_version() -> tuple[str, str]:
        try:
            return importlib.metadata.version("playwright"), "metadata"
        except Exception:
            version = getattr(playwright, "__version__", None)
            if isinstance(version, str) and version.strip():
                return version, "module_attribute"
            return "unknown", "fallback"

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserAgent has not started yet.")
        return self._page

    def is_started(self) -> bool:
        return self._playwright is not None and self._page is not None

    def needs_restart(self) -> bool:
        return self._playwright is not None and self._page is None

    @staticmethod
    def _empty_observation() -> BrowserObservation:
        return BrowserObservation(
            url="",
            title="",
            aria="",
            ref_generation=0,
            ref_snapshot="",
            refs=[],
            ok=False,
            error="empty_observation",
            timestamp=time.time(),
        )

    def _register_page(self, page: Page, *, set_active: bool = True) -> str:
        existing = self._page_ids.get(page)
        if existing:
            if set_active:
                self._active_tab_id = existing
                self._page = page
            return existing
        tab_id = uuid.uuid4().hex[:8]
        self._pages[tab_id] = page
        self._page_ids[page] = tab_id
        self._tab_actions[tab_id] = deque(maxlen=self.max_action_history)
        self._refs_by_tab[tab_id] = {}
        self._ref_generation_by_tab[tab_id] = 0

        def _on_close() -> None:
            self._pages.pop(tab_id, None)
            self._page_ids.pop(page, None)
            self._tab_actions.pop(tab_id, None)
            self._refs_by_tab.pop(tab_id, None)
            self._ref_generation_by_tab.pop(tab_id, None)
            if self._active_tab_id == tab_id:
                fallback = next(iter(self._pages.keys()), None)
                self._active_tab_id = fallback
                self._page = self._pages.get(fallback) if fallback else None

        page.on("close", _on_close)

        if set_active or self._active_tab_id is None:
            self._active_tab_id = tab_id
            self._page = page
        return tab_id

    async def start(
        self,
        *,
        mode: BrowserMode = "launch",
        headless: bool = True,
        cdp_url: str | None = None,
        cdp_headers: dict[str, str] | None = None,
        cdp_timeout_ms: int | None = None,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
        user_data_dir: str | None = None,
    ) -> None:
        if self._playwright is not None:
            if self._page is not None:
                return
            await self.close()

        try:
            await self._start_browser(
                mode=mode,
                headless=headless,
                cdp_url=cdp_url,
                cdp_headers=cdp_headers,
                cdp_timeout_ms=cdp_timeout_ms,
                viewport=viewport,
                user_agent=user_agent,
                user_data_dir=user_data_dir,
            )
        except Exception:
            await self.close()
            raise

    async def _start_browser(
        self,
        *,
        mode: BrowserMode,
        headless: bool,
        cdp_url: str | None,
        cdp_headers: dict[str, str] | None,
        cdp_timeout_ms: int | None,
        viewport: dict[str, int] | None,
        user_agent: str | None,
        user_data_dir: str | None,
    ) -> None:
        self._playwright = await async_playwright().start()
        playwright_version, source = self._detect_playwright_version()
        log.info("Playwright version: %s", playwright_version)
        log.debug("Playwright version source: %s", source)

        if mode == "launch":
            if user_data_dir:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=headless,
                    viewport=viewport,
                    user_agent=user_agent,
                )
                self._browser = self._context.browser
                self._owns_browser = True
                self._owns_context = True
            else:
                self._browser = await self._playwright.chromium.launch(headless=headless)
                self._context = await self._browser.new_context(
                    viewport=viewport,
                    user_agent=user_agent,
                )
                self._owns_browser = True
                self._owns_context = True
        else:
            if not cdp_url:
                raise ValueError("cdp_url is required for mode='cdp'.")
            self._browser = await self._playwright.chromium.connect_over_cdp(
                cdp_url,
                headers=cdp_headers,
                timeout=cdp_timeout_ms,
            )
            if self._browser.contexts:
                self._context = self._browser.contexts[0]
                self._owns_context = False
            else:
                self._context = await self._browser.new_context(
                    viewport=viewport,
                    user_agent=user_agent,
                )
                self._owns_context = True

        self._context.set_default_timeout(self.default_timeout_ms)
        self._context.on("page", lambda page: self._register_page(page, set_active=False))
        if self._context.pages:
            for page in self._context.pages:
                self._register_page(page, set_active=False)
            self._active_tab_id = next(reversed(self._pages.keys()), None)
            if self._active_tab_id:
                self._page = self._pages[self._active_tab_id]
        else:
            page = await self._context.new_page()
            self._register_page(page, set_active=True)

    async def close(self) -> None:
        if self._context is not None and self._owns_context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._browser is not None and self._owns_browser:
            # Only close browsers we launched; CDP-attached browsers are external.
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._pages = {}
        self._page_ids = {}
        self._active_tab_id = None
        self._tab_actions = {}
        self._refs_by_tab = {}
        self._ref_generation_by_tab = {}
        self._aria_snapshot_ref_supported = None
        self._owns_browser = False
        self._owns_context = False

    async def observe(self) -> BrowserObservation:
        page = self._page
        if page is None:
            return self._empty_observation()
        if page.is_closed():
            observation = self._empty_observation()
            return replace(observation, error="page_closed", timestamp=time.time())
        tab_id = self._active_tab_id or self._page_ids.get(page)
        try:
            observation = await self._observe_page(page)
        except Exception as exc:
            log.debug(
                "Observation failed: %s",
                f"{type(exc).__name__}: {exc}",
            )
            last_good = (
                self._last_good_observation_by_tab.get(tab_id) if tab_id else None
            )
            if last_good:
                return replace(
                    last_good,
                    ok=False,
                    last_good_used=True,
                    error=f"observe_failed: {type(exc).__name__}",
                    timestamp=time.time(),
                )
            observation = self._empty_observation()
            return replace(
                observation,
                error=f"observe_failed: {type(exc).__name__}",
                timestamp=time.time(),
            )
        last_good = self._last_good_observation_by_tab.get(tab_id) if tab_id else None
        if (
            tab_id
            and last_good is not None
            and not observation.refs
            and (observation.ref_error is not None or observation.ref_degraded)
            and last_good.refs
        ):
            self._ref_generation_by_tab[tab_id] = last_good.ref_generation
            restored_refs: dict[str, RefEntry] = {}
            for ref_obj in last_good.refs:
                if not isinstance(ref_obj, dict):
                    continue
                ref_id = str(ref_obj.get("ref") or "").strip()
                if not ref_id:
                    continue
                mode_raw = str(ref_obj.get("mode") or "role").lower()
                mode: RefMode = "role"
                if mode_raw in {"aria", "role", "css"}:
                    mode = mode_raw
                bbox_raw = ref_obj.get("bbox")
                bbox: dict[str, float] | None = None
                if isinstance(bbox_raw, dict):
                    try:
                        bbox = {
                            "x": float(bbox_raw.get("x", 0.0)),
                            "y": float(bbox_raw.get("y", 0.0)),
                            "width": float(bbox_raw.get("width", 0.0)),
                            "height": float(bbox_raw.get("height", 0.0)),
                        }
                    except (TypeError, ValueError):
                        bbox = None
                restored_refs[ref_id] = RefEntry(
                    ref=ref_id,
                    role=str(ref_obj.get("role") or "") or None,
                    name=str(ref_obj.get("name") or "") or None,
                    nth=int(ref_obj["nth"]) if isinstance(ref_obj.get("nth"), int) else None,
                    mode=mode,
                    selector=str(ref_obj.get("selector") or "") or None,
                    bbox=bbox,
                    frame_name=str(ref_obj.get("frame_name") or "") or None,
                    frame_url=str(ref_obj.get("frame_url") or "") or None,
                )
            if restored_refs:
                self._refs_by_tab[tab_id] = restored_refs
            observation = replace(
                observation,
                refs=last_good.refs,
                ref_snapshot=last_good.ref_snapshot,
                ref_generation=last_good.ref_generation,
                ref_degraded=True,
                last_good_used=True,
                ref_error=observation.ref_error or "ref_entries_reused_last_good",
            )
        if observation.ok and observation.refs and tab_id:
            self._last_good_observation_by_tab[tab_id] = observation
        return observation

    async def _observe_page(self, page: Page) -> BrowserObservation:
        title, title_retry_count, title_error, title_nav_race = await self._safe_page_read(
            page,
            "title",
            page.title,
            default="",
        )

        aria = ""
        aria_ref_snapshot: str | None = None
        aria_error = None
        nav_race = title_nav_race
        retry_count = title_retry_count
        locator = page.locator("body")
        if hasattr(locator, "aria_snapshot"):
            if self._aria_snapshot_ref_supported is not False:
                aria_ref_snapshot, aria_retry_count, aria_ref_error, aria_nav_race = (
                    await self._safe_page_read(
                        page,
                        "aria_snapshot_ref",
                        lambda: locator.aria_snapshot(ref=True),
                        default=None,
                    )
                )
                retry_count += aria_retry_count
                if aria_ref_error:
                    if "TypeError" in aria_ref_error:
                        if self._aria_snapshot_ref_supported is None:
                            log.debug(
                                "aria_snapshot(ref=True) unsupported; using non-ref snapshot fallback."
                            )
                        self._aria_snapshot_ref_supported = False
                    else:
                        aria_error = aria_ref_error
                else:
                    self._aria_snapshot_ref_supported = True
                nav_race = nav_race or aria_nav_race
            if aria_ref_snapshot:
                aria = aria_ref_snapshot
            else:
                aria, aria_retry_count, aria_snapshot_error, aria_snapshot_nav_race = (
                    await self._safe_page_read(
                        page,
                        "aria_snapshot",
                        locator.aria_snapshot,
                        default="",
                    )
                )
                retry_count += aria_retry_count
                nav_race = nav_race or aria_snapshot_nav_race
                if aria_snapshot_error:
                    aria_error = aria_snapshot_error
        else:
            aria_error = "aria_snapshot not available"
        if not aria:
            snapshot, snap_retry_count, snapshot_error, snapshot_nav_race = await self._safe_page_read(
                page,
                "accessibility_snapshot",
                page.accessibility.snapshot,
                default=None,
            )
            retry_count += snap_retry_count
            nav_race = nav_race or snapshot_nav_race
            if snapshot is not None:
                aria = str(snapshot)
                aria_error = None
            if snapshot_error and aria_error is None:
                aria_error = snapshot_error

        if len(aria) > self.max_aria_chars:
            aria = aria[: self.max_aria_chars] + "\n...[truncated]"

        tab_id = self._page_ids.get(page)
        ref_generation = 0
        ref_snapshot = ""
        refs: list[dict[str, Any]] = []
        ref_error = None
        ref_error_raw = None
        ref_degraded = False
        if tab_id:
            previous_generation = self._ref_generation_by_tab.get(tab_id, 0)
            previous_entries = self._refs_by_tab.get(tab_id)
            entries, ref_error, ref_error_raw, ref_degraded, ref_retry_count, ref_nav_race = (
                await self._build_ref_entries_with_fallback(
                    page,
                    aria_ref_snapshot=aria_ref_snapshot,
                )
            )
            retry_count += ref_retry_count
            nav_race = nav_race or ref_nav_race
            if entries:
                changed = self._ref_entries_materially_changed(previous_entries, entries)
                ref_generation = previous_generation + 1 if changed else previous_generation
                self._refs_by_tab[tab_id] = {entry.ref: entry for entry in entries}
                self._ref_generation_by_tab[tab_id] = ref_generation
                if aria_ref_snapshot and any(entry.mode == "aria" for entry in entries):
                    ref_snapshot = self._format_ref_snapshot_tree(aria_ref_snapshot, entries)
                else:
                    ref_snapshot = self._format_ref_snapshot(entries)
                refs = [entry.to_dict() for entry in entries]
            else:
                ref_generation = previous_generation
                cached_entries = list(self._refs_by_tab.get(tab_id, {}).values())
                if cached_entries:
                    refs = [entry.to_dict() for entry in cached_entries]
                    ref_snapshot = self._format_ref_snapshot(cached_entries)

        ok = not any([title_error, aria_error])
        error = "partial_observation" if not ok else None

        return BrowserObservation(
            url=page.url,
            title=title,
            aria=aria,
            ref_generation=ref_generation,
            ref_snapshot=ref_snapshot,
            refs=refs,
            ok=ok,
            title_error=title_error,
            aria_error=aria_error,
            ref_error=ref_error,
            ref_error_raw=ref_error_raw,
            ref_degraded=ref_degraded,
            nav_race=nav_race,
            retry_count=retry_count,
            error=error,
            timestamp=time.time(),
        )

    async def _safe_page_read(
        self,
        page: Page,
        label: str,
        func: Callable[[], Awaitable[Any]],
        *,
        default: Any,
        max_retries: int = 2,
        timeout_s: float | None = None,
    ) -> tuple[Any, int, str | None, bool]:
        retry_count = 0
        nav_race = False
        last_error: str | None = None
        read_timeout = self._OBSERVE_READ_TIMEOUT_S if timeout_s is None else timeout_s
        if read_timeout <= 0:
            try:
                return await func(), retry_count, None, nav_race
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.debug("Observation %s failed: %s", label, last_error)
                return default, retry_count, last_error, nav_race
        for attempt in range(max_retries + 1):
            try:
                read_task = asyncio.create_task(func())
                done, _ = await asyncio.wait({read_task}, timeout=read_timeout)
                if read_task in done:
                    return read_task.result(), retry_count, None, nav_race
                read_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await read_task
                raise asyncio.TimeoutError()
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if self._is_navigation_error(exc) and attempt < max_retries:
                    nav_race = True
                    retry_count += 1
                    log.debug(
                        "Retrying %s after navigation race (attempt %s/%s): %s",
                        label,
                        attempt + 1,
                        max_retries,
                        last_error,
                    )
                    await self._short_navigation_wait(page)
                    continue
                break
        log.debug("Observation %s failed: %s", label, last_error)
        return default, retry_count, last_error, nav_race

    async def _short_navigation_wait(self, page: Page) -> None:
        with contextlib.suppress(Exception):
            await page.wait_for_load_state(
                "domcontentloaded", timeout=int(self._POST_ACTION_LOAD_TIMEOUT_S * 1000)
            )
        with contextlib.suppress(Exception):
            await page.wait_for_timeout(100)

    def _is_navigation_error(self, exc: Exception) -> bool:
        message = str(exc)
        return any(marker in message for marker in self._NAVIGATION_ERROR_MARKERS)

    @staticmethod
    def _classify_ref_error(error: str | None) -> str | None:
        if not error:
            return None
        if error.startswith("TimeoutError"):
            return "ref_entries_timeout"
        if "Execution context was destroyed" in error:
            return "ref_entries_navigation_race"
        if "Frame was detached" in error:
            return "ref_entries_frame_detached"
        if "Target closed" in error or "has been closed" in error:
            return "ref_entries_target_closed"
        return "ref_entries_failed"

    async def _build_ref_entries_with_fallback(
        self,
        page: Page,
        *,
        aria_ref_snapshot: str | None,
    ) -> tuple[list[RefEntry], str | None, str | None, bool, int, bool]:
        entries, ref_retry_count, ref_error, ref_nav_race = await self._safe_page_read(
            page,
            "ref_entries",
            lambda: self._build_ref_entries(page, aria_ref_snapshot),
            default=[],
            timeout_s=max(0.0, REF_PRIMARY_READ_TIMEOUT_S),
        )
        ref_degraded = False
        raw_ref_error = ref_error
        classified_ref_error = self._classify_ref_error(ref_error)
        if not entries:
            fallback_entries, fb_retry_count, fallback_error, fb_nav_race = await self._safe_page_read(
                page,
                "ref_entries_fallback",
                lambda: self._refs_from_clickable_targets(
                    page,
                    ref_prefix="c",
                    max_items=max(1, REF_FALLBACK_MAX_ITEMS),
                ),
                default=[],
                timeout_s=max(0.0, REF_FALLBACK_READ_TIMEOUT_S),
            )
            ref_retry_count += fb_retry_count
            ref_nav_race = ref_nav_race or fb_nav_race
            if fallback_entries:
                entries = self._apply_nth_to_duplicates(fallback_entries[:MAX_REF_ENTRIES])
                classified_ref_error = "ref_entries_degraded_fallback_clickable"
                ref_degraded = True
            elif fallback_error and classified_ref_error is None:
                classified_ref_error = self._classify_ref_error(fallback_error)
                raw_ref_error = fallback_error
        if raw_ref_error:
            log.debug("Ref extraction raw error: %s", raw_ref_error)
        return (
            entries,
            classified_ref_error,
            raw_ref_error,
            ref_degraded,
            ref_retry_count,
            ref_nav_race,
        )

    async def _post_action_sync(self, page: Page, action_type: str, action: dict[str, Any]) -> None:
        navigation_prone = {"click", "click_ref", "click_role", "click_xy", "goto"}
        if action_type == "press":
            key = str(action.get("key") or "")
            if key.lower() in {"enter", "numpadenter"}:
                navigation_prone.add("press")
        if action_type == "type":
            text = str(action.get("text") or "")
            if "\n" in text or "\r" in text:
                navigation_prone.add("type")
        if action_type in navigation_prone:
            await self._short_navigation_wait(page)

    async def _build_ref_entries(
        self,
        page: Page,
        aria_ref_snapshot: str | None,
    ) -> list[RefEntry]:
        entries: list[RefEntry] = []
        if aria_ref_snapshot:
            entries = await self._refs_from_aria_snapshot(page, aria_ref_snapshot)
        bbox_count = sum(1 for entry in entries if entry.bbox)
        if not entries or bbox_count < MIN_BBOX_ENTRIES:
            fallback_entries = await self._refs_from_clickable_targets(
                page,
                ref_prefix="c",
                max_items=min(MAX_REF_ENTRIES, max(1, REF_PRIMARY_MAX_ITEMS)),
            )
            entries = self._merge_ref_entries(entries, fallback_entries)
        return self._apply_nth_to_duplicates(entries[:MAX_REF_ENTRIES])

    async def _refs_from_aria_snapshot(
        self,
        page: Page,
        aria_ref_snapshot: str,
    ) -> list[RefEntry]:
        entries: list[RefEntry] = []
        for line in aria_ref_snapshot.splitlines():
            match = ARIA_REF_LINE_RE.search(line)
            if not match:
                continue
            role = match.group("role").lower()
            name = match.group("name") or None
            ref = match.group("ref")
            if role.lower() not in INTERACTIVE_ROLES:
                continue
            locator = page.locator(f"aria-ref={ref}")
            selector = None
            bbox = None
            try:
                handle = await locator.element_handle()
            except Exception:
                handle = None
            if handle is not None:
                with contextlib.suppress(Exception):
                    bbox = await handle.bounding_box()
            entries.append(
                RefEntry(
                    ref=ref,
                    role=role,
                    name=name,
                    nth=None,
                    mode="aria",
                    selector=selector or None,
                    bbox=bbox,
                    frame_name=None,
                    frame_url=page.url or None,
                )
            )
        return entries

    async def _refs_from_clickable_targets(
        self,
        page: Page,
        *,
        ref_prefix: str,
        max_items: int,
    ) -> list[RefEntry]:
        selector = (
            "a, button, input, textarea, select, [role=button], [role=link], "
            "[role=tab], [role=menuitem], [role=checkbox], [role=radio], "
            "[role=combobox], [role=listbox], [role=menuitemcheckbox], [role=menuitemradio], "
            "[role=option], [role=searchbox], [role=slider], [role=spinbutton], [role=switch]"
        )
        entries: list[RefEntry] = []
        handle_by_entry: dict[int, Any] = {}
        tag_name_by_entry: dict[int, str] = {}
        max_scan = max(max_items * 6, max_items + 120)
        viewport = page.viewport_size or {}
        viewport_w = viewport.get("width")
        viewport_h = viewport.get("height")
        for frame in page.frames:
            try:
                handles = await frame.query_selector_all(selector)
            except Exception:
                continue
            for handle in handles:
                if len(entries) >= max_scan:
                    break
                try:
                    box = await handle.bounding_box()
                except Exception:
                    continue
                if not box:
                    continue
                if box.get("width", 0) < 2 or box.get("height", 0) < 2:
                    continue
                if viewport_w and viewport_h:
                    if (
                        box.get("x", 0) > viewport_w
                        or box.get("y", 0) > viewport_h
                        or (box.get("x", 0) + box.get("width", 0)) < 0
                        or (box.get("y", 0) + box.get("height", 0)) < 0
                    ):
                        continue
                role_attr = None
                try:
                    role_attr = await handle.get_attribute("role")
                except Exception:
                    role_attr = None
                try:
                    tag_name = await handle.evaluate("el => el.tagName.toLowerCase()")
                except Exception:
                    tag_name = ""
                try:
                    input_type = await handle.get_attribute("type")
                except Exception:
                    input_type = None
                role = self._infer_role(role_attr, tag_name, input_type)
                mode: RefMode = "css"
                entry = RefEntry(
                    ref="",
                    role=role,
                    name=None,
                    nth=None,
                    mode=mode,
                    selector=None,
                    bbox={
                        "x": float(box.get("x", 0)),
                        "y": float(box.get("y", 0)),
                        "width": float(box.get("width", 0)),
                        "height": float(box.get("height", 0)),
                    },
                    frame_name=frame.name or None,
                    frame_url=frame.url or None,
                )
                entries.append(entry)
                handle_by_entry[id(entry)] = handle
                tag_name_by_entry[id(entry)] = tag_name
            if len(entries) >= max_scan:
                break
        viewport_area = None
        if viewport_w and viewport_h:
            viewport_area = float(viewport_w) * float(viewport_h)

        def area(entry: RefEntry) -> float:
            bbox = entry.bbox or {}
            return float(bbox.get("width", 0)) * float(bbox.get("height", 0))

        def role_priority(role: str | None) -> int:
            if role in {"button", "link", "tab", "menuitem", "searchbox", "combobox"}:
                return 0
            if role in {"checkbox", "radio", "option"}:
                return 1
            return 2

        filtered: list[RefEntry] = []
        for entry in entries:
            bbox = entry.bbox or {}
            width = float(bbox.get("width", 0))
            height = float(bbox.get("height", 0))
            bbox_area = width * height
            if width < 10 or height < 10:
                continue
            if viewport_area is not None and bbox_area > viewport_area * 0.35:
                continue
            filtered.append(entry)

        base = filtered if len(filtered) >= max(5, max_items // 2) else entries
        def sort_key(entry: RefEntry) -> tuple[float, ...]:
            rounded = self._round_bbox(entry.bbox)
            if rounded:
                y_key = rounded[1]
                x_key = rounded[0]
            else:
                y_key = float((entry.bbox or {}).get("y", 0.0))
                x_key = float((entry.bbox or {}).get("x", 0.0))
            return (
                role_priority(entry.role),
                y_key,
                x_key,
                area(entry),
            )

        base.sort(key=sort_key)
        def iou(bbox_a: dict[str, float], bbox_b: dict[str, float]) -> float:
            ax1 = float(bbox_a.get("x", 0))
            ay1 = float(bbox_a.get("y", 0))
            ax2 = ax1 + float(bbox_a.get("width", 0))
            ay2 = ay1 + float(bbox_a.get("height", 0))
            bx1 = float(bbox_b.get("x", 0))
            by1 = float(bbox_b.get("y", 0))
            bx2 = bx1 + float(bbox_b.get("width", 0))
            by2 = by1 + float(bbox_b.get("height", 0))
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            return inter / (a_area + b_area - inter)

        selected: list[RefEntry] = []
        for entry in base:
            if len(selected) >= max_items:
                break
            if not entry.bbox:
                continue
            if any(
                iou(entry.bbox, chosen.bbox) > 0.65
                for chosen in selected
                if chosen.bbox
            ):
                continue
            selected.append(entry)
        for index, entry in enumerate(selected, start=1):
            entry.ref = f"{ref_prefix}{index}"
            if entry.selector is None:
                handle = handle_by_entry.get(id(entry))
                if handle is not None:
                    with contextlib.suppress(Exception):
                        entry.selector = await handle.evaluate(CSS_PATH_SCRIPT)
            if entry.name is None:
                handle = handle_by_entry.get(id(entry))
                tag_name = tag_name_by_entry.get(id(entry), "")
                if handle is not None:
                    with contextlib.suppress(Exception):
                        entry.name = await self._infer_accessible_name(handle, tag_name)
                if entry.role and entry.name:
                    entry.mode = "role"
        return selected

    @staticmethod
    def _merge_ref_entries(
        base_entries: list[RefEntry],
        fallback_entries: list[RefEntry],
    ) -> list[RefEntry]:
        merged = list(base_entries)
        selector_to_index: dict[str, int] = {}
        for idx, entry in enumerate(merged):
            if entry.selector:
                selector_to_index[entry.selector] = idx
        for entry in fallback_entries:
            if entry.selector and entry.selector in selector_to_index:
                existing = merged[selector_to_index[entry.selector]]
                if existing.bbox is None and entry.bbox is not None:
                    existing.bbox = entry.bbox
                if existing.frame_name is None and entry.frame_name:
                    existing.frame_name = entry.frame_name
                if existing.frame_url is None and entry.frame_url:
                    existing.frame_url = entry.frame_url
                continue
            merged.append(entry)
        return merged

    @staticmethod
    def _infer_role(role_attr: str | None, tag_name: str, input_type: str | None) -> str | None:
        if role_attr:
            return role_attr.lower()
        if tag_name == "a":
            return "link"
        if tag_name == "button":
            return "button"
        if tag_name == "textarea":
            return "textbox"
        if tag_name == "select":
            return "combobox"
        if tag_name == "input":
            input_kind = (input_type or "").lower()
            if input_kind in {"checkbox"}:
                return "checkbox"
            if input_kind in {"radio"}:
                return "radio"
            if input_kind in {"range"}:
                return "slider"
            if input_kind in {"search"}:
                return "searchbox"
            return "textbox"
        return None

    @staticmethod
    async def _infer_accessible_name(handle: Any, tag_name: str) -> str | None:
        for attr in ("aria-label", "alt", "title", "placeholder", "value", "name"):
            try:
                value = await handle.get_attribute(attr)
            except Exception:
                value = None
            if value:
                return value
        if tag_name in {"a", "button", "option"}:
            with contextlib.suppress(Exception):
                text = await handle.inner_text()
                if text:
                    return text
        return None

    @staticmethod
    def _apply_nth_to_duplicates(entries: list[RefEntry]) -> list[RefEntry]:
        counts: dict[tuple[str | None, str | None], int] = {}
        for entry in entries:
            if not entry.role or not entry.name:
                continue
            key = (entry.role, entry.name)
            counts[key] = counts.get(key, 0) + 1
        nth_tracker: dict[tuple[str | None, str | None], int] = {}
        for entry in entries:
            if not entry.role or not entry.name:
                continue
            key = (entry.role, entry.name)
            if counts.get(key, 0) > 1:
                entry.nth = nth_tracker.get(key, 0)
                nth_tracker[key] = entry.nth + 1
        return entries

    @staticmethod
    def _format_ref_snapshot(entries: list[RefEntry]) -> str:
        lines = []
        for entry in entries:
            role = entry.role or "element"
            name = f' "{entry.name}"' if entry.name else ""
            nth = f" (nth={entry.nth + 1})" if entry.nth is not None else ""
            lines.append(f"- {role}{name} [ref={entry.ref}]{nth}")
        return "\n".join(lines)

    @staticmethod
    def _format_ref_snapshot_tree(aria_ref_snapshot: str, entries: list[RefEntry]) -> str:
        entries_by_ref = {entry.ref: entry for entry in entries}
        parsed_nodes = BrowserAgent._parse_aria_tree_nodes(aria_ref_snapshot)
        if not parsed_nodes:
            return BrowserAgent._format_ref_snapshot(entries)

        keep_indices: set[int] = set()
        for idx, node in enumerate(parsed_nodes):
            node_ref = node.get("ref")
            if isinstance(node_ref, str) and node_ref in entries_by_ref:
                walk = idx
                while walk >= 0 and walk not in keep_indices:
                    keep_indices.add(walk)
                    parent_idx = parsed_nodes[walk]["parent"]
                    walk = int(parent_idx) if isinstance(parent_idx, int) else -1

        for idx, node in enumerate(parsed_nodes):
            if idx in keep_indices:
                continue
            role = str(node.get("role") or "").lower()
            if role not in CONTEXT_ROLES:
                continue
            parent_idx = node.get("parent")
            if isinstance(parent_idx, int) and parent_idx in keep_indices:
                keep_indices.add(idx)

        if not keep_indices:
            return BrowserAgent._format_ref_snapshot(entries)

        lines: list[str] = []
        for idx, node in enumerate(parsed_nodes):
            if idx not in keep_indices:
                continue
            role = str(node.get("role") or "element")
            node_name = node.get("name")
            name = f' "{node_name}"' if node_name else ""
            depth = int(node.get("depth") or 0)
            indent = "  " * depth
            node_ref = node.get("ref")
            ref_suffix = ""
            if isinstance(node_ref, str):
                entry = entries_by_ref.get(node_ref)
                if entry is not None:
                    nth = f" (nth={entry.nth + 1})" if entry.nth is not None else ""
                    ref_suffix = f" [ref={node_ref}]{nth}"
            lines.append(f"{indent}- {role}{name}{ref_suffix}")
        return "\n".join(lines)

    @staticmethod
    def _parse_aria_tree_nodes(aria_ref_snapshot: str) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        stack: list[tuple[int, int]] = []
        for line in aria_ref_snapshot.splitlines():
            match = ARIA_TREE_LINE_RE.search(line)
            if not match:
                continue
            indent_text = (match.group("indent") or "").replace("\t", "    ")
            indent_len = len(indent_text)
            while stack and stack[-1][0] >= indent_len:
                stack.pop()
            depth = len(stack)
            parent_idx = stack[-1][1] if stack else None
            node = {
                "depth": depth,
                "parent": parent_idx,
                "role": (match.group("role") or "").lower() or None,
                "name": match.group("name") or None,
                "ref": match.group("ref") or None,
            }
            nodes.append(node)
            stack.append((indent_len, len(nodes) - 1))
        return nodes

    def _get_ref_entry(self, tab_id: str | None, ref: str) -> RefEntry | None:
        if not tab_id:
            return None
        refs = self._refs_by_tab.get(tab_id)
        if refs is None:
            return None
        return refs.get(ref)

    def _ref_generation_matches(self, tab_id: str | None, ref_generation: int) -> bool:
        if not tab_id:
            return False
        return self._ref_generation_by_tab.get(tab_id, 0) == ref_generation

    @staticmethod
    def _round_bbox(bbox: dict[str, float] | None) -> tuple[float, float, float, float] | None:
        if not bbox:
            return None

        def _half_up(value: Any) -> float:
            if not isinstance(value, (int, float)):
                raise TypeError
            decimal_value = Decimal(str(float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            return float(decimal_value)

        x = bbox.get("x")
        y = bbox.get("y")
        width = bbox.get("width")
        height = bbox.get("height")
        try:
            return (_half_up(x), _half_up(y), _half_up(width), _half_up(height))
        except (TypeError, ValueError, InvalidOperation):
            return None

    @classmethod
    def _entry_signature(cls, entry: RefEntry) -> tuple[Any, ...]:
        return (
            entry.ref,
            entry.role,
            entry.name,
            entry.nth,
            entry.mode,
            entry.selector,
            cls._round_bbox(entry.bbox),
            entry.frame_name,
            entry.frame_url,
        )

    @classmethod
    def _ref_entries_materially_changed(
        cls,
        previous: dict[str, RefEntry] | None,
        current: list[RefEntry],
    ) -> bool:
        if previous is None:
            return bool(current)
        if len(previous) != len(current):
            return True
        for entry in current:
            prev_entry = previous.get(entry.ref)
            if prev_entry is None:
                return True
            if cls._entry_signature(prev_entry) != cls._entry_signature(entry):
                return True
        return False

    def _resolve_ref_frame(self, page: Page, entry: RefEntry) -> Page | Frame:
        if entry.frame_url and entry.frame_url == page.url:
            return page
        if entry.frame_name:
            frame = page.frame(name=entry.frame_name)
            if frame:
                return frame
        if entry.frame_url:
            frame = page.frame(url=entry.frame_url)
            if frame:
                return frame
        return page

    async def _resolve_ref_locator(self, page: Page, entry: RefEntry) -> Any:
        target = self._resolve_ref_frame(page, entry)
        candidates = []
        if entry.mode == "aria":
            candidates.append(target.locator(f"aria-ref={entry.ref}"))
        if entry.mode == "role" and entry.role and entry.name:
            candidates.append(target.get_by_role(entry.role, name=entry.name))
        if entry.mode == "css" and entry.selector:
            candidates.append(target.locator(entry.selector))
        if entry.mode == "aria" and entry.role and entry.name:
            candidates.append(target.get_by_role(entry.role, name=entry.name))
        if entry.mode == "aria" and entry.selector:
            candidates.append(target.locator(entry.selector))
        if entry.mode == "role" and entry.selector:
            candidates.append(target.locator(entry.selector))
        for locator in candidates:
            if entry.nth is not None:
                locator = locator.nth(entry.nth)
            try:
                handle = await locator.element_handle()
            except Exception:
                continue
            if handle is not None:
                return locator
        raise ValueError("ref_not_resolvable")

    async def _list_tabs(self) -> list[dict[str, Any]]:
        tabs = []
        for tab_id, page in self._pages.items():
            title = ""
            with contextlib.suppress(Exception):
                title = await page.title()
            tabs.append(
                {
                    "tab_id": tab_id,
                    "url": page.url,
                    "title": title,
                    "active": tab_id == self._active_tab_id,
                }
            )
        return tabs

    def _record_action(self, tab_id: str | None, action_type: str, details: dict[str, Any]) -> None:
        if not tab_id:
            return
        history = self._tab_actions.get(tab_id)
        if history is None:
            return
        history.append({"type": action_type, "details": details})

    @staticmethod
    def _truncate_detail(value: Any, limit: int = 120) -> str:
        text = str(value or "")
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        page = self._page
        if page is None:
            return {
                "ok": False,
                "error": "browser_not_started",
                "observation": self._empty_observation().to_dict(),
            }
        action_type = str(action.get("type") or "")
        active_tab_id = self._active_tab_id
        post_action_sync = False

        try:
            if action_type == "goto":
                await page.goto(str(action["url"]))
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"url": self._truncate_detail(action.get("url"))},
                )
                post_action_sync = True
            elif action_type == "click":
                locator = page.locator(str(action["selector"]))
                with contextlib.suppress(Exception):
                    await locator.scroll_into_view_if_needed(timeout=5_000)
                try:
                    await locator.click(timeout=self.default_timeout_ms)
                except Exception:
                    await locator.click(timeout=self.default_timeout_ms, force=True)
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"selector": self._truncate_detail(action.get("selector"))},
                )
            elif action_type == "scroll":
                delta_x_raw = action.get("delta_x", 0)
                delta_y_raw = action.get("delta_y", 800)
                after_ms_raw = action.get("after_ms", 150)
                delta_x = float(0 if delta_x_raw is None else delta_x_raw)
                delta_y = float(800 if delta_y_raw is None else delta_y_raw)
                after_ms = int(150 if after_ms_raw is None else after_ms_raw)
                await page.mouse.wheel(delta_x, delta_y)
                if after_ms > 0:
                    await page.wait_for_timeout(after_ms)
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"delta_x": delta_x, "delta_y": delta_y, "after_ms": after_ms},
                )
            elif action_type == "click_role":
                role = str(action["role"])
                name = action.get("name")
                locator = page.get_by_role(role, name=str(name) if name else None)
                with contextlib.suppress(Exception):
                    await locator.scroll_into_view_if_needed(timeout=5_000)
                try:
                    await locator.click(timeout=self.default_timeout_ms)
                except Exception:
                    await locator.click(timeout=self.default_timeout_ms, force=True)
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"role": role, "name": self._truncate_detail(name)},
                )
            elif action_type == "click_xy":
                x = float(action.get("x", 0))
                y = float(action.get("y", 0))
                button = str(action.get("button") or "left")
                clicks_raw = action.get("clicks", 1)
                clicks = int(1 if clicks_raw is None else clicks_raw)
                await page.mouse.click(x=x, y=y, button=button, click_count=clicks)
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"x": x, "y": y, "button": button, "clicks": clicks},
                )
                post_action_sync = True
            elif action_type == "click_ref":
                ref = str(action.get("ref") or "")
                ref_generation_raw = action.get("ref_generation")
                ref_generation = int(ref_generation_raw) if ref_generation_raw is not None else -1
                if not self._ref_generation_matches(active_tab_id, ref_generation):
                    return {
                        "ok": False,
                        "error": "ref_generation_mismatch",
                        "observation": (await self.observe()).to_dict(),
                    }
                entry = self._get_ref_entry(active_tab_id, ref)
                if entry is None:
                    return {
                        "ok": False,
                        "error": "unknown_ref",
                        "observation": (await self.observe()).to_dict(),
                    }
                locator = await self._resolve_ref_locator(page, entry)
                with contextlib.suppress(Exception):
                    await locator.scroll_into_view_if_needed(timeout=5_000)
                try:
                    await locator.click(timeout=self.default_timeout_ms)
                except Exception:
                    await locator.click(timeout=self.default_timeout_ms, force=True)
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"ref": ref, "generation": ref_generation},
                )
            elif action_type == "new_tab":
                url = action.get("url")
                focus = action.get("focus", True)
                new_page = await self._context.new_page()
                tab_id = self._register_page(new_page, set_active=bool(focus))
                if isinstance(url, str) and url:
                    await new_page.goto(url)
                    self._record_action(
                        tab_id,
                        action_type,
                        {"url": self._truncate_detail(url)},
                    )
                return {
                    "ok": True,
                    "tab_id": tab_id,
                    "observation": (await self.observe()).to_dict(),
                }
            elif action_type == "switch_tab":
                tab_id = str(action.get("tab_id") or "")
                target = self._pages.get(tab_id)
                if target is None:
                    return {
                        "ok": False,
                        "error": "unknown_tab",
                        "observation": (await self.observe()).to_dict(),
                    }
                self._active_tab_id = tab_id
                self._page = target
                self._record_action(tab_id, action_type, {"tab_id": tab_id})
            elif action_type == "close_tab":
                tab_id = str(action.get("tab_id") or "")
                target = self._pages.get(tab_id)
                if target is None:
                    return {
                        "ok": False,
                        "error": "unknown_tab",
                        "observation": (await self.observe()).to_dict(),
                    }
                await target.close()
                self._record_action(active_tab_id, action_type, {"tab_id": tab_id})
            elif action_type == "list_tabs":
                return {
                    "ok": True,
                    "tabs": await self._list_tabs(),
                    "observation": (await self.observe()).to_dict(),
                }
            elif action_type == "observe_tabs":
                include_aria = bool(action.get("include_aria", False))
                max_tabs_raw = action.get("max_tabs")
                max_tabs = int(max_tabs_raw) if max_tabs_raw is not None else None
                tabs = []
                for tab_id, tab_page in list(self._pages.items()):
                    if max_tabs is not None and len(tabs) >= max_tabs:
                        break
                    title = ""
                    with contextlib.suppress(Exception):
                        title = await tab_page.title()
                    aria = ""
                    if include_aria:
                        with contextlib.suppress(Exception):
                            aria = (await self._observe_page(tab_page)).aria
                    tabs.append(
                        {
                            "tab_id": tab_id,
                            "url": tab_page.url,
                            "title": title,
                            "active": tab_id == self._active_tab_id,
                            "aria": aria,
                            "last_actions": list(self._tab_actions.get(tab_id, [])),
                        }
                    )
                return {
                    "ok": True,
                    "tabs": tabs,
                    "observation": (await self.observe()).to_dict(),
                }
            elif action_type == "fill":
                await page.locator(str(action["selector"])).fill(str(action.get("text", "")))
                self._record_action(
                    active_tab_id,
                    action_type,
                    {
                        "selector": self._truncate_detail(action.get("selector")),
                        "text": self._truncate_detail(action.get("text")),
                    },
                )
            elif action_type == "fill_role":
                role = str(action["role"])
                name = action.get("name")
                await page.get_by_role(role, name=str(name) if name else None).fill(
                    str(action.get("text", ""))
                )
                self._record_action(
                    active_tab_id,
                    action_type,
                    {
                        "role": role,
                        "name": self._truncate_detail(name),
                        "text": self._truncate_detail(action.get("text")),
                    },
                )
            elif action_type == "fill_ref":
                ref = str(action.get("ref") or "")
                ref_generation_raw = action.get("ref_generation")
                ref_generation = int(ref_generation_raw) if ref_generation_raw is not None else -1
                if not self._ref_generation_matches(active_tab_id, ref_generation):
                    return {
                        "ok": False,
                        "error": "ref_generation_mismatch",
                        "observation": (await self.observe()).to_dict(),
                    }
                entry = self._get_ref_entry(active_tab_id, ref)
                if entry is None:
                    return {
                        "ok": False,
                        "error": "unknown_ref",
                        "observation": (await self.observe()).to_dict(),
                    }
                locator = await self._resolve_ref_locator(page, entry)
                text = str(action.get("text", ""))
                try:
                    await locator.fill(text)
                except Exception:
                    fallback = locator.locator("input, textarea, [contenteditable=true]").first
                    await fallback.fill(text)
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"ref": ref, "generation": ref_generation, "text": self._truncate_detail(text)},
                )
            elif action_type == "type":
                text = str(action.get("text", ""))
                await page.keyboard.type(text)
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"text": self._truncate_detail(text)},
                )
                if "\n" in text or "\r" in text:
                    post_action_sync = True
            elif action_type == "press":
                await page.keyboard.press(str(action["key"]))
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"key": self._truncate_detail(action.get("key"))},
                )
                key = str(action.get("key") or "")
                if key.lower() in {"enter", "numpadenter"}:
                    post_action_sync = True
            elif action_type == "wait_for_load":
                await page.wait_for_load_state(str(action.get("state", "load")))
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"state": self._truncate_detail(action.get("state", "load"))},
                )
            elif action_type == "hover_ref":
                ref = str(action.get("ref") or "")
                ref_generation_raw = action.get("ref_generation")
                ref_generation = int(ref_generation_raw) if ref_generation_raw is not None else -1
                if not self._ref_generation_matches(active_tab_id, ref_generation):
                    return {
                        "ok": False,
                        "error": "ref_generation_mismatch",
                        "observation": (await self.observe()).to_dict(),
                    }
                entry = self._get_ref_entry(active_tab_id, ref)
                if entry is None:
                    return {
                        "ok": False,
                        "error": "unknown_ref",
                        "observation": (await self.observe()).to_dict(),
                    }
                locator = await self._resolve_ref_locator(page, entry)
                with contextlib.suppress(Exception):
                    await locator.scroll_into_view_if_needed(timeout=5_000)
                await locator.hover()
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"ref": ref, "generation": ref_generation},
                )
            elif action_type in {"scroll_ref", "scroll_into_view_ref"}:
                ref = str(action.get("ref") or "")
                ref_generation_raw = action.get("ref_generation")
                ref_generation = int(ref_generation_raw) if ref_generation_raw is not None else -1
                if not self._ref_generation_matches(active_tab_id, ref_generation):
                    return {
                        "ok": False,
                        "error": "ref_generation_mismatch",
                        "observation": (await self.observe()).to_dict(),
                    }
                entry = self._get_ref_entry(active_tab_id, ref)
                if entry is None:
                    return {
                        "ok": False,
                        "error": "unknown_ref",
                        "observation": (await self.observe()).to_dict(),
                    }
                locator = await self._resolve_ref_locator(page, entry)
                await locator.scroll_into_view_if_needed(timeout=5_000)
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"ref": ref, "generation": ref_generation},
                )
            elif action_type == "observe":
                pass
            else:
                return {
                    "ok": False,
                    "error": f"unknown action type: {action_type}",
                    "observation": (await self.observe()).to_dict(),
                }

            if post_action_sync:
                await self._post_action_sync(page, action_type, action)
            return {"ok": True, "observation": (await self.observe()).to_dict()}
        except PlaywrightTimeoutError as exc:
            return {
                "ok": False,
                "error": f"timeout: {exc}",
                "observation": (await self.observe()).to_dict(),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"error: {type(exc).__name__}: {exc}",
                "observation": (await self.observe()).to_dict(),
            }
