from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import contextlib
from typing import Any, Literal
import uuid

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

BrowserMode = Literal["launch", "cdp"]


@dataclass
class BrowserObservation:
    url: str
    title: str
    aria: str

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title, "aria": self.aria}


class BrowserAgent:
    """Async browser controller for LLM-driven actions."""

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
        self._active_tab_id: str | None = None
        self._owns_browser = False
        self._owns_context = False
        self._tab_actions: dict[str, deque[dict[str, Any]]] = {}

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserAgent has not started yet.")
        return self._page

    def is_started(self) -> bool:
        return self._playwright is not None and self._page is not None

    def _register_page(self, page: Page, *, set_active: bool = True) -> str:
        tab_id = uuid.uuid4().hex[:8]
        self._pages[tab_id] = page
        self._tab_actions[tab_id] = deque(maxlen=self.max_action_history)

        def _on_close() -> None:
            self._pages.pop(tab_id, None)
            self._tab_actions.pop(tab_id, None)
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
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
        user_data_dir: str | None = None,
    ) -> None:
        if self._playwright is not None:
            return

        self._playwright = await async_playwright().start()

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
            self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
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
        self._context.on("page", lambda page: self._register_page(page, set_active=True))
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
        self._active_tab_id = None
        self._tab_actions = {}
        self._owns_browser = False
        self._owns_context = False

    async def observe(self) -> BrowserObservation:
        return await self._observe_page(self.page)

    async def _observe_page(self, page: Page) -> BrowserObservation:
        title = await page.title()

        aria = ""
        try:
            locator = page.locator("body")
            if hasattr(locator, "aria_snapshot"):
                aria = await locator.aria_snapshot()
            else:
                raise AttributeError("aria_snapshot not available")
        except Exception:
            try:
                snapshot = await page.accessibility.snapshot()
                aria = str(snapshot)
            except Exception:
                aria = ""

        if len(aria) > self.max_aria_chars:
            aria = aria[: self.max_aria_chars] + "\n...[truncated]"

        return BrowserObservation(url=page.url, title=title, aria=aria)

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
        page = self.page
        action_type = str(action.get("type") or "")
        active_tab_id = self._active_tab_id

        try:
            if action_type == "goto":
                await page.goto(str(action["url"]))
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"url": self._truncate_detail(action.get("url"))},
                )
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
            elif action_type == "press":
                await page.keyboard.press(str(action["key"]))
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"key": self._truncate_detail(action.get("key"))},
                )
            elif action_type == "wait_for_load":
                await page.wait_for_load_state(str(action.get("state", "load")))
                self._record_action(
                    active_tab_id,
                    action_type,
                    {"state": self._truncate_detail(action.get("state", "load"))},
                )
            elif action_type == "observe":
                pass
            else:
                return {
                    "ok": False,
                    "error": f"unknown action type: {action_type}",
                    "observation": (await self.observe()).to_dict(),
                }

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
