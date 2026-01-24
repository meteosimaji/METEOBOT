from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
    ) -> None:
        self.default_timeout_ms = default_timeout_ms
        self.max_aria_chars = max_aria_chars

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._owns_browser = False
        self._owns_context = False

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserAgent has not started yet.")
        return self._page

    def is_started(self) -> bool:
        return self._playwright is not None and self._page is not None

    async def start(
        self,
        *,
        mode: BrowserMode = "launch",
        headless: bool = True,
        cdp_url: str | None = None,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
    ) -> None:
        if self._playwright is not None:
            return

        self._playwright = await async_playwright().start()

        if mode == "launch":
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
            self._context = await self._browser.new_context(
                viewport=viewport,
                user_agent=user_agent,
            )
            self._owns_context = True

        self._context.set_default_timeout(self.default_timeout_ms)
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

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
        self._owns_browser = False
        self._owns_context = False

    async def observe(self) -> BrowserObservation:
        page = self.page
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

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        page = self.page
        action_type = str(action.get("type") or "")

        try:
            if action_type == "goto":
                await page.goto(str(action["url"]))
            elif action_type == "click":
                await page.locator(str(action["selector"])).click()
            elif action_type == "click_role":
                role = str(action["role"])
                name = action.get("name")
                await page.get_by_role(role, name=str(name) if name else None).click()
            elif action_type == "fill":
                await page.locator(str(action["selector"])).fill(str(action.get("text", "")))
            elif action_type == "fill_role":
                role = str(action["role"])
                name = action.get("name")
                await page.get_by_role(role, name=str(name) if name else None).fill(
                    str(action.get("text", ""))
                )
            elif action_type == "press":
                await page.keyboard.press(str(action["key"]))
            elif action_type == "wait_for_load":
                await page.wait_for_load_state(str(action.get("state", "load")))
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
