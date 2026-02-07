from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


class ToolGateDenied(RuntimeError):
    pass


@dataclass(slots=True)
class ToolGatePolicy:
    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    permission_mode: str = "execute"


ToolHook = Callable[[dict[str, Any]], Awaitable[None] | None]


class ToolGate:
    def __init__(
        self,
        policy: ToolGatePolicy | None = None,
        *,
        cancel_event: asyncio.Event | None = None,
        emit_event: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        pre_hooks: list[ToolHook] | None = None,
        post_hooks: list[ToolHook] | None = None,
    ) -> None:
        self._policy = policy or ToolGatePolicy()
        self._cancel_event = cancel_event
        self._emit_event = emit_event
        self._pre_hooks = pre_hooks or []
        self._post_hooks = post_hooks or []

    def raise_if_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise ToolGateDenied("Task cancellation requested.")

    def _matches_allowed(self, tool_name: str) -> bool:
        allowed = self._policy.allowed_tools
        if not allowed:
            return True
        if "*" in allowed:
            return True
        if tool_name in allowed:
            return True
        if tool_name.startswith("function:") and "function" in allowed:
            return True
        return False

    def _matches_denied(self, tool_name: str) -> bool:
        denied = self._policy.denied_tools
        if tool_name in denied:
            return True
        if tool_name.startswith("function:") and "function" in denied:
            return True
        return False

    async def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._emit_event is None:
            return
        maybe = self._emit_event(event_type, payload)
        if asyncio.iscoroutine(maybe):
            await maybe

    async def _run_hooks(self, hooks: list[ToolHook], payload: dict[str, Any]) -> None:
        for hook in hooks:
            result = hook(payload)
            if asyncio.iscoroutine(result):
                await result

    async def run(
        self,
        tool_name: str,
        args: dict[str, Any],
        func: Callable[[], Any],
    ) -> Any:
        self.raise_if_cancelled()
        if self._policy.permission_mode != "execute":
            await self._emit(
                "tool_denied",
                {"tool": tool_name, "args": args, "reason": "permission_mode"},
            )
            raise ToolGateDenied(
                f"Tool execution disabled by permission mode '{self._policy.permission_mode}'."
            )
        if self._matches_denied(tool_name) or not self._matches_allowed(tool_name):
            await self._emit(
                "tool_denied",
                {"tool": tool_name, "args": args, "reason": "not_allowed"},
            )
            raise ToolGateDenied(f"Tool '{tool_name}' is not allowed.")

        payload = {"tool": tool_name, "args": args}
        await self._emit("tool_pre", payload)
        await self._run_hooks(self._pre_hooks, payload)
        start = time.monotonic()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = await asyncio.to_thread(func)
                if inspect.isawaitable(result):
                    result = await result
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            await self._emit(
                "tool_post",
                {
                    "tool": tool_name,
                    "args": args,
                    "ok": False,
                    "duration_ms": duration_ms,
                    "error": repr(exc),
                },
            )
            raise
        duration_ms = int((time.monotonic() - start) * 1000)
        await self._emit(
            "tool_post",
            {
                "tool": tool_name,
                "args": args,
                "ok": True,
                "duration_ms": duration_ms,
            },
        )
        await self._run_hooks(self._post_hooks, payload)
        return result
