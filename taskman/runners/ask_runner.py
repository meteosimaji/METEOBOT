from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import discord
from discord.ext import commands

from taskman.models import TaskResult
from taskman.runners.base import RunnerContext
from taskman.toolgate import ToolGateDenied


@dataclass(slots=True)
class TaskContext:
    bot: commands.Bot
    channel: discord.abc.Messageable
    author: discord.abc.User
    guild: discord.Guild | None
    interaction: None = None
    message: None = None

    async def send(self, *args: Any, **kwargs: Any) -> discord.Message:
        return await self.channel.send(*args, **kwargs)

    async def reply(self, *args: Any, **kwargs: Any) -> discord.Message:
        return await self.channel.send(*args, **kwargs)


class AskRunner:
    def __init__(self, ask_cog: Any) -> None:
        self._ask = ask_cog

    async def run(self, ctx: RunnerContext) -> TaskResult:
        request = ctx.task.request
        action = str(request.get("action") or "ask")
        text = request.get("text")
        output_message_id = ctx.task.output_message_id
        task_channel_id = request.get("channel_id")
        runtime = ctx.runtime_context
        runtime_ctx = runtime.get("ctx")
        extra_images = runtime.get("extra_images")

        resolved_ctx = None
        if isinstance(runtime_ctx, commands.Context):
            resolved_ctx = runtime_ctx
        else:
            resolved_ctx = await self._rebuild_context(ctx)
        if resolved_ctx is None:
            return TaskResult(status="failed", result={"error": "Failed to rebuild context"})

        if output_message_id:
            setattr(resolved_ctx, "task_output_message_id", output_message_id)
        if task_channel_id:
            setattr(resolved_ctx, "task_output_channel_id", task_channel_id)
        setattr(resolved_ctx, "task_toolgate", ctx.toolgate)
        setattr(resolved_ctx, "task_update_runner_state", ctx.update_runner_state)
        setattr(resolved_ctx, "task_background", True)

        try:
            await self._ask._ask_impl(
                resolved_ctx,
                action,
                text,
                extra_images=extra_images,
                skip_queue=True,
            )
        except ToolGateDenied as exc:
            await self._ask._reply(
                resolved_ctx,
                content=f"\N{NO ENTRY SIGN} Task stopped: {exc}",
            )
            return TaskResult(status="cancelled", result={"error": str(exc)})

        state_key = request.get("state_key")
        runner_state: dict[str, Any] = {}
        if isinstance(state_key, str):
            response_id = None
            try:
                response_id = self._ask.bot.ai_last_response_id.get(state_key)
            except Exception:
                response_id = None
            if isinstance(response_id, str) and response_id:
                runner_state["previous_response_id"] = response_id
        return TaskResult(status="succeeded", runner_state=runner_state)

    async def cancel(self, ctx: RunnerContext) -> TaskResult | None:
        runner_state = ctx.task.runner_state
        response_id = runner_state.get("openai_response_id") or runner_state.get(
            "previous_response_id"
        )
        if not response_id:
            return TaskResult(status="cancelled", result={"cancelled": True})
        client = getattr(self._ask, "client", None)
        if client is None:
            return TaskResult(status="cancelled", result={"cancelled": True})
        cancel_fn = getattr(getattr(client, "responses", None), "cancel", None)
        if cancel_fn is None:
            return TaskResult(status="cancelled", result={"cancelled": True})
        if asyncio.iscoroutinefunction(cancel_fn):
            await cancel_fn(response_id)
        else:
            await asyncio.to_thread(cancel_fn, response_id)
        merged_state = dict(ctx.task.runner_state)
        merged_state["openai_response_id"] = response_id
        return TaskResult(
            status="cancelled",
            result={"cancelled": True, "response_id": response_id},
            runner_state=merged_state,
        )

    async def _rebuild_context(self, ctx: RunnerContext) -> commands.Context | None:
        request = ctx.task.request
        channel_id = request.get("channel_id")
        message_id = request.get("message_id")
        if isinstance(channel_id, int) and isinstance(message_id, int):
            channel = self._ask.bot.get_channel(channel_id)
            message = await self._ask._fetch_message_from_channel(
                channel_id=channel_id,
                message_id=message_id,
                channel=channel,
                actor=None,
            )
            if message is not None:
                context = await self._ask.bot.get_context(message)
                return context

        channel_obj = None
        if isinstance(channel_id, int):
            channel_obj = self._ask.bot.get_channel(channel_id)
            if channel_obj is None:
                try:
                    channel_obj = await self._ask.bot.fetch_channel(channel_id)
                except Exception:
                    channel_obj = None
        if channel_obj is None:
            return None
        guild = getattr(channel_obj, "guild", None)
        author_id = request.get("author_id")
        author = None
        if isinstance(author_id, int):
            author = self._ask.bot.get_user(author_id)
            if author is None:
                try:
                    author = await self._ask.bot.fetch_user(author_id)
                except Exception:
                    author = None
        if author is None:
            author = self._ask.bot.user
        if author is None:
            return None
        fallback_ctx = TaskContext(
            bot=self._ask.bot,
            channel=channel_obj,
            author=author,
            guild=guild if isinstance(guild, discord.Guild) else None,
        )
        return fallback_ctx  # type: ignore[return-value]
