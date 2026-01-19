from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import discord
from discord.ext import commands
import yt_dlp

from music import yt_search_results
from utils import BOT_PREFIX, defer_interaction, safe_reply, sanitize, humanize_delta, tag_error_text

log = logging.getLogger(__name__)


class SearchPlay(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._search_cache: dict[tuple[int, int, int], tuple[datetime, dict[str, object]]] = {}

    def _cache_key(self, ctx: commands.Context) -> tuple[int, int, int]:
        guild_id = ctx.guild.id if ctx.guild else 0
        channel_id = getattr(ctx.channel, "id", 0) if ctx.channel else 0
        author_id = getattr(ctx.author, "id", 0)
        return (guild_id, channel_id, author_id)

    def store_search_result(self, ctx: commands.Context, result: dict[str, object]) -> None:
        now = datetime.now(timezone.utc)
        for key, (timestamp, _) in list(self._search_cache.items()):
            if now - timestamp > timedelta(minutes=10):
                self._search_cache.pop(key, None)
        self._search_cache[self._cache_key(ctx)] = (datetime.now(timezone.utc), result)

    def pop_search_result(self, ctx: commands.Context, ttl: timedelta = timedelta(minutes=2)) -> dict[str, object] | None:
        key = self._cache_key(ctx)
        entry = self._search_cache.pop(key, None)
        if not entry:
            return None
        timestamp, result = entry
        if datetime.now(timezone.utc) - timestamp > ttl:
            return None
        return result

    @commands.hybrid_command(
        name="searchplay",
        description="Search for tracks without adding to the queue",
        help=(
            "Search YouTube with yt-dlp and list candidates without queuing anything. "
            "Use this when you want to compare results before deciding what to play.\n\n"
            "**Usage**: `/searchplay <query>`\n"
            "**Examples**: `/searchplay Escort もっぴーさうんど`\n"
            f"`{BOT_PREFIX}searchplay Escort`"
        ),
        extras={
            "category": "Music",
            "pro": (
                "Runs a yt-dlp search and lists up to five candidates with durations. "
                "Pick one result and pass its URL to /play when you are ready."
            ),
        },
    )
    async def searchplay(self, ctx: commands.Context, *, query: str | None = None) -> None:
        """Search for tracks without adding them to the queue."""
        if ctx.guild is None:
            return await safe_reply(
                ctx, tag_error_text("This command can only be used in a server."), mention_author=False
            )
        await defer_interaction(ctx)

        if not query:
            return await safe_reply(ctx, tag_error_text("Provide a search query."), mention_author=False)

        parsed = urlparse(query)
        if parsed.scheme and parsed.netloc:
            return await safe_reply(
                ctx,
                tag_error_text("This command is for search terms. Use /play for URLs."),
                mention_author=False,
            )
        if query.startswith(("www.", "youtube.com/", "youtu.be/")):
            return await safe_reply(
                ctx,
                tag_error_text("This command is for search terms. Use /play for URLs."),
                mention_author=False,
            )

        try:
            results = await yt_search_results(query, limit=5)
        except yt_dlp.utils.DownloadError:
            return await safe_reply(
                ctx,
                tag_error_text("No results found. Try a different search phrase."),
                mention_author=False,
            )
        except Exception:
            log.exception("searchplay failed for query %s", query)
            return await safe_reply(
                ctx,
                tag_error_text("Search failed. Please try again in a moment."),
                mention_author=False,
            )

        lines: list[str] = []
        structured_results: list[dict[str, object]] = []
        actions: list[dict[str, object]] = []
        for idx, item in enumerate(results, start=1):
            label = f"R{idx}"
            title = sanitize(str(item.get("title") or "Unknown"))
            url = str(item.get("url") or "")
            duration = item.get("duration")
            duration_human = humanize_delta(duration) if isinstance(duration, (int, float)) else "unknown"
            uploader = item.get("uploader")
            details = f"{duration_human}"
            if uploader:
                details += f" — {sanitize(str(uploader))}"
            line = f"{label}. {title}\n{details}\n<{url}>"
            lines.append(line)
            structured_results.append(
                {
                    "label": label,
                    "title": title,
                    "url": url,
                    "duration_s": int(duration) if isinstance(duration, (int, float)) else None,
                    "duration_human": duration_human,
                    "uploader": uploader,
                }
            )
            actions.append({"label": label, "invoke": {"name": "play", "arg": url}})

        description = "\n\n".join(lines)
        if len(description) > 2048:
            description = description[:2045] + "..."
        embed = discord.Embed(
            title="Search results",
            description=description,
            color=0x57F287,
        )
        embed.set_footer(text="Pick a result and run /play <url> to queue it.")

        search_result = {
            "query": query,
            "results": structured_results,
            "actions": actions,
            "note": "Pick a result label and /play its URL to queue it.",
        }
        try:
            ctx.search_result = search_result
        except Exception:
            log.debug("searchplay: unable to attach search_result to context", exc_info=True)
        self.store_search_result(ctx, search_result)

        await safe_reply(
            ctx,
            embed=embed,
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(SearchPlay(bot))
