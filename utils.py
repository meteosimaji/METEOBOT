import os
import re

import discord
from discord.ext import commands
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence, Tuple

BOT_PREFIX = os.getenv("BOT_PREFIX", "c!")
LONG_VIEW_TIMEOUT_S = 60 * 60 * 24 * 7  # 7 days
SUGGESTION_VIEW_TIMEOUT_S = LONG_VIEW_TIMEOUT_S
ASK_ERROR_TAG = "\u2063ASKERR\u2063"


def humanize_delta(seconds: float) -> str:
    seconds = int(seconds)
    units = [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]
    parts = []
    for suffix, size in units:
        value, seconds = divmod(seconds, size)
        if value:
            parts.append(f"{value}{suffix}")
    return " ".join(parts) if parts else "0s"


def format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def tag_error_embed(embed: discord.Embed) -> discord.Embed:
    footer_text = ""
    if embed.footer and embed.footer.text:
        footer_text = embed.footer.text
    if ASK_ERROR_TAG not in footer_text:
        embed.set_footer(text=f"{footer_text} {ASK_ERROR_TAG}".strip() if footer_text else ASK_ERROR_TAG)
    return embed


def tag_error_text(text: str) -> str:
    if ASK_ERROR_TAG in text:
        return text
    return f"{text}\n{ASK_ERROR_TAG}"


def error_embed(title: str = "⚠️ Error", desc: str = "Something went wrong.") -> discord.Embed:
    return tag_error_embed(discord.Embed(title=title, description=desc, color=0xFF0000))


async def defer_interaction(ctx: commands.Context) -> None:
    """Show a 'processing' state while a command runs."""
    if ctx.interaction and not ctx.interaction.response.is_done():
        await ctx.interaction.response.defer(thinking=True)
    else:
        await ctx.typing()


async def safe_reply(ctx: commands.Context, *args, **kwargs):
    """Send an ephemeral reply when possible.

    If the context has an interaction, pass through the ``ephemeral`` flag to
    ``ctx.reply``. Otherwise, fall back to ``ctx.reply``/``ctx.send`` without the
    flag to avoid ``TypeError`` in prefix commands.
    """
    ephemeral = kwargs.pop("ephemeral", False)
    if ctx.interaction:
        return await ctx.reply(*args, ephemeral=ephemeral, **kwargs)
    func = getattr(ctx, "reply", None) or ctx.send
    return await func(*args, **kwargs)


async def ensure_voice(ctx: commands.Context) -> bool:
    """Ensure the author and bot share a voice channel.

    Sends a reply prompting the user to join the bot's channel when the check
    fails and returns ``False``. On success returns ``True``.
    """

    guild = ctx.guild
    author = ctx.author

    # In guild channels, ctx.author is typically a Member, but discord.py types
    # it as User | Member. Guard so mypy knows `voice` is present.
    if (
        not guild
        or not isinstance(author, discord.Member)
        or not author.voice
        or not guild.voice_client
        or author.voice.channel != guild.voice_client.channel
    ):
        await safe_reply(
            ctx,
            tag_error_text("Join my voice channel first."),
            mention_author=False,
            ephemeral=True,
        )
        return False
    return True


def sanitize(text: str) -> str:
    return text.replace("@everyone", "@​everyone").replace("@here", "@​here")


MD_CHARS_RE = re.compile(r"[\*_~`]")


def strip_markdown(text: str) -> str:
    """Remove basic Discord markdown characters for searching."""
    return MD_CHARS_RE.sub("", text)


def chunk_string(items, sep: str = ", ", max_len: int = 1000) -> list[str]:
    """Split items into chunks not exceeding ``max_len`` characters."""
    chunks: list[str] = []
    buf = ""
    for it in items:
        token = str(it)
        token_len = len(token) if not buf else len(sep) + len(token)
        if not buf:
            buf = token
        elif len(buf) + token_len <= max_len:
            buf += sep + token
        else:
            chunks.append(buf)
            buf = token
    if buf:
        chunks.append(buf)
    return chunks


def build_suggestions(
    query: str,
    commands_iter: Iterable[commands.Command],
    events: Iterable,
    *,
    primary_count: int = 3,
    max_results: int = 10,
    threshold: float = 0.35,
) -> Tuple[List[str], List[str]]:
    """Rank similar commands/events by fuzzy match.

    Always returns up to ``primary_count`` top suggestions (if any exist) based
    on raw similarity so the user sees immediate options. Remaining candidates
    up to ``max_results`` are filtered by ``threshold`` and returned as extras
    for optional expansion (e.g., a "Show more" button). Hidden commands are
    ignored.
    """

    query = query.strip().lower()
    if not query:
        return [], []

    candidates: list[tuple[str, str]] = []
    for cmd in commands_iter:
        if getattr(cmd, "hidden", False):
            continue
        names = [cmd.qualified_name, *getattr(cmd, "aliases", [])]
        candidates.extend((f"/{name}", name) for name in names)

    for event in events:
        event_name = getattr(event, "name", None)
        if not event_name:
            continue
        candidates.append((f"[{event_name}]", event_name))

    ranked: list[tuple[float, str]] = []
    for display, name in candidates:
        match = SequenceMatcher(None, query, name.lower()).ratio()
        if match > 0:
            ranked.append((match, display))

    ranked.sort(key=lambda item: item[0], reverse=True)

    primary = ranked[:primary_count]
    remaining = ranked[primary_count:max_results]
    extras = [(score, display) for score, display in remaining if score >= threshold]

    def fmt(items: Sequence[tuple[float, str]]) -> List[str]:
        return [f"- {display} ({score * 100:.0f}%)" for score, display in items]

    return fmt(primary), fmt(extras)


class SuggestionView(discord.ui.View):
    """Provides a button to reveal additional suggestions."""

    def __init__(self, extras: List[str], *, timeout: float | None = SUGGESTION_VIEW_TIMEOUT_S) -> None:
        super().__init__(timeout=timeout)
        self.extras = extras

    @discord.ui.button(label="Show more", style=discord.ButtonStyle.secondary)
    async def show_more(
        self, interaction: discord.Interaction, _: discord.ui.Button
    ) -> None:
        content = "More similar commands and events:\n" + "\n".join(self.extras)
        await interaction.response.send_message(content, ephemeral=True)
