import discord
from discord import app_commands
from discord.ext import commands
from utils import BOT_PREFIX, SuggestionView, build_suggestions, defer_interaction
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Iterable, List, Tuple

CATEGORIES = [
    ("All", "\U0001F50E"),
    ("AI", "\U0001F916"),
    ("Games", "\U0001F3AE"),
    ("Moderation", "\U0001F6E1\ufe0f"),
    ("Music", "\U0001F3B5"),
    ("Tools", "\U0001F9F0"),
    ("Utility", "\u2699\ufe0f"),
]

GENRE_DESC = {
    "All": (
        "Use the buttons below to browse command categories and discover "
        "everything the bot can do."
    ),
    "AI": (
        "Commands and events powered by AI for chat, images or translation."
    ),
    "Games": (
        "Playful commands and game-related events to enjoy with friends "
        "or track scores."
    ),
    "Moderation": (
        "Server management commands for admins and moderators."
    ),
    "Music": (
        "Commands for playing tunes, managing the queue and adjusting playback."
    ),
    "Tools": (
        "Handy converters and generators for media and text."
    ),
    "Utility": (
        "General information and helper commands for everyday use."
    ),
}


def make_help_pages(bot: commands.Bot, category: str) -> List[discord.Embed]:
    title = "\U0001F916 Bot Help"
    if category != "All":
        title += f" - {category}"
    commands_iter = sorted(
        [c for c in bot.commands if not c.hidden], key=lambda c: c.qualified_name
    )
    events = sorted(getattr(bot, "events", []), key=lambda e: e.name)
    entries: List[Tuple[str, str]] = []
    for c in commands_iter:
        cat = c.extras.get("category", "Utility")
        if category != "All" and cat != category:
            continue
        text = c.description if category == "All" else (c.help or c.description or "-")
        entries.append((f"/{c.qualified_name}", text or "-"))
    for e in events:
        cat = getattr(e, "category", "Utility")
        if category != "All" and cat != category:
            continue
        text = e.destination if category == "All" else (e.plus or e.destination or "-")
        entries.append((f"[{e.name}]", text or "-"))
    pages: List[discord.Embed] = []
    per_page = 25
    total_pages = max(1, (len(entries) + per_page - 1) // per_page)
    for i in range(total_pages):
        embed = discord.Embed(
            title=title,
            description=GENRE_DESC.get(category, ""),
            color=0xE67E22,
            timestamp=datetime.now(timezone.utc),
        )
        for name, value in entries[i * per_page : (i + 1) * per_page]:
            embed.add_field(name=name, value=value, inline=False)
        embed.set_footer(text=f"Crafted with care ✨ • Page {i + 1}/{total_pages}")
        pages.append(embed)
    return pages


def make_command_help_embed(bot: commands.Bot, name: str) -> discord.Embed | None:
    """Build a help embed for a command or event."""
    cmd = bot.get_command(name)
    if cmd and not cmd.hidden:
        extras = getattr(cmd, "extras", {})
        embed = discord.Embed(
            title=f"\U0001F4D6 Command Help: /{cmd.qualified_name}",
            color=0xE67E22,
            timestamp=datetime.now(timezone.utc),
        )
        destination = cmd.description
        if destination:
            embed.add_field(name="Destination", value=destination, inline=False)
        if cmd.help:
            embed.add_field(name="Plus", value=cmd.help, inline=False)
        pro = extras.get("pro")
        if pro:
            embed.add_field(name="Pro", value=pro, inline=False)
        if cmd.usage:
            embed.add_field(
                name="Usage",
                value=f"/{cmd.qualified_name} {cmd.usage}",
                inline=False,
            )
        embed.set_footer(text="Crafted with care ✨")
        return embed
    info = next((e for e in getattr(bot, "events", []) if e.name == name), None)
    if not info:
        return None
    embed = discord.Embed(
        title=f"\U0001F4D6 Event Help: {info.name}",
        color=0xE67E22,
        timestamp=datetime.now(timezone.utc),
    )
    if info.destination:
        embed.add_field(name="Destination", value=info.destination, inline=False)
    if info.plus:
        embed.add_field(name="Plus", value=info.plus, inline=False)
    if info.pro:
        embed.add_field(name="Pro", value=info.pro, inline=False)
    if info.example:
        embed.add_field(name="Example", value=info.example, inline=False)
    embed.set_footer(text="Crafted with care ✨")
    return embed


class HelpView(discord.ui.View):
    def __init__(self, bot: commands.Bot, category: str = "All") -> None:
        super().__init__(timeout=None)
        self.bot = bot
        self.category = category
        self.pages = make_help_pages(bot, category)
        self.page = 0
        self.update_buttons()

    def update_buttons(self) -> None:
        self.btn_all.disabled = self.category == "All"
        self.btn_ai.disabled = self.category == "AI"
        self.btn_games.disabled = self.category == "Games"
        self.btn_mod.disabled = self.category == "Moderation"
        self.btn_music.disabled = self.category == "Music"
        self.btn_tools.disabled = self.category == "Tools"
        self.btn_util.disabled = self.category == "Utility"
        single_page = len(self.pages) == 1
        self.btn_prev.disabled = single_page
        self.btn_next.disabled = single_page

    async def redraw(self, interaction: discord.Interaction) -> None:
        self.pages = make_help_pages(self.bot, self.category)
        self.page = min(self.page, len(self.pages) - 1)
        self.update_buttons()
        await interaction.response.edit_message(
            embed=self.pages[self.page], view=self
        )

    @discord.ui.button(label="All \U0001F50E", style=discord.ButtonStyle.secondary)
    async def btn_all(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.category = "All"
        self.page = 0
        await self.redraw(interaction)

    @discord.ui.button(label="AI \U0001F916", style=discord.ButtonStyle.secondary)
    async def btn_ai(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.category = "AI"
        self.page = 0
        await self.redraw(interaction)

    @discord.ui.button(label="Games \U0001F3AE", style=discord.ButtonStyle.secondary)
    async def btn_games(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.category = "Games"
        self.page = 0
        await self.redraw(interaction)

    @discord.ui.button(label="Moderation \U0001F6E1\ufe0f", style=discord.ButtonStyle.secondary)
    async def btn_mod(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.category = "Moderation"
        self.page = 0
        await self.redraw(interaction)

    @discord.ui.button(label="Music \U0001F3B5", style=discord.ButtonStyle.secondary)
    async def btn_music(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.category = "Music"
        self.page = 0
        await self.redraw(interaction)

    @discord.ui.button(label="Tools \U0001F9F0", style=discord.ButtonStyle.secondary)
    async def btn_tools(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.category = "Tools"
        self.page = 0
        await self.redraw(interaction)

    @discord.ui.button(label="Utility \u2699\ufe0f", style=discord.ButtonStyle.secondary)
    async def btn_util(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.category = "Utility"
        self.page = 0
        await self.redraw(interaction)

    @discord.ui.button(label="Prev \u2B05\uFE0F", style=discord.ButtonStyle.primary)
    async def btn_prev(self, interaction: discord.Interaction, _: discord.ui.Button):
        if len(self.pages) > 1:
            if self.page == 0:
                self.page = len(self.pages) - 1
            else:
                self.page -= 1
            await self.redraw(interaction)

    @discord.ui.button(label="Next \u27A1\uFE0F", style=discord.ButtonStyle.primary)
    async def btn_next(self, interaction: discord.Interaction, _: discord.ui.Button):
        if len(self.pages) > 1:
            if self.page == len(self.pages) - 1:
                self.page = 0
            else:
                self.page += 1
            await self.redraw(interaction)


class Help(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def _suggestions(self, query: str) -> Tuple[List[str], List[str]]:
        return build_suggestions(query, self.bot.commands, getattr(self.bot, "events", []))

    def _autocomplete_items(self) -> Iterable[tuple[str, str, str]]:
        for cmd in self.bot.commands:
            if getattr(cmd, "hidden", False):
                continue
            summary = cmd.description or "Command"
            yield cmd.qualified_name, f"/{cmd.qualified_name}", summary
        for event in getattr(self.bot, "events", []):
            name = getattr(event, "name", None)
            if not name:
                continue
            summary = getattr(event, "destination", "Event") or "Event"
            yield name, f"[{name}]", summary

    def _rank_autocomplete(self, query: str) -> List[tuple[str, str]]:
        query = query.strip().lower()
        ranked: list[tuple[float, str, str]] = []
        for value, label, summary in self._autocomplete_items():
            score = 0.0
            if query:
                score = SequenceMatcher(None, query, value.lower()).ratio()
                if value.lower().startswith(query):
                    score += 0.5
            ranked.append((score, value, f"{label} – {summary}"))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not query:
            # When no query is provided, present alphabetically to make browsing easy.
            ranked.sort(key=lambda item: item[1])
        return [(value, label) for _, value, label in ranked[:25]]

    async def help_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        suggestions = self._rank_autocomplete(current)
        return [app_commands.Choice(name=label[:100], value=value) for value, label in suggestions]

    async def _reply(self, ctx: commands.Context, **kwargs) -> None:
        # Avoid passing invalid view objects to discord responses.
        view = kwargs.get("view", discord.utils.MISSING)
        if view is None or not isinstance(view, discord.ui.View):
            kwargs.pop("view", None)
        if ctx.interaction:
            if ctx.interaction.response.is_done():
                await ctx.interaction.followup.send(**kwargs)
            else:
                await ctx.interaction.response.send_message(**kwargs)
        else:
            await ctx.send(**kwargs)

    @commands.hybrid_command(
        name="help",
        description="Browse commands and events or get detailed help",
        help=(
            "Browse commands or get details for a specific one. Without an "
            "argument an interactive menu appears with buttons for each "
            "category.\n\n"
            "**Usage**: `/help [command]`\n"
            "**Examples**: `/help ping`\n"
            f"`{BOT_PREFIX}help ping`"
        ),
        extras={
            "category": "Utility",
            "pro": (
                "Running `/help` opens a button-driven menu showing AI, Games, "
                "Moderation, Music, Tools and Utility sections. Selecting a "
                "category filters the commands. `/help <command>` jumps straight "
                "to a detailed embed with usage notes and any extra tips."
            )
        },
    )
    @app_commands.autocomplete(command=help_autocomplete)
    async def help(self, ctx: commands.Context, *, command: str | None = None) -> None:
        """Show help for a command or list commands."""
        await defer_interaction(ctx)
        if command:
            embed = make_command_help_embed(self.bot, command)
            if embed:
                return await self._reply(ctx, embed=embed)
            suggestions, extras = self._suggestions(command.lower())
            prefix = getattr(ctx, "prefix", None) or BOT_PREFIX
            if suggestions:
                message = (
                    "Command not found. Did you mean:\n"
                    + "\n".join(suggestions)
                    + (
                        "\n…and more similar matches. Use the button below to see them."
                        if extras
                        else ""
                    )
                    + f"\n\nUse /help or {prefix}help to explore more commands and events."
                )
            else:
                message = (
                    "Command not found.\n"
                    f"Use /help or {prefix}help to search for commands and events."
                )
            view = SuggestionView(extras) if extras else None
            return await self._reply(ctx, content=message, view=view)
        else:
            view = HelpView(self.bot, "All")
            await self._reply(ctx, embed=view.pages[0], view=view)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Help(bot))
