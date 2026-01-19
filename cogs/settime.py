import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from utils import BOT_PREFIX, LONG_VIEW_TIMEOUT_S, defer_interaction, safe_reply, tag_error_text

DATA_PATH = Path("guild_timezones.json")  # { "<guild_id>": int_offset }
MIN_OFS, MAX_OFS = -12, 14  # UTC-12 .. UTC+14
TZ_VIEW_TIMEOUT_S = LONG_VIEW_TIMEOUT_S

# Major cities (1-hour increments only)
CITIES_BY_OFS: dict[int, list[str]] = {
    -12: ["Baker Island"],
    -11: ["American Samoa"],
    -10: ["Honolulu"],
    -9: ["Anchorage"],
    -8: ["Los Angeles", "Vancouver"],
    -7: ["Denver"],
    -6: ["Chicago", "Mexico City"],
    -5: ["New York", "Bogotá"],
    -4: ["Santiago", "Caracas"],
    -3: ["São Paulo", "Buenos Aires"],
    -2: ["South Georgia"],
    -1: ["Azores"],
    0: ["London", "Accra"],
    1: ["Berlin", "Paris", "Rome"],
    2: ["Athens", "Cairo"],
    3: ["Moscow", "Riyadh", "Nairobi"],
    4: ["Dubai", "Baku"],
    5: ["Tashkent", "Karachi"],
    6: ["Almaty", "Dhaka"],
    7: ["Bangkok", "Jakarta"],
    8: ["Beijing", "Singapore", "Hong Kong"],
    9: ["Tokyo", "Seoul"],
    10: ["Sydney", "Port Moresby"],
    11: ["Nouméa"],
    12: ["Auckland", "Fiji"],
    13: ["Samoa"],
    14: ["Kiritimati"],
}


def load_tz() -> dict[str, int]:
    if DATA_PATH.exists():
        try:
            return json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_tz(data: dict[str, int]) -> None:
    tmp = DATA_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DATA_PATH)


def get_guild_offset(bot: commands.Bot, guild_id: int) -> int:
    # Default is UTC+9 (Japan)
    cache = getattr(bot, "guild_tz", {})
    key = str(guild_id)
    if isinstance(cache, dict) and key in cache:
        return int(cache[key])
    # Lazy load from file
    data = load_tz()
    ofs = int(data.get(key, 9))
    bot.guild_tz = {**data}
    return ofs


def set_guild_offset(bot: commands.Bot, guild_id: int, ofs: int) -> None:
    data = load_tz()
    data[str(guild_id)] = int(ofs)
    save_tz(data)
    bot.guild_tz = {**data}


def fmt_ofs(ofs: int) -> str:
    sign = "+" if ofs >= 0 else "-"
    return f"UTC{sign}{abs(ofs):02d}:00"


def preview_time(ofs: int) -> str:
    tz = timezone(timedelta(hours=ofs))
    return datetime.now(tz).strftime("%Y-%m-%d (%a) %H:%M:%S")


def make_embed(guild: discord.Guild, ofs: int) -> discord.Embed:
    cities = ", ".join(CITIES_BY_OFS.get(ofs, [])) or "—"
    e = discord.Embed(
        title="🕰️ Server Timezone",
        description="Use the sleek buttons below to adjust the UTC offset an hour at a time.",
        color=0x8E7CC3,
    )
    e.add_field(name="Current Offset", value=f"**{fmt_ofs(ofs)}**", inline=True)
    e.add_field(name="Major Cities", value=cities, inline=True)
    e.add_field(name="Local Time Preview", value=f"`{preview_time(ofs)}`", inline=False)
    e.set_footer(text=f"{guild.name} • UTC-12 to UTC+14 (1h steps)")
    return e


class TzView(discord.ui.View):
    def __init__(self, bot: commands.Bot, guild: discord.Guild, start_ofs: int, *, author_id: int, timeout: float | None = TZ_VIEW_TIMEOUT_S):
        super().__init__(timeout=timeout)
        self.bot = bot
        self.guild = guild
        self.ofs = start_ofs
        self.author_id = author_id
        self.message: discord.Message | None = None

    async def update(self, interaction: discord.Interaction):
        await interaction.response.edit_message(embed=make_embed(self.guild, self.ofs), view=self)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id != self.author_id:
            await interaction.response.send_message(
                tag_error_text("Only the admin who invoked this can use these buttons."), ephemeral=True
            )
            return False
        if interaction.user and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                tag_error_text("Only administrators can use this action."), ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="-1h", style=discord.ButtonStyle.secondary)
    async def minus(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.ofs = max(MIN_OFS, self.ofs - 1)
        await self.update(interaction)

    @discord.ui.button(label="+1h", style=discord.ButtonStyle.secondary)
    async def plus(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.ofs = min(MAX_OFS, self.ofs + 1)
        await self.update(interaction)

    @discord.ui.button(label="Save", style=discord.ButtonStyle.success)
    async def save(self, interaction: discord.Interaction, button: discord.ui.Button):
        set_guild_offset(self.bot, self.guild.id, self.ofs)
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        await interaction.response.edit_message(
            embed=make_embed(self.guild, self.ofs).set_footer(text=f"{self.guild.name} • Settings saved"),
            view=self,
        )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        await interaction.response.edit_message(content="Canceled.", embed=None, view=self)


class SetTime(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Ensure cache attribute exists
        if not hasattr(self.bot, "guild_tz"):
            self.bot.guild_tz: dict[str, int] = load_tz()

    @commands.hybrid_command(
        name="settime",
        description="Configure the server's timezone (admin only)",
        help=(
            "Change the timezone used for this server's commands. Administrator permissions required.\n\n"
            "**Usage**: `/settime` then use the -1h/+1h buttons to adjust the offset.\n"
            "Press “Save” to apply or “Cancel” to abort.\n"
            "**Examples**: `/settime`\n"
            f"`{BOT_PREFIX}settime`"
        ),
        extras={
            "category": "Moderation",
            "short": "Set timezone",
            "long_help": (
                "Administrators can set the timezone for commands like `/uptime`. "
                "Adjust the UTC offset using the buttons in one‑hour steps, then press "
                "Save to store the new setting."
            ),
        },
    )
    @commands.has_guild_permissions(administrator=True)
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    @app_commands.guild_only()
    async def settime(self, ctx: commands.Context):
        await defer_interaction(ctx)
        if ctx.guild is None:
            await safe_reply(
                ctx,
                tag_error_text("Please use this inside a server."),
                mention_author=False,
                ephemeral=True,
            )
            return
        ofs = get_guild_offset(self.bot, ctx.guild.id)
        view = TzView(self.bot, ctx.guild, ofs, author_id=ctx.author.id)
        embed = make_embed(ctx.guild, ofs)

        await safe_reply(ctx, embed=embed, view=view, ephemeral=True)


async def setup(bot: commands.Bot):
    await bot.add_cog(SetTime(bot))
