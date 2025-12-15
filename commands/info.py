"""Information commands for users and servers."""

from datetime import datetime, timezone

import discord
from discord.ext import commands

from utils import BOT_PREFIX, defer_interaction


def _summarize_mentions(items: list[str], *, limit: int = 950, separator: str = ", ") -> str:
    if not items:
        return "No roles"

    parts: list[str] = []
    total = 0
    for item in items:
        addition = len(separator) if parts else 0
        next_total = total + addition + len(item)
        if next_total > limit:
            remaining = len(items) - len(parts)
            if remaining > 0:
                parts.append(f"(+{remaining} more)")
            break

        parts.append(item)
        total = next_total

    return separator.join(parts)


class Info(commands.Cog):
    """Show details about users or the current server."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(
        name="serverinfo",
        description="Display a polished snapshot of the current server.",
        help=(
            "Get a rich overview of the server including members, channels, "
            "boosts and creation date. Run it in a server to see counts and "
            "top roles.\n\n"
            "**Usage**: `/serverinfo`\n"
            f"`{BOT_PREFIX}serverinfo`"
        ),
        extras={
            "category": "Utility",
            "pro": (
                "Highlights the most important server metrics at a glance: member "
                "breakdown, channel counts, boosts, features and creation date "
                "with timestamps you can hover for local time."
            ),
            "destination": "Posts a rich server summary embed in the current channel.",
            "plus": "Run inside a server to see member, channel, boost, and role details with timestamps.",
        },
    )
    async def serverinfo(self, ctx: commands.Context) -> None:
        guild = ctx.guild
        if not guild:
            if ctx.interaction and ctx.interaction.response.is_done():
                await ctx.interaction.followup.send(
                    "Run this command inside a server to view its details.",
                    ephemeral=True,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            elif ctx.interaction:
                await ctx.interaction.response.send_message(
                    "Run this command inside a server to view its details.",
                    ephemeral=True,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await ctx.reply(
                    "Run this command inside a server to view its details.",
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            return

        await defer_interaction(ctx)

        humans = bots = None
        if guild.members:
            humans = sum(1 for member in guild.members if not member.bot)
            bots = sum(1 for member in guild.members if member.bot)
        text_channels = len(guild.text_channels)
        voice_channels = len(guild.voice_channels)
        stage_channels = len(getattr(guild, "stage_channels", []))
        forum_channels = len(getattr(guild, "forum_channels", []))
        total_channels = text_channels + voice_channels + stage_channels + forum_channels
        top_roles = [role.mention for role in guild.roles[1:6]]
        created_ts = int(guild.created_at.replace(tzinfo=timezone.utc).timestamp())

        features = guild.features
        feature_list = ", ".join(sorted(features)) if features else "None"

        embed = discord.Embed(
            title="\U0001F3F0 Server Overview",
            description=f"Insights for **{guild.name}**",
            color=0x3683FF,
            timestamp=datetime.now(timezone.utc),
        )
        embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
        embed.add_field(name="Owner", value=guild.owner.mention if guild.owner else "-", inline=True)
        embed.add_field(name="Server ID", value=str(guild.id), inline=True)
        member_lines = [f"Total: {guild.member_count}"]
        if humans is not None and bots is not None:
            member_lines.append(f"Humans: {humans} | Bots: {bots}")
        else:
            member_lines.append("Humans/Bots: unavailable (member list not cached)")

        embed.add_field(name="Members", value="\n".join(member_lines), inline=False)
        embed.add_field(
            name="Channels",
            value=(
                f"Text: {text_channels}\n"
                f"Voice: {voice_channels}\n"
                f"Stage: {stage_channels}\n"
                f"Forum: {forum_channels}\n"
                f"Total: {total_channels}"
            ),
            inline=True,
        )
        embed.add_field(
            name="Boosts",
            value=f"Level {guild.premium_tier} ({guild.premium_subscription_count or 0} boosts)",
            inline=True,
        )
        embed.add_field(
            name="Created",
            value=f"<t:{created_ts}:F>\n<t:{created_ts}:R>",
            inline=False,
        )
        embed.add_field(
            name="Roles",
            value=", ".join(top_roles) if top_roles else "No roles",
            inline=False,
        )
        embed.add_field(name="Features", value=feature_list, inline=False)
        embed.set_footer(text="Crafted with care ✨")

        if ctx.interaction:
            if ctx.interaction.response.is_done():
                await ctx.interaction.followup.send(
                    embed=embed, allowed_mentions=discord.AllowedMentions.none()
                )
            else:
                await ctx.interaction.response.send_message(
                    embed=embed, allowed_mentions=discord.AllowedMentions.none()
                )
        else:
            await ctx.reply(
                embed=embed,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    @commands.hybrid_command(
        name="userinfo",
        description="Show a member's profile with timestamps and roles.",
        help=(
            "Inspect a member's account age, server join date, top role, roles "
            "and activity. The target defaults to you if left empty.\n\n"
            "**Usage**: `/userinfo [member]`\n"
            f"`{BOT_PREFIX}userinfo @name`"
        ),
        extras={
            "category": "Utility",
            "pro": (
                "Displays both absolute and relative timestamps for account "
                "creation and server join dates, plus role highlights and current "
                "status/activity when available."
            ),
            "destination": "Posts a detailed profile embed to the channel where it's invoked.",
            "plus": "Target defaults to you; include a member mention or ID to inspect others in the server.",
        },
    )
    async def userinfo(self, ctx: commands.Context, member: discord.Member | discord.User | None = None) -> None:  # type: ignore[override]
        target = member or ctx.author

        await defer_interaction(ctx)

        created_ts = int(target.created_at.replace(tzinfo=timezone.utc).timestamp())
        joined_value = "Not in this server"
        joined_ts: int | None = None
        display_roles = "No roles"
        top_role = "-"

        if isinstance(target, discord.Member):
            joined_ts = int(target.joined_at.replace(tzinfo=timezone.utc).timestamp()) if target.joined_at else None
            roles = [role.mention for role in target.roles[1:]]
            if roles:
                display_roles = _summarize_mentions(roles)
            top_role = target.top_role.mention if target.top_role else "-"
            if joined_ts:
                joined_value = f"<t:{joined_ts}:F>\n<t:{joined_ts}:R>"

        embed = discord.Embed(
            title="\U0001F464 User Profile",
            description=f"Details for **{target}**",
            color=target.colour if isinstance(target, discord.Member) else 0x3683FF,
            timestamp=datetime.now(timezone.utc),
        )
        avatar_url = target.display_avatar.url if hasattr(target, "display_avatar") else None
        embed.set_thumbnail(url=avatar_url)
        embed.add_field(name="User", value=target.mention if hasattr(target, "mention") else str(target), inline=True)
        embed.add_field(name="User ID", value=str(target.id), inline=True)
        embed.add_field(name="Account Created", value=f"<t:{created_ts}:F>\n<t:{created_ts}:R>", inline=False)
        embed.add_field(name="Joined Server", value=joined_value, inline=False)
        embed.add_field(name="Top Role", value=top_role, inline=True)
        embed.add_field(name="Roles", value=display_roles, inline=False)

        activity = "-"
        if isinstance(target, discord.Member):
            rich_activity = target.activity
            if rich_activity:
                activity = getattr(rich_activity, "name", str(rich_activity))
            intents = getattr(ctx.bot, "intents", None)
            presences_enabled = bool(intents and intents.presences)

            status_value = str(target.status).title()
            if not presences_enabled and status_value == "Offline":
                status_value = "Unknown (presence intent disabled)"
            if activity == "-" and not presences_enabled:
                activity = "Unavailable (presence intent disabled)"

            embed.add_field(name="Status", value=status_value, inline=True)
            embed.add_field(name="Activity", value=activity, inline=True)

        embed.set_footer(text="Crafted with care ✨")

        if ctx.interaction:
            if ctx.interaction.response.is_done():
                await ctx.interaction.followup.send(
                    embed=embed, allowed_mentions=discord.AllowedMentions.none()
                )
            else:
                await ctx.interaction.response.send_message(
                    embed=embed, allowed_mentions=discord.AllowedMentions.none()
                )
        else:
            await ctx.reply(
                embed=embed,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Info(bot), override=True)
