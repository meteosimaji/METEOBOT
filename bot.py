"""Discord bot entry point."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from events import EventInfo
from utils import BOT_PREFIX, SuggestionView, build_suggestions, safe_reply

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# write logs both to console and to a persistent file for later review
file_handler = logging.FileHandler("bot.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(file_handler)

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

COMMANDS_PATH = BASE_DIR / "commands"
COGS_PATH = BASE_DIR / "cogs"
EVENTS_PATH = BASE_DIR / "events"


class Bot(commands.Bot):
    """Bot implementation with async extension loading."""

    def __init__(self, prefix: str) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.presences = True
        super().__init__(
            command_prefix=commands.when_mentioned_or(prefix),
            intents=intents,
            help_command=None,
        )
        self.launch_time = datetime.now(timezone.utc)
        self.events: list[EventInfo] = []

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if not self.user:
            return

        content = (message.content or "").strip()

        mention_prefix = re.compile(rf"^\s*<@!?{self.user.id}>\s+")
        if mention_prefix.match(content):
            ctx = await self.get_context(message)
            if getattr(ctx, "command", None) is not None:
                await self.invoke(ctx)
            return

        await self.process_commands(message)

    async def on_ready(self) -> None:
        """Log when the bot has successfully logged in."""
        if self.user:
            log.info("Logged in as %s (ID %s)", self.user, self.user.id)
        else:
            log.info("Logged in")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.competing, name="/help"
            )
        )

    async def setup_hook(self) -> None:  # type: ignore[override]
        successes, failures = await self.load_all_extensions()
        log.info("Extensions loaded: %d success, %d failed", len(successes), len(failures))
        if failures:
            log.info("Failed extensions: %s", ", ".join(failures))
        ask_cog = self.get_cog("Ask")
        if ask_cog and hasattr(ask_cog, "start_operator_server"):
            try:
                await ask_cog.start_operator_server()
            except Exception:
                log.exception("Failed to start operator server")
        synced = await self.tree.sync()
        names = ", ".join(cmd.name for cmd in synced)
        log.info("Synced %d application command(s): %s", len(synced), names)

    async def load_all_extensions(self) -> tuple[list[str], list[str]]:
        """Load every extension under the commands and cogs directories."""

        successes: list[str] = []
        failures: list[str] = []

        paths = [COMMANDS_PATH, COGS_PATH, EVENTS_PATH]
        extensions: list[str] = []

        # Collect all potential extensions first so we can report any that
        # weren't attempted due to errors during discovery.
        for base in paths:
            if not base.exists():
                continue
            for file in sorted(base.glob("*.py")):
                if file.name.startswith("_") or file.name == "__init__.py":
                    continue
                extensions.append(f"{base.name}.{file.stem}")

        for ext in extensions:
            try:
                await self.load_extension(ext)
                log.info("Loaded extension %s", ext)
                successes.append(ext)
            except Exception:
                log.exception("Failed to load extension %s", ext)
                failures.append(ext)

        # Log a summary showing exactly how many extensions were discovered
        # versus how many successfully loaded. This helps diagnose missing
        # commands like the purge module.
        log.info("Discovered %d extensions", len(extensions))
        return successes, failures

    async def on_command_error(  # type: ignore[override]
        self, ctx: commands.Context, error: commands.CommandError
    ) -> None:
        """Send a friendly notice when a prefix command is missing."""

        if isinstance(error, commands.CommandNotFound):
            invoked = (getattr(ctx, "invoked_with", "") or "").lower()
            prefix = getattr(ctx, "prefix", None) or BOT_PREFIX
            suggestions: list[str] = []
            extras: list[str] = []
            if invoked:
                suggestions, extras = build_suggestions(
                    invoked, self.commands, getattr(self, "events", [])
                )

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
            await safe_reply(ctx, message, mention_author=False, view=view)
            return

        await super().on_command_error(ctx, error)

    async def on_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        """Handle application command errors gracefully."""

        if isinstance(error, app_commands.TransformerError):
            message = (
                "I couldn't understand one of the options you entered. "
                "Please pick from the autocomplete list or try /help for details."
            )
        else:
            log.exception("Application command failed", exc_info=error)
            message = (
                "Something went wrong while running that command. "
                "Please try again or see /help for available options."
            )

        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)

def main() -> None:
    """Bot startup sequence."""

    load_dotenv()
    token = os.getenv("DISCORD_BOT_TOKEN")
    prefix = os.getenv("BOT_PREFIX", "c!")
    if not token or token.startswith("YOUR_"):
        raise SystemExit("ERROR: valid DISCORD_BOT_TOKEN not set")

    bot = Bot(prefix)
    bot.run(token)


if __name__ == "__main__":
    main()
