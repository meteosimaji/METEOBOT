import logging

from discord.ext import commands

from utils import BOT_PREFIX, safe_reply, tag_error_text

log = logging.getLogger(__name__)


class OperatorCommand(commands.Cog):
    """Expose the operator browser panel."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.hybrid_command(  # type: ignore[arg-type]
        name="operator",
        description="Open the operator panel for manual browser control.",
        usage="",
        help=(
            "Open the operator panel link for this channel's browser session. "
            "Use it when a site needs a login, CAPTCHA, or manual navigation.\n\n"
            "**Usage**: `/operator`\n"
            "**Examples**: `/operator`\n"
            f"`{BOT_PREFIX}operator`"
        ),
        extras={
            "category": "AI",
            "destination": "Generate the operator panel link for manual browser control.",
            "plus": "Use this when the browser needs a login, CAPTCHA, or manual input.",
            "pro": (
                "Creates a signed, time-limited operator panel URL that lets you drive the "
                "same Playwright session used by /ask, including logins and manual navigation."
            ),
        },
    )
    async def operator(self, ctx: commands.Context) -> None:
        ask_cog = self.bot.get_cog("Ask")
        if not ask_cog or not hasattr(ask_cog, "handle_operator_command"):
            await safe_reply(
                ctx,
                tag_error_text("Operator panel support isn't available right now."),
                mention_author=False,
                ephemeral=True,
            )
            return

        try:
            await ask_cog.handle_operator_command(ctx)
        except Exception:
            log.exception("Failed to open operator panel")
            await safe_reply(
                ctx,
                tag_error_text("Failed to open the operator panel."),
                mention_author=False,
                ephemeral=True,
            )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(OperatorCommand(bot))
