# Bot

## Setup
1. Copy `.env.example` to `.env` and set:
   - `DISCORD_BOT_TOKEN` — your Discord bot token.
   - `BOT_PREFIX` — prefix for text commands.
   - `OPENAI_TOKEN` — API key for the `/ask` command.

## Commands
- `ask` — Ask the AI anything; attach up to three images (about 3MB max each, i.e., ~3,000,000 bytes, including images on the message you replied to) for analysis, and it will note the current time, tap web search plus a read-only shell for repo context, and use a code interpreter for calculations. Oversized images are automatically resized/compressed toward the limit. Admins can pick action `reset` to clear channel history while non-admin reset requests are treated as normal questions. Reply-based image pickup works best with prefix commands/mentions; slash commands rely on explicitly attached files.
- `help` — Browse commands and events or get detailed help.
- `messages` — Show recent messages in the current channel.
- `ping` — Check the bot's responsiveness with style and speed!
- `purge` — Bulk delete messages using flexible filters.
- `serverinfo` — Display a polished snapshot of the current server.
- `settime` — Configure a server's timezone with interactive buttons.
- `uptime` — Show how long the bot has been running.
- `userinfo` — Show a member's profile with timestamps and roles.

Prefix commands work with either the configured `BOT_PREFIX` or by mentioning the bot (for example, `@Bot ping`). Messages that start with a bot mention or reply to the bot will fall back to `ask` when they are not recognized as commands.

## Events
- _None listed yet_
