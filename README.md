# Bot

## Setup
1. Copy `.env.example` to `.env` and set:
   - `DISCORD_BOT_TOKEN` — your Discord bot token.
   - `BOT_PREFIX` — prefix for text commands.
   - `OPENAI_TOKEN` — API key for `/ask`, `/image`, and flag reaction translation.

## Commands
- `ask` — Ask the AI anything; attach up to three images (about 3MB max each, i.e., ~3,000,000 bytes, including images on the message you replied to) for analysis, and it will note the current time, tap web search plus a read-only shell for repo context, and use a code interpreter for calculations. Oversized images are automatically resized/compressed toward the limit. Admins can pick action `reset` to clear channel history while non-admin reset requests are treated as normal questions. Reply-based image pickup works best with prefix commands/mentions; slash commands rely on explicitly attached files. Message link fetching respects both user and bot permissions, even across guilds.
- `autoleave` — Toggle Auto Leave (disconnect when idle/bot-only) or view the current state.
- `bye` — Stop playback, clear the queue, and leave voice.
- `help` — Browse commands and events or get detailed help.
- `image` — Generate or edit a 1024x1024 PNG with gpt-image-1.5; attach images, include public HTTPS image URLs, or paste Discord message links with attachments in the prompt to edit (first image is the base, the rest are references; up to 16 inputs, each under 50MB). HEIC/HEIF uploads are accepted and converted internally when `pillow-heif` is available.
- `loopmode` — Set looping to off, track, or queue (or display the current mode).
- `messages` — Show recent messages in the current channel.
- `now` — Show the current track with a progress bar.
- `pause` / `resume` — Pause or resume playback.
- `ping` — Check the bot's responsiveness with style and speed!
- `play` — Queue music by URL or search phrase (YouTube/Spotify/Niconico). Prefix usage also supports file attachments.
- `purge` — Bulk delete messages using flexible filters.
- `queue` — Show interactive queue panel (pause/resume, loop, skip, bye, remove, speed/pitch, toggle Auto Leave).
- `remove` — Remove recently added songs by addition order (1=latest, 2=previous, etc.).
- `seek` — Seek within the current track (e.g. `1:23`, `+30`, `-10`).
- `serverinfo` — Display a polished snapshot of the current server.
- `settime` — Configure a server's timezone with interactive buttons.
- `skip` — Skip the current track.
- `tex` — Render LaTeX to crisp PNGs with both white and transparent backgrounds plus a PDF; both preview variants are attached together. Supports up to four pages (keeps Discord attachments under the cap) and works great when invoked via `/ask` tool-calling.
- `tune` — Set speed and pitch (`/tune 1.2 0.9`).
- `uptime` — Show how long the bot has been running.
- `userinfo` — Show a member's profile with timestamps, roles, and bot status.

### Music prerequisites
- Install **FFmpeg** and ensure `ffmpeg` is on PATH (required for audio decoding).
- Install Python deps: `yt-dlp` and `PyNaCl` (voice).
- Grant the bot **Connect** and **Speak** permissions in voice channels.
- Enable the **Voice States** intent (and any other intents your host requires) so the bot can see voice joins.

Prefix commands work with either the configured `BOT_PREFIX` or by mentioning the bot (for example, `@Bot ping`). Messages that start with a bot mention or reply to the bot will fall back to `ask` when they are not recognized as commands.

### TeX prerequisites
- Install **Tectonic** (required; uses untrusted mode for safety).
- Install **Ghostscript** (`gs`) for PDF → PNG rasterization (multi-page PNGs are supported up to the configured page cap, capped at four pages to stay within Discord's attachment limits).
- For Japanese documents that use `jsarticle`/`uplatex`, switch to `bxjsarticle` (XeTeX-compatible) before rendering.
- Wrap equations with math delimiters (`$...$`, `\[...\]`) or provide a full LaTeX document. Single-line auto-wrap can be enabled via `LATEXBOT_AUTOWRAP=1` if desired.

## Events
- `flag_translate` — Translate a message when a user reacts with a country flag emoji (any channel that supports text chat; includes embeds and images, notifies about cooldowns/permissions, requires permission to read message history).
- `voice_auto_leave` — Leave voice automatically when only bots remain in the voice channel (Auto Leave must be on).
