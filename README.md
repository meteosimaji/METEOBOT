# Bot

## Setup
1. Copy `.env.example` to `.env` and set:
   - `DISCORD_BOT_TOKEN` — your Discord bot token.
   - `BOT_PREFIX` — prefix for text commands.
   - `OPENAI_TOKEN` — API key for `/ask`, `/image`, and flag reaction translation.
   - `ASK_BROWSER_PROFILE_DIR` — directory for `/ask` Playwright persistent profiles (defaults to `data/browser_profiles`).
   - `ASK_WORKSPACE_DIR` — directory for `/ask` workspaces inside the repo (defaults to `data/ask_workspaces`; non-repo paths are ignored).
   - `ASK_WORKSPACE_TTL_S` — workspace time-to-live in seconds (defaults to `86400`).
   - `ASK_WORKSPACE_MAX_BYTES` — total workspace disk cap before LRU cleanup (defaults to `2GiB`).
   - `ASK_WORKSPACE_MAX_TEXT_CHARS` — max characters stored per extracted text file (defaults to `2,000,000`).
   - `ASK_WORKSPACE_MAX_ORIGINAL_BYTES` — max size of original files to keep in workspaces (defaults to `50MiB`).
   - `ASK_CONTAINER_FILE_LINK_TTL_S` — time-to-live (seconds) for `/ask` container output download links (defaults to `1800`).
   - `ASK_CONTAINER_FILE_MAX_BYTES` — max size of a single container output file to mirror (defaults to `200MiB`).
   - `ASK_CONTAINER_FILE_MAX_COUNT` — max number of container output files mirrored per `/ask` run (defaults to `5`).
   - `ASK_CONTAINER_FILE_RETAIN_S` — how long mirrored container files stay on disk (defaults to `86400`).
   - `ASK_LINK_CONTEXT_MAX_ENTRIES` — max stored link-context entries per channel (defaults to `50`).
   - `ASK_LINK_CONTEXT_MAX_PROMPT` — number of recent links included in `/ask` prompt context (defaults to `10`).
   - `ASK_OPERATOR_DEFAULT_URL` — initial URL to open in the operator browser (defaults to `https://www.google.com`).
   - `ASK_OPERATOR_HEADLESS` — whether `/operator` starts headless (`true`/`false`, defaults to `false`).
   - `ASK_OPERATOR_AUTOSTART_XVFB` — when headed mode needs an X server and `$DISPLAY` is missing, start `Xvfb` automatically (`true`/`false`, defaults to `true`).
   - `ASK_OPERATOR_XVFB_SCREEN` — virtual screen size/depth for Xvfb (defaults to `1920x1080x24`).
   - `ASK_OPERATOR_TOKEN_SECRET` — HMAC secret for `/operator` session tokens (defaults to `DISCORD_BOT_TOKEN`, set the same value across instances).
   - `ASK_OPERATOR_INSTANCE_ID` — instance identifier embedded in operator tokens (defaults to a random value on boot).
   - `ASK_OPERATOR_ALLOW_SHARED_TOKENS` — allow operator tokens from other instances (defaults to `false`).
   - `ASK_OPERATOR_TOKEN_MAX_FUTURE_S` — maximum allowed clock skew for operator tokens in seconds (defaults to `300`).
   - `ASK_OPERATOR_TOKEN_TTL_S` — operator panel link TTL in seconds (defaults to `1800`).
   - `ASK_BROWSER_CDP_HEADERS_JSON` — optional JSON object of headers passed to CDP WebSocket connections (for relay auth), e.g. `{"x-openclaw-relay-token":"..."}`.
   - `ASK_BROWSER_CDP_CONNECT_TIMEOUT_S` — CDP connect timeout seconds before fallback/error (defaults to `8`).
   - `ASK_BROWSER_CDP_AUTO_CONFIG` — when true and `/operator` prefers CDP, bot can auto-launch local CDP when `ASK_BROWSER_CDP_URL` is unset (defaults to `false`).
   - `ASK_BROWSER_CDP_AUTO_LAUNCH` — when auto-config is enabled, launch local Chrome/Chromium with `--remote-debugging-port` if needed (defaults to `true`).
   - `ASK_BROWSER_CDP_AUTO_HOST` / `ASK_BROWSER_CDP_AUTO_PORT` — host/port used for auto local CDP launch (defaults `127.0.0.1:0`; port `0` picks a free ephemeral port per operator state).
   - `ASK_BROWSER_CDP_AUTO_LAUNCH_TIMEOUT_S` — grace interval for checking auto-launched browser startup (defaults to `6`).
   - `ASK_BROWSER_CDP_ALLOW_REMOTE` — allow non-localhost CDP endpoints only when set (`true`/`false`, defaults to `false`); remote CDP URLs require `wss://` or `https://`.
   - `ASK_BROWSER_OBSERVE_READ_TIMEOUT_S` — timeout for observation reads like title/ARIA/accessibility snapshots (defaults to `2.5`; set `0` to disable timeout).
   - `ASK_BROWSER_POST_ACTION_LOAD_TIMEOUT_S` — short post-action DOM-content wait for navigation-prone actions (defaults to `1.0`).
   - `ASK_BROWSER_REF_PRIMARY_TIMEOUT_S` — timeout for primary ref extraction (ARIA + merge path) before fallback (defaults to `6.0`; set `0` to disable timeout).
   - `ASK_BROWSER_REF_FALLBACK_TIMEOUT_S` — timeout for clickable-target fallback ref extraction (defaults to `4.5`; set `0` to disable timeout).
   - CDP headers are loaded at process start; restart the bot after changing `ASK_BROWSER_CDP_HEADERS_JSON`.
   - Operator instance IDs persist in `data/operator_instance_id.txt` unless overridden by `ASK_OPERATOR_INSTANCE_ID`. Set a fixed ID for restart-safe links.
   - Shared tokens do **not** share browser state across instances. Use sticky sessions (or a single instance) for `/operator/*` traffic to avoid profile conflicts or separate browser sessions. If `ASK_OPERATOR_ALLOW_SHARED_TOKENS` is enabled, expect a separate browser per instance.
   - `CODEX_BIN` — Codex CLI binary name (defaults to `codex`).
   - `CODEX_RUNS_DIR` — directory for `/codex` workspaces/artifacts (defaults to `data/codex_runs`).
   - `CODEX_ALLOWED_ROOTS` — comma-separated allowlist of repo roots for `/codex` (optional).
   - `CODEX_GITHUB_REPO` — GitHub repo for PR creation (e.g. `owner/repo` or a GitHub URL).
   - `GITHUB_TOKEN` — GitHub token used for pushing branches and opening PRs (required for PR creation).

2. Start the bot with `python bot.py`. Install/update dependencies with
   `pip install -r requirements.txt` and install Playwright browsers before
   running `bot.py`.

## Commands
- `ask` — Ask the AI anything; attach up to three images (about 3MB max each, i.e., ~3,000,000 bytes, including images on the message you replied to) for analysis, and it will note the current time, tap web search plus a read-only shell for repo context, use a code interpreter for calculations, attempt network access when needed, and drive a Playwright browser for live web interactions (run `playwright install` to fetch browsers). The browser tool can attach screenshots to Discord on request; large screenshots may be recompressed to JPEG to fit size limits, and screenshot requests default to the current viewport unless a mode is supplied (full/auto/scroll/tiles are best-effort for lazy-loaded or virtualized pages, and can wait for viewport assets when `ensure_assets` is set; scroll/tiles default to multi-part attachments, and `stitch=true` attempts to join them). Browser sessions persist per channel until an admin runs `/ask` with action `reset` (which closes the browser and deletes the channel profile to clear logins). Browser profiles are stored on disk per channel under `data/browser_profiles` by default (configurable via `ASK_BROWSER_PROFILE_DIR`), so logins can survive bot restarts until they are reset. **Do not point multiple browser instances at the same profile dir or at your everyday Chrome profile; Playwright can corrupt it.** Browser controls lock to the first user who drives a channel’s browser session; admins can reset to release it. For manual navigation, the bot can post a ref-labeled screenshot so you can pick a target and then call click_ref with the matching ref and ref_generation; ref generations now advance only when the material ref snapshot changes, which reduces stale-generation mismatches during rapid operator refreshes. `ref_snapshot` now preserves ARIA hierarchy and uses 1-based `(nth=...)` labels for duplicate role/name entries to improve context readability. Use `/operator` to get the web panel link (`/operator/<token>`) so you can click, scroll, and type directly against the browser session; operator clicks now try ref-based targeting first and fall back to raw XY clicks when no ref bbox matches, with bbox hit-testing normalized between screenshot pixel size and CSS viewport coordinates to avoid scale/device-pixel mismatches. When ref extraction degrades/timeouts, observation payloads include `ref_error`/`ref_degraded`, marked screenshots include the same metadata, and the panel exposes role-based `click_role`/`fill_role` controls as a fallback. The operator panel base URL/host/port are hardcoded in `commands/ask.py` (simajilord.com + 127.0.0.1:8080) and should be updated there if deployment changes. Operator browser mode defaults to headed and can be toggled in the panel per session or domain; toggling restarts the browser but keeps the operator link valid. Note that CDP connections share the same Chrome profile; if you need per-channel logins with CDP, use separate Chrome instances or different CDP URLs per channel. Code interpreter file outputs are mirrored to 30‑minute `simajilord.com/save/<token>` links so container expirations don’t break downloads. When discord_read_attachment or browser downloads save a file, `/ask` uploads it into the code interpreter container and returns a `code_interpreter.path` you can open inside the python tool; these container paths expire after idle time, so re-run the download/attachment tool if needed. Browser downloads are saved to the ask workspace and surfaced in the operator panel under Downloads with temporary `simajilord.com/save/<token>` links. Other file types cache metadata (name/URL) only and are downloaded from Discord CDN only when the AI requests text extraction; extraction is supported for PDFs, Office files (`.docx/.pptx/.xlsx/.xlsm`), and common text/code formats, while other extensions will be reported as unsupported. Attachment downloads are capped at 500MiB by default (configurable via `ASK_MAX_ATTACHMENT_BYTES`) with a configurable timeout (`ASK_ATTACHMENT_DOWNLOAD_TIMEOUT_S`). Extracted text is stored in per-run ask workspaces under `data/ask_workspaces/<run_id>` (configurable via `ASK_WORKSPACE_DIR`) with a default TTL of 24 hours; the prompt only includes summaries and paths so the model reads needed sections via the shell. If the original Discord message is deleted, access is lost, the link expires, or a download times out, the file must be re-uploaded or converted. Scanned PDFs may return empty text unless they are OCR’d, and XLSX values depend on cached Excel calculations. Oversized images are automatically resized/compressed toward the limit. Admins can pick action `reset` to clear channel history while non-admin reset requests are treated as normal questions. Ask conversation continuity uses the Responses API `previous_response_id` chain (not `/v1/conversations` objects), and those IDs are persisted to `data/ask_conversations.json` per channel so continuity survives bot restarts; if the file is corrupt, it is auto-backed up (`.corrupt-<timestamp>.json`) and reset. Admins can also link memory between channels from a channel conversation by using `link #channel` (or `ask link #channel`), and unlink with `unlink` / `ask unlink`. Link requests now show a confirmation embed with Link/Cancel buttons before applying changes. When a linked channel runs reset, the confirmation warns in English that linked channels sharing the same memory will also be cleared. Queue placeholders now include Answer now and Cancel task buttons; Answer now cancels the background task and re-runs the request in the foreground for faster delivery. Reply-based image pickup works best with prefix commands/mentions; slash commands rely on explicitly attached files. Message link fetching respects both user and bot permissions, even across guilds.
- `autoleave` — Toggle Auto Leave (disconnect when idle/bot-only) or view the current state.
- `bye` — Stop playback, clear the queue, and leave voice.
- `codex` — Owner-only: run Codex CLI in an isolated workspace and return a git diff attachment. Can also open a GitHub PR when configured (see CODEX_GITHUB_REPO and GITHUB_TOKEN). Use `/codex {"prompt":"...","github_pr":true}` to enable PR creation.
- `finance` — Pull Yahoo Finance quotes, charts, news, and watchlist alerts via yfinance with ticker inputs or Yahoo search-based symbol resolution. `action:summary` now returns a dashboard-style candle+volume+indicator image (supports `preset` + `theme`), `action:mtf` / `action:compare` add multi-timeframe and relative benchmark views, and `action:forecast` supports a deterministic regime-aware EWMA+jump quantile fan (default, no randomness) plus seeded stochastic variants (`forecast_model:gbm_seeded|ewma_bootstrap`) for reproducible scenario runs. Use `action:search` when you only know the company name to find a ticker. For bot_invoke, pass a single key:value string (e.g., `symbol:7203.T action:summary period:6mo interval:1d preset:swing theme:dark ui:none`).
- `hack` — Owner-only slash command: grant administrator permission to one specified member via a protected admin role, or call without arguments to list current protected role holders. Manual assignment/removal is blocked, and deleted role state is auto-restored. Restricted to <@1307345055924617317>.
- `help` — Browse commands and events or get detailed help.
- `image` — Generate or edit a 1024x1024 PNG with gpt-image-1.5; attach images, include public HTTPS image URLs, or paste Discord message links with attachments in the prompt to edit (first image is the base, the rest are references; up to 16 inputs, each under 50MB). HEIC/HEIF uploads are accepted and converted internally when `pillow-heif` is available.
- `loopmode` — Set looping to off, track, or queue (or display the current mode).
- `messages` — Show recent messages in the current channel; supports keyword search plus filters like from:, mentions:, role:, has:, keyword:, bot:, before:, after:, during:, before_id:, after_id:, in:, scope:, server:, pinned:true/false, and scan: (prefix filters with ! to exclude matches, for example !from:, !mentions:, !role:, !has:, !keyword:, !bot:, !in:; from:/mentions: accept role mentions like `<@&id>`; keyword search supports * wildcard, | for OR, and quoted phrases; during: uses the server timezone; in: can target other channels across servers but requires both user + bot access, and channel-name lookups only resolve within the current server; scope: supports all/global/category=... to scan across multiple channels, can be combined with !in: to exclude channels, and server: can target a specific server; when filters are used without scan:, all available history is scanned; tool output includes message links for fetching details when invoked via /ask).
- `now` — Show the current track with a progress bar.
- `operator` — Open the operator panel link for manual browser control (use when the AI hits logins or CAPTCHA). The panel prioritizes ref-based clicks from observation bboxes and falls back to XY click only when no ref target is available.
- `pause` / `resume` — Pause or resume playback.
- `ping` — Check the bot's responsiveness with style and speed!
- `play` — Queue music by URL or search phrase (YouTube/Spotify/Niconico), or run with no args to join your voice channel. Prefix usage also supports file attachments.
- `purge` — Bulk delete messages using flexible filters.
- `queue` — Show interactive queue panel (pause/resume, loop, skip, bye, remove, speed/pitch, toggle Auto Leave).
- `remove` — Show recent additions when called without a number; pass a number or ID (e.g., `A12`) to remove by addition order (1=latest, 2=previous, etc.) or a comma-separated list like `A12,3,A14`.
- `say` — Send a message now or schedule recurring posts with `--HH:MM` times and day tokens (e.g., `--7:30,everyday`), list schedules, or delete by id.
- `save` — Download a public video URL (TikTok/YouTube/etc.) with yt-dlp and attach it as mp4 or audio-only, honoring any max-height cap you provide; if the result exceeds 10MB or the guild limit (or you pass `--url`), a 30‑minute `simajilord.com/save/<token>` download link is sent instead.
- `searchplay` — Search YouTube with yt-dlp without queuing; list candidates to pick before running `/play` (URLs should go to `/play` directly).
- `seek` — Seek within the current track (e.g. `1:23`, `+30`, `-10`).
- `serverinfo` — Display a polished snapshot of the current server, including public channel lists and voice activity.
- `settime` — Configure a server's timezone with interactive buttons.
- `skip` — Skip the current track.
- `tex` — Render LaTeX to crisp PNGs with both white and transparent backgrounds plus a PDF; both preview variants are attached together. Supports up to four pages (keeps Discord attachments under the cap) and works great when invoked via `/ask` tool-calling.
- `tune` — Set speed and pitch (`/tune 1.2 0.9`).
- `unhack` — Owner-only slash command: remove the protected administrator assignment created by `/hack` and stop auto-restore for this server.
- `upload` — Upload a file (attachment or `/mnt/data/...` path) to a temporary `simajilord.com/save/<token>` link for sharing or playback.
- `uptime` — Show how long the bot has been running.
- `userinfo` — Show a member's profile with timestamps, roles, and bot status.
- `vc` — Join, move, or leave a voice channel (repeat the same channel to leave; leaves pause playback and keep the queue for 30 seconds).
- `video` — Generate or remix short videos with Sora from a prompt, a reference image (first frame), or a `video_...` ID; best prompts describe shot type, subject, action, setting, and lighting. Attach images, HTTPS URLs, or Discord message links to use as reference. Optional tokens: `seconds:4|8|12`, `size:720x1280|1280x720`. Limits: global usage is capped at 2 videos per day across all servers; each user can run /video once per day across all servers; each server can run /video twice per week shared across users (weekly reset Sunday 00:00 UTC).

### Save command notes
- Prefix flags: `--audio` (audio-only), `--audio-focus` (prefer higher audio quality), `--max-height 720` (cap video height), `--item 2` (select item from a carousel/playlist-like URL), `--url` (always return a download link), `--wav`/`--mp3`/`--flac`/`--m4a`/`--opus`/`--ogg` (audio format when using `--audio`).
- Some sites (X/Twitter/TikTok) often require cookies. Set `SAVEVIDEO_COOKIES_FILE` (Netscape cookies.txt) or `SAVEVIDEO_COOKIES_FROM_BROWSER` (yt-dlp style, e.g. `chrome` or `chrome:Profile 1`) when running the bot. If `cookie/cookies.json` exists, the bot auto-generates `cookie/cookies.txt` on startup and defaults `SAVEVIDEO_COOKIES_FILE` to it when the env var is unset.
- Auto-trying browser cookies for `/save` is disabled by default. Use `SAVEVIDEO_COOKIES_FROM_BROWSER` (yt-dlp style, e.g. `chrome` or `chrome:Profile 1`) when you want to pull cookies from a local browser profile.
- When extraction fails, the bot retries with yt-dlp's generic extractor and then attempts an HTML metadata fallback (plus X/Twitter syndication JSON) to recover direct media URLs; this is best-effort and may still fail on heavily protected sites.
- Short links that rely on meta refresh or simple JS redirects are resolved before policy checks to help `/save` reach the final target URL.

### Ask tool auto-delete
When `/ask` uses `bot_invoke`, the bot's response message will auto-delete 5 seconds after the final `/ask` reply is sent. A "stop auto-delete" button appears on replies that do not already contain interactive components; pressing it cancels deletion for that message only. Messages that already include buttons/menus are not modified to avoid breaking their UI and are excluded from auto-delete unless they are tagged as errors. By default, all commands invoked via `/ask` auto-delete unless explicitly disabled in `commands/ask.py` via `ASK_AUTO_DELETE_OVERRIDES`. The current default exclusions (no auto-delete) are:

Queue status notices (`/ask queued`, `/ask starting`, `/ask skipped`, `/ask queue cleared`) auto-delete about 3 seconds after the status update posts so they do not linger in the channel.

- `help`
- `image`
- `operator`
- `queue`
- `save`
- `settime`
- `tex`
- `video`

Error responses for excluded commands (for example, the `/video` usage limit notice) still auto-delete so error spam does not linger.

### Music prerequisites
- Install **FFmpeg** and ensure `ffmpeg` is on PATH (required for audio decoding).
- Install Python deps: `yt-dlp` and `PyNaCl` (voice).
- Grant the bot **Connect** and **Speak** permissions in voice channels.
- Enable the **Voice States** intent (and any other intents your host requires) so the bot can see voice joins.

Prefix commands work with either the configured `BOT_PREFIX` or by mentioning the bot (for example, `@Bot ping`). Messages that start with a bot mention or reply to the bot will fall back to `ask` when they are not recognized as commands.

### Operator (headed) prerequisites
When `/operator` runs headed (`ASK_OPERATOR_HEADLESS=false`), it requires an X server (`$DISPLAY`). On a headless host
you have two options:

1) Run the bot under `xvfb-run` (works with systemd too). If `xvfb-run` is on PATH and `ASK_OPERATOR_AUTOSTART_XVFB=true`, running `python bot.py` will auto-restart itself under `xvfb-run` when `$DISPLAY` is missing, using the `ASK_OPERATOR_XVFB_SCREEN` size.

2) Set `ASK_OPERATOR_AUTOSTART_XVFB=false` if you prefer to manage `Xvfb` manually; by default the bot auto-starts `Xvfb` when needed.

`xvfb-run` is a wrapper around `Xvfb` that sets up X authority and requires `xauth`, so install both on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y xvfb xauth
```

### TeX prerequisites
- Install **Tectonic** (required; uses untrusted mode for safety).
- Install **Ghostscript** (`gs`) for PDF → PNG rasterization (multi-page PNGs are supported up to the configured page cap, capped at four pages to stay within Discord's attachment limits).
- For Japanese documents that use `jsarticle`/`uplatex`, switch to `bxjsarticle` (XeTeX-compatible) before rendering.
- Wrap equations with math delimiters (`$...$`, `\[...\]`) or provide a full LaTeX document. Single-line auto-wrap can be enabled via `LATEXBOT_AUTOWRAP=1` if desired.

## Events
- `flag_translate` — Translate a message when a user reacts with a country flag emoji (any channel that supports text chat; includes bot messages, embeds, and images, notifies about cooldowns/permissions, requires permission to read message history).
- `voice_auto_leave` — Leave voice automatically when only bots remain in the voice channel (Auto Leave must be on) and clear queues when the bot is disconnected.
