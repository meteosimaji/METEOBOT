---
name: ask-recipes
description: "Recipes for combining /ask tools and bot commands (music search/play, user profile lookup, message context) when tool sequencing is unclear."
---

# Ask tool recipes

Use this document when tool sequencing is unclear or when chaining two or more tools/commands. Prefer quick keyword search first:

# How to search
- List recipes: `rg -n -m 50 "^## " docs/skills/ask-recipes/SKILL.md`
- Find a recipe by ID marker: `rg -n -m 1 "@-- BEGIN:id:music --" docs/skills/ask-recipes/SKILL.md -A 120`
- Keyword search (fallback): `rg -n -m 200 "music" docs/skills/ask-recipes/SKILL.md`

# Recipe titles

- Title: Music playback (search → play)
  Goal: Play a song from search terms by listing candidates and then running /play with a chosen URL.
- Title: User icon/profile
  Goal: Show a user's icon/profile details via /userinfo.
- Title: Context scan
  Goal: Fetch recent messages for context via /messages with filters.
- Title: Preflight attachment/link pickup
  Goal: Always fetch message context and attachment tokens before other tool calls.
- Title: Message link context
  Goal: Quote and summarize a linked message before answering.
- Title: Attachment read/extract
  Goal: Read a file attachment and extract text safely.
- Title: Math rendering decision
  Goal: Decide when to use /tex and apply a consistent format.
- Title: Command lookup before suggestion
  Goal: Use bot_commands before suggesting any bot command.
- Title: Queue remove (2-step)
  Goal: Fetch a queue list before removing a specific item.

# Index

-- (id:attachments) Attachment read/extract: `attachments`, `read`, `extract`, `pdf`, `docx`, `xlsx`
-- (id:cmdlookup) Command lookup before suggestion: `bot_commands`, `help`, `commands`
-- (id:linkctx) Message link context: `link`, `message url`, `context`
-- (id:messages) Context scan: `messages`, `context`, `filters`
-- (id:music) Music playback (search → play): `searchplay`, `play`, `duration`
-- (id:preflight) Preflight attachment/link pickup: `preflight`, `attachments`, `links`
-- (id:remove) Queue remove (2-step): `remove`, `queue`
-- (id:tex) Math rendering decision: `tex`, `math`, `equation`
-- (id:userinfo) User icon/profile: `userinfo`, `avatar`, `profile`

@-- BEGIN:id:music --
## (id:music) Music playback (search → play)

Goal: play a song when the user provides a title, artist, or vibe.

**Input cues**
- User gives search terms (title/artist/series/vibe), no URL.

**Steps**
1. Call `/searchplay` with the search terms to list candidates with durations.
2. Pick the best match (or ask the user to choose), then call `/play` with the URL.

**Notes**
- `/searchplay` is for search terms only. If the user provides a URL, skip to `/play`.
- Use the durations from `/searchplay` to avoid long loop/extended versions.

**Tool calls**
- `bot_invoke({"name":"searchplay","arg":"<search terms>"})`
- `bot_invoke({"name":"play","arg":"<selected url>"})`
@-- END:id:music --

@-- BEGIN:id:userinfo --
## (id:userinfo) User icon/profile

Goal: show a user's icon/profile details.

**Input cues**
- User asks for their icon, profile, or a member's details.

**Steps**
1. Call `/userinfo` with a mention or ID if provided; otherwise leave arg empty.
2. Use the embed's avatar/profile details in your response.

**Tool calls**
- `bot_invoke({"name":"userinfo","arg":"@name"})`
- `bot_invoke({"name":"userinfo","arg":""})` (defaults to the requester)
@-- END:id:userinfo --

@-- BEGIN:id:messages --
## (id:messages) Context scan

Goal: understand recent conversation context in the channel.

**Input cues**
- User asks for recent context, previous messages, or summary cues.

**Steps**
1. Use `/messages` with optional filters to narrow scope.
2. If no filters are given, default is 50 messages; narrow with time/channel filters if needed.

**Tool calls**
- `bot_invoke({"name":"messages","arg":"<optional filters/keywords>"})`

**Examples**
- `bot_invoke({"name":"messages","arg":""})`
- `bot_invoke({"name":"messages","arg":"10 from:@user"})`
- `bot_invoke({"name":"messages","arg":"keyword:launch before:2026-01-01 scan:500"})`
@-- END:id:messages --

@-- BEGIN:id:preflight --
## (id:preflight) Preflight attachment/link pickup

Goal: always fetch request context and attachment tokens before using other tools.

**Input cues**
- The user references files, images, or message links.
- Any multi-step tool chain where attachments might be needed.

**Steps**
1. Call `discord_fetch_message` with url:'' to capture current request context.
2. Call `discord_list_attachments` to list cached attachment tokens for this ask.
3. If a message link is present, call `discord_fetch_message` with that link too.
4. Proceed with the next recipe/tool once attachments/links are known.

**Tool calls**
- `discord_fetch_message({"url":""})`
- `discord_list_attachments({"message_id":"","channel_id":""})`
- `discord_fetch_message({"url":"<message link>"})`
@-- END:id:preflight --

@-- BEGIN:id:linkctx --
## (id:linkctx) Message link context

Goal: quote and summarize a linked message before answering the user’s request.

**Input cues**
- The user pasted a message URL or asked about a specific message.

**Steps**
1. Call `discord_fetch_message` with the message URL.
2. Summarize author/time/content/embeds/attachments as quoted context.
3. Answer the user’s request using that context.

**Tool calls**
- `discord_fetch_message({"url":"<message link>"})`
@-- END:id:linkctx --

@-- BEGIN:id:attachments --
## (id:attachments) Attachment read/extract

Goal: read attachment content safely and extract text as needed.

**Input cues**
- The user asks to read a PDF, doc, slide, sheet, or text file.

**Steps**
1. Call `discord_list_attachments` to list available tokens.
2. Call `discord_read_attachment` with the chosen token (use `max_chars` if needed).
3. If the result is empty or garbled, explain scan/ToUnicode issues and request a text-based file or images for OCR.

**Tool calls**
- `discord_list_attachments({"message_id":"","channel_id":""})`
- `discord_read_attachment({"token":"<token>","max_chars":6000})`
@-- END:id:attachments --

@-- BEGIN:id:tex --
## (id:tex) Math rendering decision

Goal: choose between plain text and /tex, and format equations consistently.

**Input cues**
- Multi-line equations, alignment needs, or formatting-sensitive math.

**Steps**
1. Use plain text for short, simple expressions.
2. Use `/tex` for multi-line or alignment-sensitive equations.
3. For multi-line: use `\\[\\begin{aligned} ... \\end{aligned}\\]` and align with `&`.

**Tool calls**
- `bot_invoke({"name":"tex","arg":"\\[E=mc^2\\]"})`
- `bot_invoke({"name":"tex","arg":"\\[\\begin{aligned}a&=b+c\\\\d&=e+f\\end{aligned}\\]"})`
@-- END:id:tex --

@-- BEGIN:id:cmdlookup --
## (id:cmdlookup) Command lookup before suggestion

Goal: avoid suggesting non-existent commands by checking bot_commands first.

**Input cues**
- The user asks “can you do X?” or you are about to suggest a bot command.

**Steps**
1. Call `bot_commands` to list available commands.
2. Suggest or invoke only commands from the list.

**Tool calls**
- `bot_commands({})`
@-- END:id:cmdlookup --

@-- BEGIN:id:remove --
## (id:remove) Queue remove (2-step)

Goal: remove a specific queue entry using the required two-step flow.

**Input cues**
- The user asks to remove a song from the queue.

**Steps**
1. Call `/remove` with an empty arg to get the queue list with IDs.
2. Call `/remove` again with the chosen ID/number.

**Tool calls**
- `bot_invoke({"name":"remove","arg":""})`
- `bot_invoke({"name":"remove","arg":"<id>"})`
@-- END:id:remove --
