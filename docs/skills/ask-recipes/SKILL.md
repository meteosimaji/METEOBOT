---
name: ask-recipes
description: "Recipes for combining /ask tools and bot commands (music search/play, user profile lookup, message context) when tool sequencing is unclear."
---

# Ask tool recipes

Use this document only when tool sequencing is unclear. Prefer quick keyword search first:

- `rg -n -m 200 "music" docs/skills/ask-recipes/SKILL.md`
- `rg -n -m 200 "userinfo" docs/skills/ask-recipes/SKILL.md`
- `rg -n -m 200 "messages" docs/skills/ask-recipes/SKILL.md`

## Recipe titles

- Title: Music playback (search → play)
  Goal: Play a song from search terms by listing candidates and then running /play with a chosen URL.
- Title: User icon/profile
  Goal: Show a user's icon/profile details via /userinfo.
- Title: Context scan
  Goal: Fetch recent messages for context via /messages with filters.

## Index

- Music playback (search → play): `#music`, `searchplay`, `play`, `duration`
- User icon/profile: `#userinfo`, `avatar`, `profile`
- Context scan: `#messages`, `context`, `filters`

## Music playback recipe (#music)

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

## User icon/profile recipe (#userinfo)

Goal: show a user's icon/profile details.

**Input cues**
- User asks for their icon, profile, or a member's details.

**Steps**
1. Call `/userinfo` with a mention or ID if provided; otherwise leave arg empty.
2. Use the embed's avatar/profile details in your response.

**Tool calls**
- `bot_invoke({"name":"userinfo","arg":"@name"})`
- `bot_invoke({"name":"userinfo","arg":""})` (defaults to the requester)

## Context scan recipe (#messages)

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
