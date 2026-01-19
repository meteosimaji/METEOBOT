# Repository Guidelines

## Structure
- Slash commands live under `commands/` as individual extensions with an async `setup(bot)` function that adds a cog.
- Regular cogs go under `cogs/` and also expose `setup(bot)`.
- Event listeners are under `events/`. Each event module defines an `EVENT_INFO` dataclass instance and appends it to `bot.events` in `setup()`.

## Contributing
- Keep command and event names sorted alphabetically in help outputs. Use the
  helper functions in `commands/help.py` and include the extras fields shown in
  that module (`category`, `pro`).
- Provide clear `destination`, `plus` and `pro` descriptions for every command
  and event so the `/help` output is easy to understand.
- Whenever a command or event is added or changed, update the README's lists of
  commands and events (sorted alphabetically).
- `/ask` tool-call auto-delete behavior is configured in `commands/ask.py` via
  `ASK_AUTO_DELETE_OVERRIDES`; if you change the defaults, update the README's
  "Ask tool auto-delete" section to match.

- For any code change run syntax checks with:
  ```bash
  python -m py_compile $(git ls-files '*.py')
  ```
  Fix issues before committing.
- Add any new dependencies to `requirements.txt`.
