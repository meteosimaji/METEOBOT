# EventInfo Field Reference

`EventInfo` provides metadata for each automatic event so the `/help` command can display consistent summaries.

## Fields

| Field | Description |
| --- | --- |
| `name` | Short identifier shown in help and the README. |
| `destination` | One-line summary of what the event does. |
| `plus` | Optional extra guidance for the help view. |
| `pro` | Optional advanced notes for power users. |
| `category` | Help category; defaults to "Utility". |
| `example` | Optional usage example shown in help. |

## Registering an event

Each event module defines an `EVENT_INFO` instance and appends it to `bot.events` in `setup()`:

```python
from discord.ext import commands
from events import EventInfo

EVENT_INFO = EventInfo(
    name="quote",
    destination="reply with just a mention to create a quote image",
    category="Utility",
)

async def setup(bot: commands.Bot) -> None:
    if not hasattr(bot, "events"):
        bot.events = []
    bot.events.append(EVENT_INFO)
    await bot.add_cog(Quote(bot))
```

Ensure `setup()` both appends `EVENT_INFO` and loads the cog so `/help` and other tooling can discover the event.
