from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from taskman.models import TaskRecord, TaskResult
from taskman.toolgate import ToolGate


@dataclass(slots=True)
class RunnerContext:
    task: TaskRecord
    workspace_dir: Path
    toolgate: ToolGate
    emit_event: Callable[[str, dict[str, Any]], Awaitable[None] | None]
    heartbeat: Callable[[], Awaitable[None]]
    update_runner_state: Callable[[dict[str, Any]], Awaitable[None]]
    runtime_context: dict[str, Any] = field(default_factory=dict)


class Runner(Protocol):
    async def run(self, ctx: RunnerContext) -> TaskResult:
        ...

    async def cancel(self, ctx: RunnerContext) -> TaskResult | None:
        ...
