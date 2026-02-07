from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "recovering",
]
TaskLane = Literal["main", "subagent", "background"]
TaskKind = Literal["ask", "subagent", "poll", "tool", "maintenance"]


@dataclass(slots=True)
class TaskSpec:
    kind: TaskKind
    lane: TaskLane
    state_key: str
    request: dict[str, Any]
    parent_task_id: str | None = None
    output_message_id: str | None = None


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    kind: TaskKind
    lane: TaskLane
    state_key: str
    status: TaskStatus
    created_at: int
    updated_at: int
    heartbeat_at: int | None
    output_message_id: str | None
    request: dict[str, Any] = field(default_factory=dict)
    runner_state: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    parent_task_id: str | None = None
    cancel_requested: bool = False


@dataclass(slots=True)
class TaskResult:
    status: TaskStatus
    result: dict[str, Any] = field(default_factory=dict)
    runner_state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class TaskEvent:
    task_id: str
    ts: int
    type: str
    payload: dict[str, Any]
