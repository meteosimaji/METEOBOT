from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskman.models import TaskEvent, TaskRecord, TaskResult, TaskSpec
from taskman.runners.base import Runner, RunnerContext
from taskman.store import TaskStore
from taskman.toolgate import ToolGate, ToolGatePolicy


@dataclass(slots=True)
class LaneScheduler:
    lane: str
    limit: int
    queue: asyncio.Queue[str]
    semaphore: asyncio.Semaphore
    locks: dict[str, asyncio.Lock]


class TaskManager:
    def __init__(
        self,
        store: TaskStore,
        *,
        workspace_root: Path,
        lane_limits: dict[str, int] | None = None,
    ) -> None:
        self._store = store
        self._workspace_root = workspace_root
        self._lane_limits = lane_limits or {"main": 2, "subagent": 4, "background": 8}
        self._lanes: dict[str, LaneScheduler] = {}
        self._runners: dict[str, Runner] = {}
        self._worker_tasks: list[asyncio.Task] = []
        self._runtime_context: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._ready = False

    async def start(self) -> None:
        if self._ready:
            return
        await self._store.initialize()
        self._setup_lanes()
        await self.restore_on_boot()
        self._worker_tasks = [
            asyncio.create_task(self._lane_worker(lane)) for lane in self._lanes.values()
        ]
        self._ready = True

    def shutdown(self) -> None:
        for task in self._worker_tasks:
            task.cancel()
        self._worker_tasks.clear()
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def _setup_lanes(self) -> None:
        for lane, limit in self._lane_limits.items():
            self._lanes[lane] = LaneScheduler(
                lane=lane,
                limit=limit,
                queue=asyncio.Queue(),
                semaphore=asyncio.Semaphore(limit),
                locks={},
            )

    def attach_runner(self, kind: str, runner: Runner) -> None:
        self._runners[kind] = runner

    def set_runtime_context(self, task_id: str, payload: dict[str, Any]) -> None:
        self._runtime_context[task_id] = payload

    async def get_task(self, task_id: str) -> TaskRecord | None:
        return await self._store.get_task(task_id)

    def get_runtime_context(self, task_id: str) -> dict[str, Any]:
        payload = self._runtime_context.get(task_id)
        if isinstance(payload, dict):
            return payload
        return {}

    async def submit(self, spec: TaskSpec, *, task_id: str) -> str:
        if not self._ready:
            await self.start()
        now = int(time.time())
        record = TaskRecord(
            task_id=task_id,
            kind=spec.kind,
            lane=spec.lane,
            state_key=spec.state_key,
            status="queued",
            created_at=now,
            updated_at=now,
            heartbeat_at=None,
            output_message_id=spec.output_message_id,
            request=spec.request,
            runner_state={},
            result={},
            parent_task_id=spec.parent_task_id,
            cancel_requested=False,
        )
        await self._store.upsert_task(record)
        await self._enqueue_task(record)
        return record.task_id

    async def _enqueue_task(self, task: TaskRecord) -> None:
        lane = self._lanes.get(task.lane)
        if lane is None:
            raise ValueError(f"Unknown lane: {task.lane}")
        await lane.queue.put(task.task_id)

    async def restore_on_boot(self) -> None:
        pending = await self._store.list_tasks_by_status(
            ["queued", "running", "recovering"]
        )
        now = int(time.time())
        for task in pending:
            if task.status in {"running", "recovering"}:
                await self._store.update_status(
                    task.task_id,
                    status="recovering",
                    updated_at=now,
                    heartbeat_at=now,
                )
            await self._enqueue_task(task)

    async def cancel(self, task_id: str) -> None:
        task = await self._store.get_task(task_id)
        if task is None:
            return
        if task.status in {"cancelled", "succeeded", "failed"}:
            return
        if not isinstance(task.runner_state, dict):
            task.runner_state = {}
        now = int(time.time())
        runtime_context = self._runtime_context.get(task_id, {})
        runtime_response_id = (
            runtime_context.get("openai_response_id") if isinstance(runtime_context, dict) else None
        )
        if (
            isinstance(runtime_response_id, str)
            and runtime_response_id
            and task.runner_state.get("openai_response_id") != runtime_response_id
        ):
            task.runner_state["openai_response_id"] = runtime_response_id
            await self._store.update_runner_state(task_id, task.runner_state, now)
        cancel_result = {"cancelled": True}
        if task.status in {"queued", "recovering"}:
            await self._store.update_result(
                task_id, result=cancel_result, status="cancelled", updated_at=now
            )
            await self._store.update_status(
                task_id, status="cancelled", updated_at=now, cancel_requested=True
            )
        else:
            await self._store.update_status(
                task_id, status=task.status, updated_at=now, cancel_requested=True
            )
        cancel_event = self._cancel_events.get(task_id)
        if cancel_event is None:
            cancel_event = asyncio.Event()
            self._cancel_events[task_id] = cancel_event
        cancel_event.set()
        runner = self._runners.get(task.kind)
        if runner is None:
            return
        ctx = await self._build_runner_context(task)
        try:
            result = await runner.cancel(ctx)
        except Exception:
            result = None
        if result is None:
            return
        updated_at = int(time.time())
        await self._store.update_runner_state(task.task_id, result.runner_state, updated_at)
        await self._store.update_result(
            task.task_id,
            result=result.result or cancel_result,
            status=result.status,
            updated_at=updated_at,
        )

    async def queued_position(self, *, state_key: str, lane: str) -> int:
        queued = await self._store.list_tasks_by_state_key(
            state_key=state_key, lane=lane, status="queued"
        )
        return len(queued)

    async def cancel_state_key(self, state_key: str) -> int:
        cancelled = 0
        for status in ("queued", "running", "recovering"):
            tasks = await self._store.list_tasks_by_state_key(
                state_key=state_key, lane=None, status=status
            )
            for task in tasks:
                await self.cancel(task.task_id)
                cancelled += 1
        return cancelled

    async def _lane_worker(self, lane: LaneScheduler) -> None:
        while True:
            task_id = await lane.queue.get()
            await lane.semaphore.acquire()
            asyncio.create_task(self._run_task(task_id, lane))

    async def _run_task(self, task_id: str, lane: LaneScheduler) -> None:
        lock = None
        try:
            task = await self._store.get_task(task_id)
            if task is None:
                return
            if task.cancel_requested or task.status == "cancelled":
                if not task.result:
                    await self._store.update_result(
                        task.task_id,
                        result={"cancelled": True},
                        status="cancelled",
                        updated_at=int(time.time()),
                    )
                return
            lock = lane.locks.get(task.state_key)
            if lock is None:
                lock = asyncio.Lock()
                lane.locks[task.state_key] = lock
            async with lock:
                await self._execute_task(task)
        finally:
            lane.semaphore.release()
            self._runtime_context.pop(task_id, None)
            self._cancel_events.pop(task_id, None)

    async def _execute_task(self, task: TaskRecord) -> None:
        runner = self._runners.get(task.kind)
        if runner is None:
            await self._store.update_result(
                task.task_id,
                result={"error": "No runner registered."},
                status="failed",
                updated_at=int(time.time()),
            )
            return
        now = int(time.time())
        await self._store.update_status(
            task.task_id,
            status="running",
            updated_at=now,
            heartbeat_at=now,
        )
        cancel_event = self._cancel_events.get(task.task_id)
        if cancel_event and cancel_event.is_set():
            await self._store.update_result(
                task.task_id,
                result={"cancelled": True},
                status="cancelled",
                updated_at=int(time.time()),
            )
            return
        ctx = await self._build_runner_context(task)
        result: TaskResult
        try:
            result = await runner.run(ctx)
        except Exception as exc:
            result = TaskResult(status="failed", result={"error": repr(exc)}, error=repr(exc))
        cancel_event = self._cancel_events.get(task.task_id)
        if (cancel_event and cancel_event.is_set()) or task.cancel_requested or result.status == "cancelled":
            result.status = "cancelled"
            if not isinstance(result.result, dict):
                result.result = {}
            result.result.setdefault("cancelled", True)
        updated_at = int(time.time())
        await self._store.update_runner_state(task.task_id, result.runner_state, updated_at)
        await self._store.update_result(
            task.task_id,
            result=result.result,
            status=result.status,
            updated_at=updated_at,
        )

    async def _build_runner_context(self, task: TaskRecord) -> RunnerContext:
        workspace_dir = self._workspace_root / task.task_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        cancel_event = self._cancel_events.get(task.task_id)
        if cancel_event is None:
            cancel_event = asyncio.Event()
            self._cancel_events[task.task_id] = cancel_event

        async def _emit_event(event_type: str, payload: dict[str, Any]) -> None:
            await self._store.append_event(
                TaskEvent(
                    task_id=task.task_id,
                    ts=int(time.time()),
                    type=event_type,
                    payload=payload,
                )
            )

        async def _heartbeat() -> None:
            now = int(time.time())
            await self._store.update_status(
                task.task_id,
                status="running",
                updated_at=now,
                heartbeat_at=now,
                cancel_requested=None,
            )

        async def _update_runner_state(update: dict[str, Any]) -> None:
            if not update:
                return
            existing = await self._store.get_task(task.task_id)
            merged_state: dict[str, Any] = {}
            if existing and isinstance(existing.runner_state, dict):
                merged_state.update(existing.runner_state)
            merged_state.update(update)
            runtime_context = self._runtime_context.get(task.task_id)
            if isinstance(runtime_context, dict):
                runtime_context.update(update)
            await self._store.update_runner_state(task.task_id, merged_state, int(time.time()))

        runtime_context = self._runtime_context.get(task.task_id, {})
        policy = runtime_context.get("tool_policy")
        if not isinstance(policy, ToolGatePolicy):
            policy = self._policy_from_request(task)
        toolgate = ToolGate(
            policy=policy,
            cancel_event=cancel_event,
            emit_event=_emit_event,
        )
        return RunnerContext(
            task=task,
            workspace_dir=workspace_dir,
            toolgate=toolgate,
            emit_event=_emit_event,
            heartbeat=_heartbeat,
            update_runner_state=_update_runner_state,
            runtime_context=runtime_context,
        )

    @staticmethod
    def _policy_from_request(task: TaskRecord) -> ToolGatePolicy:
        raw_policy = task.request.get("tool_policy") if isinstance(task.request, dict) else None
        if not isinstance(raw_policy, dict):
            return ToolGatePolicy()
        allowed = raw_policy.get("allowed_tools")
        denied = raw_policy.get("denied_tools")
        permission_mode = raw_policy.get("permission_mode") or "execute"
        allowed_set = {str(item) for item in allowed} if isinstance(allowed, list) else set()
        denied_set = {str(item) for item in denied} if isinstance(denied, list) else set()
        return ToolGatePolicy(
            allowed_tools=allowed_set,
            denied_tools=denied_set,
            permission_mode=str(permission_mode),
        )
