from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from taskman.models import TaskEvent, TaskRecord, TaskStatus


class TaskStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._init_lock:
            await asyncio.to_thread(self._init_db)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    heartbeat_at INTEGER,
                    output_message_id TEXT,
                    parent_task_id TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    request_json TEXT NOT NULL,
                    runner_state_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_lane_status ON tasks(lane, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_state_key ON tasks(state_key)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    mime TEXT,
                    description TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id)"
            )
            conn.commit()

    async def upsert_task(self, task: TaskRecord) -> None:
        await asyncio.to_thread(self._upsert_task, task)

    def _upsert_task(self, task: TaskRecord) -> None:
        payload = self._serialize_task(task)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, kind, lane, state_key, status, created_at, updated_at,
                    heartbeat_at, output_message_id, parent_task_id, cancel_requested,
                    request_json, runner_state_json, result_json
                ) VALUES (
                    :task_id, :kind, :lane, :state_key, :status, :created_at, :updated_at,
                    :heartbeat_at, :output_message_id, :parent_task_id, :cancel_requested,
                    :request_json, :runner_state_json, :result_json
                )
                ON CONFLICT(task_id) DO UPDATE SET
                    kind=excluded.kind,
                    lane=excluded.lane,
                    state_key=excluded.state_key,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    heartbeat_at=excluded.heartbeat_at,
                    output_message_id=excluded.output_message_id,
                    parent_task_id=excluded.parent_task_id,
                    cancel_requested=excluded.cancel_requested,
                    request_json=excluded.request_json,
                    runner_state_json=excluded.runner_state_json,
                    result_json=excluded.result_json
                """,
                payload,
            )
            conn.commit()

    async def get_task(self, task_id: str) -> TaskRecord | None:
        return await asyncio.to_thread(self._get_task, task_id)

    def _get_task(self, task_id: str) -> TaskRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    async def list_tasks_by_status(
        self, statuses: Iterable[TaskStatus], *, lane: str | None = None
    ) -> list[TaskRecord]:
        return await asyncio.to_thread(self._list_tasks_by_status, list(statuses), lane)

    def _list_tasks_by_status(
        self, statuses: list[TaskStatus], lane: str | None
    ) -> list[TaskRecord]:
        if not statuses:
            return []
        placeholders = ",".join(["?"] * len(statuses))
        params: list[Any] = list(statuses)
        clause = f"status IN ({placeholders})"
        if lane:
            clause += " AND lane = ?"
            params.append(lane)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE {clause} ORDER BY created_at ASC",
                params,
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    async def list_tasks_by_state_key(
        self, *, state_key: str, lane: str | None = None, status: TaskStatus | None = None
    ) -> list[TaskRecord]:
        return await asyncio.to_thread(
            self._list_tasks_by_state_key, state_key, lane, status
        )

    def _list_tasks_by_state_key(
        self, state_key: str, lane: str | None, status: TaskStatus | None
    ) -> list[TaskRecord]:
        params: list[Any] = [state_key]
        clause = "state_key = ?"
        if lane:
            clause += " AND lane = ?"
            params.append(lane)
        if status:
            clause += " AND status = ?"
            params.append(status)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE {clause} ORDER BY created_at ASC",
                params,
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    async def update_status(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        updated_at: int,
        heartbeat_at: int | None = None,
        cancel_requested: bool | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._update_status, task_id, status, updated_at, heartbeat_at, cancel_requested
        )

    def _update_status(
        self,
        task_id: str,
        status: TaskStatus,
        updated_at: int,
        heartbeat_at: int | None,
        cancel_requested: bool | None,
    ) -> None:
        fields = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, updated_at]
        if heartbeat_at is not None:
            fields.append("heartbeat_at = ?")
            params.append(heartbeat_at)
        if cancel_requested is not None:
            fields.append("cancel_requested = ?")
            params.append(1 if cancel_requested else 0)
        params.append(task_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ?",
                params,
            )
            conn.commit()

    async def update_runner_state(self, task_id: str, runner_state: dict[str, Any], updated_at: int) -> None:
        await asyncio.to_thread(self._update_runner_state, task_id, runner_state, updated_at)

    def _update_runner_state(
        self, task_id: str, runner_state: dict[str, Any], updated_at: int
    ) -> None:
        payload = json.dumps(runner_state, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET runner_state_json = ?, updated_at = ? WHERE task_id = ?",
                (payload, updated_at, task_id),
            )
            conn.commit()

    async def update_result(
        self,
        task_id: str,
        *,
        result: dict[str, Any],
        status: TaskStatus,
        updated_at: int,
    ) -> None:
        await asyncio.to_thread(self._update_result, task_id, result, status, updated_at)

    def _update_result(
        self, task_id: str, result: dict[str, Any], status: TaskStatus, updated_at: int
    ) -> None:
        payload = json.dumps(result, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET result_json = ?, status = ?, updated_at = ? WHERE task_id = ?",
                (payload, status, updated_at, task_id),
            )
            conn.commit()

    async def append_event(self, event: TaskEvent) -> None:
        await asyncio.to_thread(self._append_event, event)

    def _append_event(self, event: TaskEvent) -> None:
        payload = json.dumps(event.payload, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO task_events (ts, task_id, type, payload_json) VALUES (?, ?, ?, ?)",
                (event.ts, event.task_id, event.type, payload),
            )
            conn.commit()

    def _serialize_task(self, task: TaskRecord) -> dict[str, Any]:
        payload = asdict(task)
        payload["request_json"] = json.dumps(task.request, ensure_ascii=False)
        payload["runner_state_json"] = json.dumps(task.runner_state, ensure_ascii=False)
        payload["result_json"] = json.dumps(task.result, ensure_ascii=False)
        payload["cancel_requested"] = 1 if task.cancel_requested else 0
        payload.pop("request")
        payload.pop("runner_state")
        payload.pop("result")
        return payload

    def _row_to_task(self, row: sqlite3.Row) -> TaskRecord:
        request = json.loads(row["request_json"])
        runner_state = json.loads(row["runner_state_json"])
        result = json.loads(row["result_json"])
        return TaskRecord(
            task_id=row["task_id"],
            kind=row["kind"],
            lane=row["lane"],
            state_key=row["state_key"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            heartbeat_at=row["heartbeat_at"],
            output_message_id=row["output_message_id"],
            parent_task_id=row["parent_task_id"],
            cancel_requested=bool(row["cancel_requested"]),
            request=request if isinstance(request, dict) else {},
            runner_state=runner_state if isinstance(runner_state, dict) else {},
            result=result if isinstance(result, dict) else {},
        )
