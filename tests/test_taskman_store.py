from __future__ import annotations

import asyncio
import time
from pathlib import Path

from taskman.models import TaskRecord
from taskman.store import TaskStore


def _make_task(task_id: str) -> TaskRecord:
    now = int(time.time())
    return TaskRecord(
        task_id=task_id,
        kind="ask",
        lane="main",
        state_key="1:2",
        status="queued",
        created_at=now,
        updated_at=now,
        heartbeat_at=None,
        output_message_id="123",
        request={"text": "hi"},
        runner_state={},
        result={},
    )


def test_task_store_roundtrip(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "taskman.sqlite")
    asyncio.run(store.initialize())

    task = _make_task("task-1")
    asyncio.run(store.upsert_task(task))
    loaded = asyncio.run(store.get_task("task-1"))

    assert loaded is not None
    assert loaded.task_id == "task-1"
    assert loaded.request == {"text": "hi"}


def test_task_store_list_by_status(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "taskman.sqlite")
    asyncio.run(store.initialize())
    task1 = _make_task("task-1")
    task2 = _make_task("task-2")
    task2.status = "running"
    asyncio.run(store.upsert_task(task1))
    asyncio.run(store.upsert_task(task2))

    queued = asyncio.run(store.list_tasks_by_status(["queued"]))
    running = asyncio.run(store.list_tasks_by_status(["running"]))

    assert [task.task_id for task in queued] == ["task-1"]
    assert [task.task_id for task in running] == ["task-2"]


def test_task_store_list_by_state_key_without_initialize(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "taskman.sqlite")

    tasks = asyncio.run(store.list_tasks_by_state_key(state_key="1:2"))

    assert tasks == []


def test_task_store_reinitializes_after_db_removed(tmp_path: Path) -> None:
    db_path = tmp_path / "taskman.sqlite"
    store = TaskStore(db_path)
    asyncio.run(store.initialize())
    assert db_path.exists()
    db_path.unlink()

    tasks = asyncio.run(store.list_tasks_by_state_key(state_key="1:2"))

    assert tasks == []
    assert db_path.exists()
