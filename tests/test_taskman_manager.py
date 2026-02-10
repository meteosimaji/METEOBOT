from __future__ import annotations

import asyncio
import time
from pathlib import Path

from taskman.manager import TaskManager
from taskman.models import TaskRecord
from taskman.store import TaskStore


def _make_task(task_id: str, *, state_key: str, status: str) -> TaskRecord:
    now = int(time.time())
    return TaskRecord(
        task_id=task_id,
        kind="ask",
        lane="main",
        state_key=state_key,
        status=status,
        created_at=now,
        updated_at=now,
        heartbeat_at=None,
        output_message_id="100",
        request={"channel_id": 123},
        runner_state={},
        result={},
    )


def test_queue_snapshot_counts_running_and_recovering(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "taskman.sqlite")
    asyncio.run(store.initialize())
    asyncio.run(store.upsert_task(_make_task("queued-1", state_key="1:1", status="queued")))
    asyncio.run(store.upsert_task(_make_task("running-1", state_key="1:1", status="running")))
    asyncio.run(store.upsert_task(_make_task("recovering-1", state_key="1:1", status="recovering")))
    asyncio.run(store.upsert_task(_make_task("queued-other", state_key="9:9", status="queued")))

    manager = TaskManager(store, workspace_root=tmp_path / "work")
    queued, active_count = asyncio.run(manager.queue_snapshot(state_key="1:1", lane="main"))

    assert [task.task_id for task in queued] == ["queued-1"]
    assert active_count == 2
