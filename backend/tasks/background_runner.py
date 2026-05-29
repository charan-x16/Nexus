import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from langgraph.types import Command
from rich.console import Console

from backend.db.connection import execute_query, fetch_rows
from backend.graphs.research_graph import get_compiled_graph
from backend.schemas.workflow import (
    AgentMessage,
    WorkflowState,
    serialize_workflow_state,
)
from middleware.request_queue import QueueDecision, workflow_execution_queue


console = Console()


@dataclass(frozen=True)
class WorkflowJob:
    run_id: str
    goal: str
    project_id: str


class WorkflowRunner:
    def __init__(self) -> None:
        self.run_queue: asyncio.Queue[WorkflowJob] = asyncio.Queue()
        self.active_runs: dict[str, asyncio.Task[None]] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._stopped.clear()
            self._worker_task = asyncio.create_task(self.worker(), name="workflow-runner")
            console.log("[workflow_runner] started")

    async def stop(self) -> None:
        self._stopped.set()
        if self._worker_task is not None:
            self._worker_task.cancel()
            await _suppress_cancelled(self._worker_task)
        for task in list(self.active_runs.values()):
            task.cancel()
        await asyncio.gather(*self.active_runs.values(), return_exceptions=True)
        self.active_runs.clear()
        console.log("[workflow_runner] stopped")

    async def submit(self, run_id: str, goal: str, project_id: str) -> QueueDecision:
        decision = await workflow_execution_queue.reserve(run_id)
        await self.run_queue.put(
            WorkflowJob(run_id=run_id, goal=goal, project_id=project_id)
        )
        console.log(
            f"[workflow_runner] queued run {run_id} "
            f"status={decision.status} position={decision.position}"
        )
        return decision

    async def worker(self) -> None:
        while not self._stopped.is_set():
            job = await self.run_queue.get()
            task = asyncio.create_task(
                self._run_with_tracking(job),
                name=f"workflow-run-{job.run_id}",
            )
            self.active_runs[job.run_id] = task
            task.add_done_callback(
                lambda completed, run_id=job.run_id: self._finish_task(run_id, completed)
            )

    async def get_status(self, run_id: str) -> str:
        task = self.active_runs.get(run_id)
        if task is not None and not task.done():
            rows = await fetch_rows(
                """
                SELECT status
                FROM workflow_runs
                WHERE id = $1
                """,
                UUID(str(run_id)),
            )
            if rows:
                return str(rows[0]["status"])
            return "running"
        rows = await fetch_rows(
            """
            SELECT status
            FROM workflow_runs
            WHERE id = $1
            """,
            UUID(str(run_id)),
        )
        return str(rows[0]["status"]) if rows else "unknown"

    async def cancel(self, run_id: str) -> None:
        task = self.active_runs.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            await _suppress_cancelled(task)
        await workflow_execution_queue.release(run_id)
        state = await _load_state_for_runner(run_id)
        state["awaiting_approval"] = False
        state["status"] = "cancelled"
        state["messages"] = list(state.get("messages", [])) + [
            AgentMessage(
                agent="system",
                role="status",
                content="Workflow was cancelled by the user.",
                timestamp=datetime.now(timezone.utc),
            )
        ]
        await _update_status_with_retry(run_id, "cancelled", state)

    async def _run_with_tracking(self, job: WorkflowJob) -> None:
        try:
            await workflow_execution_queue.wait_for_slot(job.run_id)
            state = await _load_state_for_runner(job.run_id)
            if state.get("status") == "cancelled":
                return
            state["status"] = "researching"
            state["awaiting_approval"] = False
            await _update_status_with_retry(job.run_id, "researching", state)

            config = {"configurable": {"thread_id": job.run_id}}
            graph = get_compiled_graph()
            graph_result = await graph.ainvoke(
                Command(resume={"approved": True}),
                config=config,
            )
            final_state = await _graph_state_or_result(graph, config, graph_result)
            final_state["awaiting_approval"] = False
            final_state["status"] = "completed"
            await _update_status_with_retry(job.run_id, "completed", final_state)
            await _persist_agent_messages(job.run_id, final_state.get("messages", []))
            console.log(f"[workflow_runner] completed run {job.run_id}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failed_state = await _load_state_for_runner(job.run_id)
            failed_state["awaiting_approval"] = False
            failed_state["status"] = "failed"
            failed_state["final_output"] = f"Workflow failed: {exc}"
            failed_state["messages"] = list(failed_state.get("messages", [])) + [
                AgentMessage(
                    agent="system",
                    role="error",
                    content=str(exc),
                    timestamp=datetime.now(timezone.utc),
                )
            ]
            await _update_status_with_retry(job.run_id, "failed", failed_state)
            console.log(f"[workflow_runner] failed run {job.run_id}: {exc}")
        finally:
            await workflow_execution_queue.release(job.run_id)

    def _finish_task(self, run_id: str, task: asyncio.Task[None]) -> None:
        self.active_runs.pop(run_id, None)
        self.run_queue.task_done()
        asyncio.create_task(workflow_execution_queue.release(run_id))
        if task.cancelled():
            console.log(f"[workflow_runner] cancelled run {run_id}")
        elif task.exception() is not None:
            console.log(f"[workflow_runner] task error for {run_id}: {task.exception()}")


workflow_runner = WorkflowRunner()


async def _graph_state_or_result(
    graph: Any,
    config: dict[str, Any],
    graph_result: Any,
) -> WorkflowState:
    try:
        snapshot = await graph.aget_state(config)
        values = getattr(snapshot, "values", None)
        if isinstance(values, dict):
            return dict(values)
    except Exception:
        pass
    return dict(graph_result) if isinstance(graph_result, dict) else {}


async def _load_state_for_runner(run_id: str) -> WorkflowState:
    rows = await fetch_rows(
        """
        SELECT state
        FROM workflow_runs
        WHERE id = $1
        """,
        UUID(str(run_id)),
    )
    if not rows:
        raise ValueError(f"Workflow run {run_id} was not found.")
    value = rows[0]["state"]
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    return {}


async def _update_status_with_retry(
    run_id: str,
    status_value: str,
    state: WorkflowState,
) -> None:
    state["status"] = status_value
    for attempt in range(3):
        try:
            await execute_query(
                """
                UPDATE workflow_runs
                SET status = $2,
                    state = $3::jsonb,
                    updated_at = NOW()
                WHERE id = $1
                """,
                UUID(str(run_id)),
                status_value,
                json.dumps(serialize_workflow_state(state)),
            )
            return
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(0.5 * (2**attempt))


async def _persist_agent_messages(
    run_id: str,
    messages: list[AgentMessage] | list[dict[str, Any]],
) -> None:
    for raw_message in messages:
        message = (
            raw_message
            if isinstance(raw_message, AgentMessage)
            else AgentMessage.model_validate(raw_message)
        )
        await execute_query(
            """
            INSERT INTO agent_messages (id, run_id, agent_name, role, content, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            uuid4(),
            UUID(str(run_id)),
            message.agent,
            message.role,
            message.content,
            message.timestamp,
        )


async def _suppress_cancelled(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return
