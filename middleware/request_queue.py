import asyncio
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class QueueDecision:
    status: str
    position: int
    estimated_wait_minutes: int


class WorkflowExecutionQueue:
    def __init__(self, max_concurrent: int = 3) -> None:
        self.max_concurrent = max_concurrent
        self.active_runs: set[str] = set()
        self.waiting_runs: deque[str] = deque()
        self._condition = asyncio.Condition()

    async def reserve(self, run_id: str) -> QueueDecision:
        async with self._condition:
            if run_id in self.active_runs:
                return QueueDecision(status="researching", position=0, estimated_wait_minutes=0)
            if run_id in self.waiting_runs:
                position = self._position_unlocked(run_id)
                return QueueDecision(
                    status="queued",
                    position=position,
                    estimated_wait_minutes=_estimate_wait(position),
                )
            if len(self.active_runs) < self.max_concurrent:
                self.active_runs.add(run_id)
                return QueueDecision(status="researching", position=0, estimated_wait_minutes=0)
            self.waiting_runs.append(run_id)
            position = len(self.waiting_runs)
            return QueueDecision(
                status="queued",
                position=position,
                estimated_wait_minutes=_estimate_wait(position),
            )

    async def wait_for_slot(self, run_id: str) -> None:
        async with self._condition:
            if run_id in self.active_runs:
                return
            if run_id not in self.waiting_runs:
                self.waiting_runs.append(run_id)
            while True:
                first_waiting = self.waiting_runs and self.waiting_runs[0] == run_id
                if first_waiting and len(self.active_runs) < self.max_concurrent:
                    self.waiting_runs.popleft()
                    self.active_runs.add(run_id)
                    return
                await self._condition.wait()

    async def release(self, run_id: str) -> None:
        async with self._condition:
            self.active_runs.discard(run_id)
            try:
                self.waiting_runs.remove(run_id)
            except ValueError:
                pass
            self._condition.notify_all()

    async def position(self, run_id: str) -> int:
        async with self._condition:
            return self._position_unlocked(run_id)

    async def snapshot(self) -> dict[str, object]:
        async with self._condition:
            return {
                "max_concurrent": self.max_concurrent,
                "active_runs": sorted(self.active_runs),
                "waiting_runs": list(self.waiting_runs),
            }

    def _position_unlocked(self, run_id: str) -> int:
        for index, queued_run_id in enumerate(self.waiting_runs, start=1):
            if queued_run_id == run_id:
                return index
        return 0


def _estimate_wait(position: int) -> int:
    return max(1, position * 5) if position else 0


workflow_execution_queue = WorkflowExecutionQueue(max_concurrent=3)
