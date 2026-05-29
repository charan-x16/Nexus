import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


@dataclass(frozen=True)
class RateLimit:
    max_requests: int
    window_seconds: int


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: RateLimit) -> tuple[bool, int]:
        now = time.time()
        timestamps = self._requests[key]
        while timestamps and timestamps[0] <= now - limit.window_seconds:
            timestamps.popleft()
        if len(timestamps) >= limit.max_requests:
            retry_after = max(1, int(limit.window_seconds - (now - timestamps[0])))
            return False, retry_after
        timestamps.append(now)
        return True, 0


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.limiter = SlidingWindowRateLimiter()
        self.general_limit = RateLimit(max_requests=100, window_seconds=60)
        self.workflow_limit = RateLimit(max_requests=10, window_seconds=3600)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        client_ip = _client_ip(request)
        general_ok, general_retry = self.limiter.check(
            f"general:{client_ip}",
            self.general_limit,
        )
        if not general_ok:
            return _rate_limited(general_retry, "API rate limit exceeded.")

        if request.method == "POST" and request.url.path.rstrip("/") == "/workflows":
            workflow_ok, workflow_retry = self.limiter.check(
                f"workflow:{client_ip}",
                self.workflow_limit,
            )
            if not workflow_ok:
                return _rate_limited(
                    workflow_retry,
                    "Workflow submission limit exceeded.",
                )

        return await call_next(request)


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(retry_after: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": detail, "retry_after": retry_after},
        headers={"Retry-After": str(retry_after)},
    )
