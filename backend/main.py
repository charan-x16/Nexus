from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middleware.rate_limiter import RateLimiterMiddleware

from backend.api.routes.monitoring import router as monitoring_router
from backend.api.routes.observability import metrics_router, router as observability_router
from backend.api.routes.projects import router as projects_router
from backend.api.routes.workflows import router as workflows_router
from backend.config import settings
from backend.db.checkpointer import close_checkpointer, setup_checkpointer
from backend.db.connection import close_pool, init_pool, run_migrations
from backend.monitoring.scheduler import monitoring_scheduler
from backend.tasks.background_runner import workflow_runner


async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_pool()
    await run_migrations()
    setup_checkpointer(settings.DATABASE_URL)
    await workflow_runner.start()
    await monitoring_scheduler.start()
    yield
    await monitoring_scheduler.stop()
    await workflow_runner.stop()
    close_checkpointer()
    await close_pool()


app = FastAPI(
    title="Nexus AI Operating System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimiterMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router)
app.include_router(workflows_router)
app.include_router(monitoring_router)
app.include_router(observability_router)
app.include_router(metrics_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
    }
