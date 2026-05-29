from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.workflows import router as workflows_router
from backend.config import settings
from backend.db.checkpointer import close_checkpointer, setup_checkpointer
from backend.db.connection import close_pool, init_pool, run_migrations


async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_pool()
    await run_migrations()
    setup_checkpointer(settings.DATABASE_URL)
    yield
    close_checkpointer()
    await close_pool()


app = FastAPI(
    title="Nexus AI Operating System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflows_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
    }
