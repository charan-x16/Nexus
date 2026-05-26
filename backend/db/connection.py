from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import asyncpg

from backend.config import settings

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=2,
            max_size=10,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def get_pool() -> AsyncIterator[asyncpg.Pool]:
    pool = await init_pool()
    yield pool


async def execute_query(query: str, *args: Any) -> str:
    pool = await init_pool()
    async with pool.acquire() as connection:
        return await connection.execute(query, *args)


async def fetch_rows(query: str, *args: Any) -> list[asyncpg.Record]:
    pool = await init_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(query, *args)
    return list(rows)


async def run_migrations() -> None:
    pool = await init_pool()
    migrations_dir = Path(__file__).resolve().parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))

    async with pool.acquire() as connection:
        for migration_file in migration_files:
            await connection.execute(migration_file.read_text(encoding="utf-8"))
