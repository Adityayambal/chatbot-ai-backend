from __future__ import annotations

from pathlib import Path

from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base

# Columns added to existing tables after the first release. create_all() only creates missing
# *tables*, so an existing SQLite file gets these via ALTER TABLE (which can only add columns).
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "chat_sessions": {
        "client_id": "VARCHAR(64)",
        "status": "VARCHAR(24) NOT NULL DEFAULT 'open'",
        "title": "TEXT",
        "updated_at": "DATETIME",
        "closed_at": "DATETIME",
    },
}


def _add_missing_columns(conn: Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
    conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_chat_sessions_client_status ON chat_sessions (client_id, status)"
    )


def _ensure_sqlite_dir(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    # sqlite+aiosqlite:///./data/app.db -> ./data/app.db
    path_part = database_url.split("///", 1)[-1]
    if path_part and path_part != ":memory:":
        Path(path_part).parent.mkdir(parents=True, exist_ok=True)


def make_engine(database_url: str) -> AsyncEngine:
    _ensure_sqlite_dir(database_url)
    return create_async_engine(database_url, echo=False)


async def init_db(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if engine.dialect.name == "sqlite":
            await conn.run_sync(_add_missing_columns)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
