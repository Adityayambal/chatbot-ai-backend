from __future__ import annotations

import pytest

from app.config import Settings
from app.db.database import init_db, make_engine
from tests.fakes import build_fixture_kb


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path),
        openai_api_key="",
        admin_token="test-admin-token",
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
    )


@pytest.fixture
async def fixture_kb(settings: Settings):
    return await build_fixture_kb(settings)


@pytest.fixture
async def db_session_factory(settings: Settings):
    engine = make_engine(settings.database_url)
    factory = await init_db(engine)
    yield factory
    await engine.dispose()


@pytest.fixture
async def db(db_session_factory):
    async with db_session_factory() as session:
        yield session
