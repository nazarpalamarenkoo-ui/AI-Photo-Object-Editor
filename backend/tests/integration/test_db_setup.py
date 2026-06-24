import pytest
from sqlalchemy import text
from app.db.db_connect import Base


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_database_connection(db_session):
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.integration
@pytest.mark.db
@pytest.mark.asyncio
async def test_tables_created(db_engine):
    async with db_engine.begin() as conn:
        tables = await conn.run_sync(lambda c: list(Base.metadata.tables.keys()))
    assert 'users' in tables
    assert 'images' in tables
    assert 'detections' in tables