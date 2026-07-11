from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config.settings import settings
from contextlib import asynccontextmanager

DATABASE_URL = settings.DATABASE_URL

Base = declarative_base()

engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True
)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@asynccontextmanager
async def get_db_session():

    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()