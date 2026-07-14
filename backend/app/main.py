import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import configure_logging, get_logger, RequestLoggingMiddleware

configure_logging()

from app.db.db_connect import engine, Base
from app.api.auth.routes import router as auth_router
from app.api.v1.user import router as user_router
from app.api.v1.image import router as image_router
from app.api.v1.detection import router as detection_router
from app.api.v1.ml import router as ml_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_starting")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("app_started")
    yield
    logger.info("app_shutting_down")


app = FastAPI(
    title="Image Editor API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(image_router)
app.include_router(detection_router)
app.include_router(ml_router)


@app.get("/health")
async def health():
    return {"status": "ok"}