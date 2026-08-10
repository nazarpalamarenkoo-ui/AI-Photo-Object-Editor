import asyncio
import io
import os
import time
import uuid

import httpx
import pytest

API_BASE_URL = os.getenv("SMOKE_API_URL", "http://localhost:8000")
ML_PREFIX = "/ml"

JOB_POLL_INTERVAL = float(os.getenv("SMOKE_JOB_POLL_INTERVAL", "1"))
JOB_TIMEOUT = float(os.getenv("SMOKE_JOB_TIMEOUT", "100_000_000"))
PICKUP_TIMEOUT = float(os.getenv("SMOKE_JOB_PICKUP_TIMEOUT", "30"))

TEST_PASSWORD = "SmokeTest123!"


@pytest.fixture(scope="session")
def api_base_url() -> str:
    return API_BASE_URL


@pytest.fixture
async def client(api_base_url):
    async with httpx.AsyncClient(base_url=api_base_url, timeout=60.0) as c:
        yield c

@pytest.fixture(scope="session")
def sample_photo_with_objects_bytes() -> bytes:
    from pathlib import Path

    asset_path = Path(__file__).parent / "assets" / "sample_photo.jpg"
    return asset_path.read_bytes()

async def _create_test_user_and_token():
    from app.services.user_service import UserService
    from app.repository.user_repo import UserRepository
    from app.db.db_connect import get_db_session
    from app.api.auth.auth import create_access_token

    suffix = uuid.uuid4().hex[:10]
    username = f"smoke_{suffix}"
    email = f"smoke_{suffix}@example.com"

    service = UserService(user_repo=UserRepository(get_db_session))
    user = await service.create_user(username=username, email=email, password=TEST_PASSWORD)

    token = create_access_token(data={"sub": user.username})
    return user, token


@pytest.fixture
async def test_user_and_token():
    user, token = await _create_test_user_and_token()
    return user, token


@pytest.fixture
async def authed_client(client, test_user_and_token):
    _, token = test_user_and_token
    client.headers["Authorization"] = f"Bearer {token}"
    return client


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(120, 160, 200)).save(buf, format="JPEG")
    return buf.getvalue()


async def wait_for_job(client: httpx.AsyncClient, job_id: str, *,
                        ml_prefix: str = ML_PREFIX,
                        timeout: float = JOB_TIMEOUT,
                        interval: float = JOB_POLL_INTERVAL) -> dict:
    deadline = time.monotonic() + timeout
    last_body = None

    while time.monotonic() < deadline:
        resp = await client.get(f"{ml_prefix}/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        last_body = body

        if body["status"] == "not_found":
            raise AssertionError(f"job {job_id} disappeared: {body}")
        if body["status"] == "complete":
            return body

        await asyncio.sleep(interval)

    raise TimeoutError(f"job {job_id} did not complete in {timeout}s, last: {last_body}")


async def wait_until_picked_up(client: httpx.AsyncClient, job_id: str, *,
                                ml_prefix: str = ML_PREFIX,
                                timeout: float = PICKUP_TIMEOUT,
                                interval: float = 0.5) -> dict:
    deadline = time.monotonic() + timeout
    last_body = None

    while time.monotonic() < deadline:
        resp = await client.get(f"{ml_prefix}/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        last_body = body

        if body["status"] not in ("deferred", "queued"):
            return body

        await asyncio.sleep(interval)

    raise TimeoutError(
        f"job {job_id} is still {last_body and last_body.get('status')} after {timeout}s "
        f"— worker may be down or stuck loading models"
    )