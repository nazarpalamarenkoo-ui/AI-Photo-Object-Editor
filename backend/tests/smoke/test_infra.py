import pytest
from sqlalchemy import text

pytestmark = pytest.mark.smoke


async def test_postgres_select_1():
    from app.db.db_connect import get_db_session

    async with get_db_session() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1


async def test_redis_ping():
    import redis.asyncio as redis
    from app.config.settings import settings

    r = redis.from_url(settings.REDIS_URL)
    try:
        assert await r.ping() is True
    finally:
        await r.aclose()


async def test_r2_bucket_reachable():
    """
    R2 (Cloudflare) is S3-compatible; settings.py exposes R2_ENDPOINT,
    S3_BUCKET, ACCESS_KEY, SECRET_KEY. region_name="auto" is what R2 expects.
    """
    import boto3
    from app.config.settings import settings

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT,
        aws_access_key_id=settings.ACCESS_KEY,
        aws_secret_access_key=settings.SECRET_KEY,
        region_name="auto",
    )
    response = s3.head_bucket(Bucket=settings.S3_BUCKET)
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200