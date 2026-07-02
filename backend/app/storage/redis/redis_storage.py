import pickle
from typing import Optional

import redis.asyncio as redis

from app.config.settings import settings


class RedisStorage:

    def __init__(self):
        self.redis = redis.from_url(
            settings.REDIS_URL,
            encoding="utf8",
            decode_responses=False,
        )
        self.default_ttl = 3600

    async def set(self, key: str, value: bytes, ttl: Optional[int] = None):
        await self.redis.setex(key, ttl or self.default_ttl, value)

    async def get(self, key: str) -> Optional[bytes]:
        return await self.redis.get(key)

    async def delete(self, *keys: str):
        await self.redis.delete(*keys)

    async def exists(self, key: str) -> bool:
        return (await self.redis.exists(key)) > 0

    async def cache_image(
        self,
        image_id: int,
        image_data: bytes,
        suffix: str = "processed",
        ttl: int = 7200,
    ):
        key = f"image:{image_id}:{suffix}"
        await self.set(key, image_data, ttl)
        return key

    async def get_cache_image(
        self,
        image_id: int,
        suffix: str = "processed",
    ):
        return await self.get(f"image:{image_id}:{suffix}")


    async def cache_detections(
        self,
        image_id: int,
        detections: list,
        ttl: int = 3600,
    ):
        await self.set(
            f"detections:{image_id}",
            pickle.dumps(detections),
            ttl,
        )

    async def get_cached_detections(self, image_id: int):
        data = await self.get(f"detections:{image_id}")
        return pickle.loads(data) if data else None

    async def cache_segments(
        self,
        image_id: int,
        segments: list,
        ttl: int = 7200,
    ):
        await self.set(
            f"segments:{image_id}",
            pickle.dumps(segments),
            ttl,
        )

    async def get_cached_segments(self, image_id: int):
        data = await self.get(f"segments:{image_id}")
        return pickle.loads(data) if data else None

    async def invalidate_segments(self, image_id: int):
        await self.delete(f"segments:{image_id}")

    async def invalidate_image(self, image_id: int):
        await self.delete(
            f"image:{image_id}:processed",
            f"image:{image_id}:thumbnail",
            f"detections:{image_id}",
            f"segments:{image_id}",
        )

    async def close(self):
        await self.redis.aclose()