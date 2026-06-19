import redis.asyncio as redis
import pickle
from typing import Optional
from app.config.settings import settings


class RedisImageCache:

    MAX_HISTORY = 10

    def __init__(self):
        self.redis = redis.from_url(
            settings.REDIS_URL,
            encoding='utf8',
            decode_responses=False
        )
        self.defaul_ttl = 3600

    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        await self.redis.setex(key, ttl or self.defaul_ttl, value)

    async def get(self, key: str) -> Optional[bytes]:
        return await self.redis.get(key)  # type: ignore

    async def delete(self, key: str) -> None:
        await self.redis.delete(key)

    async def exists(self, key: str) -> bool:
        result = await self.redis.exists(key)
        return result > 0  # type: ignore

    async def cache_image(self, image_id: int, image_data: bytes, suffix: str = 'processed', ttl: Optional[int] = None) -> str:
        key = f'image:{image_id}:{suffix}'
        await self.set(key, image_data, ttl)
        return key

    async def get_cache_image(self, image_id: int, suffix: str = 'processed') -> Optional[bytes]:
        key = f'image:{image_id}:{suffix}'
        return await self.get(key)

    async def cache_detections(self, image_id: int, detections: list, ttl: int = 3600) -> str:
        key = f'detections:{image_id}'
        serialized = pickle.dumps(detections)
        await self.set(key, serialized, ttl)
        return key

    async def get_cached_detections(self, image_id: int) -> Optional[list]:
        key = f'detections:{image_id}'
        data = await self.get(key)
        if data:
            return pickle.loads(data)  # type: ignore
        return None


    async def push_undo_state(self, image_id: int, image_bytes: bytes, label: str, ttl: int = 7200) -> int:
        undo_key = f'image:{image_id}:undo_stack'
        redo_key = f'image:{image_id}:redo_stack'

        entry = pickle.dumps({'bytes': image_bytes, 'label': label})

        pipe = self.redis.pipeline()
        pipe.lpush(undo_key, entry)
        pipe.ltrim(undo_key, 0, self.MAX_HISTORY - 1)
        pipe.expire(undo_key, ttl)
        pipe.delete(redo_key)
        pipe.llen(undo_key)
        results = await pipe.execute()  # type: ignore

        return int(results[-1])

    async def pop_undo_state(self, image_id: int) -> Optional[dict]:
        undo_key = f'image:{image_id}:undo_stack'
        data = await self.redis.lpop(undo_key)  # type: ignore
        if data:
            return pickle.loads(data)  # type: ignore
        return None

    async def push_redo_state(self, image_id: int, image_bytes: bytes, label: str, ttl: int = 7200) -> None:
        redo_key = f'image:{image_id}:redo_stack'
        entry = pickle.dumps({'bytes': image_bytes, 'label': label})

        pipe = self.redis.pipeline()
        pipe.lpush(redo_key, entry)
        pipe.ltrim(redo_key, 0, self.MAX_HISTORY - 1)
        pipe.expire(redo_key, ttl)
        await pipe.execute()  # type: ignore

    async def pop_redo_state(self, image_id: int) -> Optional[dict]:
        redo_key = f'image:{image_id}:redo_stack'
        data = await self.redis.lpop(redo_key)  # type: ignore
        if data:
            return pickle.loads(data)  # type: ignore
        return None

    async def get_history_labels(self, image_id: int) -> list[str]:
        undo_key = f'image:{image_id}:undo_stack'
        entries = await self.redis.lrange(undo_key, 0, -1)  # type: ignore
        labels = []
        for e in entries:
            try:
                labels.append(pickle.loads(e)['label'])  # type: ignore
            except Exception:
                pass
        return labels

    async def clear_history(self, image_id: int) -> None:
        await self.redis.delete(
            f'image:{image_id}:undo_stack',
            f'image:{image_id}:redo_stack'
        )

    async def invalidate_image(self, image_id: int) -> None:
        await self.redis.delete(
            f'image:{image_id}:processed',
            f'image:{image_id}:thumbnail',
            f'detections:{image_id}',
            f'image:{image_id}:undo_stack',
            f'image:{image_id}:redo_stack',
        )

    async def close(self) -> None:
        await self.redis.aclose()  # type: ignore