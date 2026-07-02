import pickle

from app.storage.redis.redis_storage import RedisStorage


class RedisHistory(RedisStorage):

    MAX_HISTORY = 10

    async def push_undo_state(
        self,
        image_id: int,
        image_bytes: bytes,
        label: str,
        ttl: int = 7200,
    ) -> int:

        undo_key = f"image:{image_id}:undo_stack"
        redo_key = f"image:{image_id}:redo_stack"

        entry = pickle.dumps(
            {
                "bytes": image_bytes,
                "label": label,
            }
        )

        pipe = self.redis.pipeline()

        pipe.lpush(undo_key, entry)
        pipe.ltrim(undo_key, 0, self.MAX_HISTORY - 1)
        pipe.expire(undo_key, ttl)
        pipe.delete(redo_key)
        pipe.llen(undo_key)

        result = await pipe.execute()

        return int(result[-1])

    async def pop_undo_state(self, image_id: int):

        data = await self.redis.lpop(f"image:{image_id}:undo_stack") # type: ignore

        if data:
            return pickle.loads(data) # type: ignore

        return None

    async def push_redo_state(
        self,
        image_id: int,
        image_bytes: bytes,
        label: str,
        ttl: int = 7200,
    ):

        redo_key = f"image:{image_id}:redo_stack"

        entry = pickle.dumps(
            {
                "bytes": image_bytes,
                "label": label,
            }
        )

        pipe = self.redis.pipeline()

        pipe.lpush(redo_key, entry)
        pipe.ltrim(redo_key, 0, self.MAX_HISTORY - 1)
        pipe.expire(redo_key, ttl)

        await pipe.execute()

    async def pop_redo_state(self, image_id: int):

        data = await self.redis.lpop(f"image:{image_id}:redo_stack") # type: ignore

        if data:
            return pickle.loads(data) # type: ignore

        return None

    async def get_history_labels(
        self,
        image_id: int,
    ) -> list[str]:

        entries = await self.redis.lrange(f"image:{image_id}:undo_stack", 0,-1) # type: ignore

        labels = []

        for entry in entries:
            try:
                labels.append(
                    pickle.loads(entry)["label"]
                )
            except Exception:
                pass

        return labels

    async def clear_history(
        self,
        image_id: int,
    ):

        await self.delete(
            f"image:{image_id}:undo_stack",
            f"image:{image_id}:redo_stack",
        )