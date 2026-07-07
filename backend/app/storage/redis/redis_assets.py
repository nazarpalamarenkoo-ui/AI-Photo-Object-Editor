import pickle
import uuid
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

from PIL import Image

from app.storage.redis.redis_storage import RedisStorage


class RedisAssetsStorage(RedisStorage):
    """
    User's library of cut-out (extracted) objects, backed by Redis.

    Keys:
        asset:{user_id}:{asset_id}:meta   — pickled dict (no heavy bytes)
        asset:{user_id}:{asset_id}:data   — raw bytes of the extracted RGBA PNG object
        asset:{user_id}:{asset_id}:thumb  — raw bytes of the preview thumbnail (for panel display)
        assets_index:{user_id}            — sorted set, score=created_at, member=asset_id
    """

    ASSET_TTL = 60 * 60 * 24 * 30      # 30 days
    MAX_ASSETS_PER_USER = 200
    THUMB_SIZE = (160, 160)

    def _keys(self, user_id: int, asset_id: str) -> Dict[str, str]:
        """Build the Redis key names (meta/data/thumb) for a given asset."""
        base = f"asset:{user_id}:{asset_id}"
        return {"meta": f"{base}:meta", "data": f"{base}:data", "thumb": f"{base}:thumb"}

    def _index_key(self, user_id: int) -> str:
        """Return the sorted-set key that indexes all asset IDs for a user."""
        return f"assets_index:{user_id}"

    def _make_thumbnail(self, extracted_bytes: bytes) -> bytes:
        """
        Generate a PNG thumbnail (bounded by THUMB_SIZE) from extracted
        object bytes, preserving the alpha channel.

        Args:
            extracted_bytes: Raw bytes of the extracted RGBA PNG object.

        Returns:
            PNG-encoded thumbnail bytes.
        """
        img = Image.open(BytesIO(extracted_bytes)).convert("RGBA")
        img.thumbnail(self.THUMB_SIZE, Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def save_asset(
        self,
        user_id: int,
        extracted_bytes: bytes,
        source_image_id: int,
        object_size: tuple,
        area_pixels: int,
        label: Optional[str] = None,
        s3_url: Optional[str] = None,
        ttl: int = ASSET_TTL,
    ) -> Dict:
        """
        Save a new extracted object as an asset: generates a thumbnail,
        stores metadata/data/thumbnail in Redis with a TTL, adds it to
        the user's asset index, and enforces the per-user asset limit.

        Args:
            user_id: ID of the asset owner.
            extracted_bytes: Raw bytes of the extracted RGBA PNG object.
            source_image_id: ID of the image the object was extracted from.
            object_size: (width, height) of the extracted object.
            area_pixels: Number of non-transparent pixels in the object.
            label: Optional human-readable label for the asset.
            s3_url: Optional S3 URL if the asset was also persisted to S3.
            ttl: Time-to-live in seconds for the stored Redis keys.

        Returns:
            The metadata dict for the newly created asset (without bytes).
        """
        asset_id = uuid.uuid4().hex
        created_at = datetime.utcnow()
        thumb_bytes = self._make_thumbnail(extracted_bytes)

        meta = {
            "asset_id": asset_id,
            "user_id": user_id,
            "source_image_id": source_image_id,
            "object_size": object_size,
            "area_pixels": area_pixels,
            "label": label,
            "s3_url": s3_url,
            "created_at": created_at.isoformat(),
        }

        keys = self._keys(user_id, asset_id)
        pipe = self.redis.pipeline()
        pipe.setex(keys["meta"], ttl, pickle.dumps(meta))
        pipe.setex(keys["data"], ttl, extracted_bytes)
        pipe.setex(keys["thumb"], ttl, thumb_bytes)
        pipe.zadd(self._index_key(user_id), {asset_id: created_at.timestamp()})
        pipe.expire(self._index_key(user_id), ttl)
        await pipe.execute()

        await self._enforce_limit(user_id)
        return meta

    async def _enforce_limit(self, user_id: int) -> None:
        """
        Ensure the user's asset count stays within MAX_ASSETS_PER_USER by
        deleting the oldest assets once the limit is exceeded.

        Args:
            user_id: ID of the asset owner.
        """
        index_key = self._index_key(user_id)
        count = await self.redis.zcard(index_key)
        if count <= self.MAX_ASSETS_PER_USER:
            return

        overflow = count - self.MAX_ASSETS_PER_USER
        oldest = await self.redis.zrange(index_key, 0, overflow - 1)
        for raw_id in oldest:
            asset_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            await self.delete_asset(user_id, asset_id)

    async def get_asset(self, user_id: int, asset_id: str, with_bytes: bool = True) -> Optional[Dict]:
        """
        Fetch a single asset's metadata, optionally including its raw
        extracted bytes.

        Args:
            user_id: ID of the asset owner.
            asset_id: ID of the asset to fetch.
            with_bytes: If True, also load and attach `extracted_bytes`
                to the returned dict.

        Returns:
            The asset metadata dict (with `extracted_bytes` if requested),
            or None if the asset doesn't exist / has expired.
        """
        keys = self._keys(user_id, asset_id)
        meta_raw = await self.redis.get(keys["meta"])
        if not meta_raw:
            return None

        meta = pickle.loads(meta_raw)
        if with_bytes:
            meta["extracted_bytes"] = await self.redis.get(keys["data"])
        return meta

    async def get_thumbnail(self, user_id: int, asset_id: str) -> Optional[bytes]:
        """
        Fetch the thumbnail PNG bytes for an asset.

        Args:
            user_id: ID of the asset owner.
            asset_id: ID of the asset.

        Returns:
            Thumbnail PNG bytes, or None if not found.
        """
        return await self.redis.get(self._keys(user_id, asset_id)["thumb"])

    async def list_assets(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        List a user's assets (metadata only, no bytes), ordered from
        newest to oldest.

        Args:
            user_id: ID of the asset owner.
            limit: Maximum number of assets to return.
            offset: Number of most-recent assets to skip (for pagination).

        Returns:
            A list of asset metadata dicts.
        """
        index_key = self._index_key(user_id)
        ids = await self.redis.zrevrange(index_key, offset, offset + limit - 1)
        if not ids:
            return []

        meta_keys = [
            self._keys(user_id, raw_id.decode() if isinstance(raw_id, bytes) else raw_id)["meta"]
            for raw_id in ids
        ]
        raw_metas = await self.redis.mget(meta_keys)

        results = []
        for raw in raw_metas:
            if raw:
                results.append(pickle.loads(raw))
        return results

    async def rename_asset(self, user_id: int, asset_id: str, label: str) -> Optional[Dict]:
        """
        Update an asset's label, preserving its existing Redis TTL.

        Args:
            user_id: ID of the asset owner.
            asset_id: ID of the asset to rename.
            label: New label to store.

        Returns:
            The updated metadata dict, or None if the asset doesn't exist.
        """
        meta = await self.get_asset(user_id, asset_id, with_bytes=False)
        if not meta:
            return None
        meta["label"] = label
        keys = self._keys(user_id, asset_id)
        ttl = await self.redis.ttl(keys["meta"])
        await self.redis.setex(keys["meta"], ttl if ttl > 0 else self.ASSET_TTL, pickle.dumps(meta))
        return meta

    async def delete_asset(self, user_id: int, asset_id: str) -> bool:
        """
        Delete an asset's metadata, extracted bytes, thumbnail, and its
        entry in the user's asset index.

        Args:
            user_id: ID of the asset owner.
            asset_id: ID of the asset to delete.

        Returns:
            True if the asset's metadata key existed and was deleted,
            False otherwise.
        """
        keys = self._keys(user_id, asset_id)
        pipe = self.redis.pipeline()
        pipe.delete(keys["meta"], keys["data"], keys["thumb"])
        pipe.zrem(self._index_key(user_id), asset_id)
        result = await pipe.execute()
        return bool(result[0])
