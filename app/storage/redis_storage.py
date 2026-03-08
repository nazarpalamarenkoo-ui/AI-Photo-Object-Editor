import redis.asyncio as redis
import pickle
from typing import Optional, Any
from app.config.settings import settings

class RedisImageCache:
    
    def init(self):
        
        self.redis = redis.from_url(
            settings.REDIS_URL,
            encoding = 'utf8',
            decode_responses = False # Binary data
        )
        self.defaul_ttl = 3600 # 1 hour
        
    async def set(
        self,
        key: str,
        value: bytes,
        ttl: int = None
    ) -> None:
        
        await self.redis.setex(
            key,
            ttl or self.defaul_ttl,
            value
        )
        
    async def get(self, key: str) -> Optional[bytes]:
        
        return await self.redis.get(key)
        
    async def delete(self, key: str) -> None:
        
        await self.redis.delete(key)
        
    async def exists(self, key: str) -> bool:
        
        return await self.redis.exists(key) > 0
    
    async def cache_image(
        self,
        image_id: int,
        image_data: bytes,
        suffix: str = 'processed',
        ttl: int = None
    ) -> str:
        
        key = f'image:{image_id}:{suffix}'
        await self.set(key, image_data, ttl)
        return key
    
    async def get_cache_image(
        self,
        image_id: int,
        suffix: str = 'processed'
    ) -> Optional[bytes]:
        
        key = f'image:{image_id}:{suffix}'
        return await self.get(key)
        
    async def cache_detections(
        self,
        image_id: int,
        detections: list,
        ttl: int = 3600
    ) -> str:
        
        key = f'detections:{image_id}'
        serialized = pickle.dumps(detections)
        await self.set(key, serialized, ttl)
        
        return key
    
    async def get_cached_detections(
        self,
        image_id: int
    ) -> Optional[list]:
        
        key = f'detections:{image_id}'
        data = await self.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    async def invalidate_image(self, image_id: int) -> None:
        
        keys = [
            f'image:{image_id}:processed',
            f'image:{image_id}:thumbnail',
            f'detections:{image_id}'
        ]
        
        for key in keys:
            await self.delete(key)
            
    async def close(self) -> None:
        
        await self.redis.close()