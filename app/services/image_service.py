from code import interact
from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.image_repo import ImageRepository
from app.storage.s3_storage import S3Storage
from app.storage.redis_storage import RedisImageCache
from app.db.models.image import Image

class ImageService:
    
    def __init__(
        self,
        db: AsyncSession,
        s3: S3Storage,
        redis_cache: RedisImageCache,
        image_repo: ImageRepository
    ):
        
        self.db = db
        self.s3 = s3
        self.redis = redis_cache
        self.image_repo = image_repo
        
    async def upload_image(self, file: UploadFile, user_id: int) -> Image:
        self._validate_file(file)
        
        storage_path = f'uploads/{user_id}/{file.filename}'
        
        file_content = await file.read()  # читаємо один раз
        
        s3_url = await self.s3.upload_bytes(
            data=file_content,
            path=storage_path,
            content_type=file.content_type
        )
        
        cache_key = await self.redis.cache_image(
            image_id=0,
            image_data=file_content,
            suffix='original'
        )
        
        image = await self.image_repo.create(
            filename=file.filename,
            storage_path=s3_url,
            user_id=user_id,
            cache_key=cache_key
        )
        
        real_cache_key = await self.redis.cache_image(
            image_id=image.id,
            image_data=file_content,
            suffix='original'
        )
        
        image.cache_key = real_cache_key
        await self.image_repo.update(image)
        
        return image
    
    async def get_image(
        self,
        image_id: int,
        user_id: int
    ) -> Image:
        
        image = await self.image_repo.get_by_id(image_id)
        
        if not image:
            raise ValueError(f'Image {image_id} not found')
        
        if image.user_id != user_id:
            raise ValueError('Unauthorized: image belongs to different user')
        
        return image
    
    async def get_user_image(
        self,
        user_id: int,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Image]:
        
        images = await self.image_repo.get_user_images(user_id)
        
        # Apply pagination if provided
        if offset is not None:
            images = images[offset:]
        if limit is not None:
            images = images[:limit]
            
        return images
    
    async def delete_image(
        self,
        image_id: int,
        user_id: int
    ) -> bool:
        
        image = await self.get_image(image_id, user_id)
        
        # Delete from s3
        await self.s3.delete(image.storage_path)
        
        # Invalidate cache
        await self.redis.invalidate_image(image_id)
        
        # Delete from DB
        success = await self.image_repo.delete(image_id)
        
        return success
    
    async def download_image(
        self,
        image_id: int,
        user_id: int
    ) -> bytes:
        
        """
        Download images bytes from S3
        
        Args: 
            1. image_id: ID of image
            2. user_id: ID of requesting user
            
        Returns:
            bytes: Image data
        """
        # Get image (with auth check)
        image = await self.get_image(image_id, user_id)
        
        # Download bytes from S3
        image_bytes = await self.s3.download(image.storage_path)
        
        return image_bytes
    
    async def get_presigned_url(
        self,
        image_id: int,
        user_id: int,
        expiration: int = 3600
    ) -> str:
        
        image = await self.get_image(image_id, user_id)
        
        # Generate presigned URL
        url = await self.s3.get_presigned_url(
            path = image.storage_path,
            expiration = expiration
        )
        
        return url
    
    def _validate_file(self, file: UploadFile) -> None:
        """
        Validate uploaded file.
        
        Args:
            file: Uploaded file
        """
        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        if file.content_type not in allowed_types:
            raise ValueError(
                f"Invalid file type: {file.content_type}. "
                f"Allowed types: {', '.join(allowed_types)}"
            )
        
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if file.size and file.size > max_size:
            raise ValueError(
                f"File too large: {file.size / (1024*1024):.2f}MB. "
                f"Max size: 10MB"
            )