from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth.auth import get_current_user
from app.db.db_connect import get_db
from app.db.models.user import User
from app.db.schemas.image import ImageResponse
from app.repository.image_repo import ImageRepository
from app.services.image_service import ImageService
from app.storage.s3_storage import S3Storage
from app.storage.redis_storage import RedisImageCache

router = APIRouter(prefix="/images", tags=["Images"])

def get_image_service(db: AsyncSession = Depends(get_db)) -> ImageService:
    return ImageService(
        db=db,
        s3=S3Storage(),
        redis_cache=RedisImageCache(),
        image_repo=ImageRepository(db)
    )

@router.post("/upload", response_model=ImageResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    service: ImageService = Depends(get_image_service)
):
    """Upload image to S3 and save metadata to DB."""
    try:
        image = await service.upload_image(file=file, user_id=current_user.id)
        return image
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=list[ImageResponse])
async def get_user_images(
    limit: Optional[int] = Query(None, ge=1, le=100),
    offset: Optional[int] = Query(None, ge=0),
    current_user: User = Depends(get_current_user),
    service: ImageService = Depends(get_image_service)
):
    """Get all images for current user with optional pagination."""
    images = await service.get_user_image(
        user_id=current_user.id,
        limit=limit,
        offset=offset
    )
    return images


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: ImageService = Depends(get_image_service)
):
    """Get image metadata by ID."""
    try:
        image = await service.get_image(image_id=image_id, user_id=current_user.id)
        return image
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{image_id}/download")
async def download_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: ImageService = Depends(get_image_service)
):
    """Download image bytes directly from S3."""
    try:
        image_bytes = await service.download_image(
            image_id=image_id,
            user_id=current_user.id
        )
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename={image_id}.jpg"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{image_id}/url")
async def get_presigned_url(
    image_id: int,
    expiration: int = Query(3600, ge=60, le=86400),
    current_user: User = Depends(get_current_user),
    service: ImageService = Depends(get_image_service)
):
    """Get temporary presigned URL for image download."""
    try:
        url = await service.get_presigned_url(
            image_id=image_id,
            user_id=current_user.id,
            expiration=expiration
        )
        return {"url": url, "expires_in": expiration}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{image_id}", status_code=204)
async def delete_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    service: ImageService = Depends(get_image_service)
):
    """Delete image from S3, Redis cache and DB."""
    try:
        await service.delete_image(image_id=image_id, user_id=current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))