from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.pipeline import MLPipeline, get_pipeline
from app.storage.s3_storage import S3Storage
from app.storage.redis_storage import RedisImageCache
from app.repository.image_repo import ImageRepository
from app.repository.detection_repo import DetectionRepository
from app.db.models.detection import Detection
from app.db.models.image import Image

class MLService:
    
    """
    ML Service - coordinates ML pipeline with database and storage.
    
    Workflow:
        User uploads image -> detect_objects -> save to DB
        -> User select bbox -> remove/replace_object -> save result
    """
    
    def __init__(
        self,
        db: AsyncSession,
        s3_storage: S3Storage,
        redis_storage: RedisImageCache,
        image_repo: ImageRepository,
        detection_repo: DetectionRepository,
        pipeline: Optional[MLPipeline] = None,
        device: str = 'cpu'
    ):
        """
        Initialize ML Service with injected dependencies.
        
        Args:
            1. db: SQLAlchemy async session
            2. s3_storage: S3/R2 storage client
            3. redis_storage: Redis cache client
            4. image_repo: Image repository
            5. detection_repo: Detection repository
            6: pipeline: ML pipeline instance (default: auto-created)
            7: device: Device for ML operation ('cuda' or 'cpu')
        """
        
        self.db = db
        self.s3 = s3_storage
        self.redis = redis_storage
        self.image_repo = image_repo
        self.detection_repo = detection_repo
        self.pipeline = pipeline or get_pipeline(device = device)
    
    
    async def _get_current_image_bytes(self, image_id: int, storage_path: str) -> bytes:
        """
        Get current working image bytes.
        Tries Redis current state first, falls back to S3 original.
        """
        cached = await self.redis.get_cache_image(image_id, suffix='current_state')
        if cached:
            return cached
        return await self.s3.download(storage_path)
 
    async def _save_current_state(self, image_id: int, image_bytes: bytes) -> None:
        """Save current working state to Redis (TTL 2 hours)."""
        await self.redis.cache_image(
            image_id=image_id,
            image_data=image_bytes,
            suffix='current_state',
            ttl=7200
        )
 
    async def reset_current_state(self, image_id: int) -> None:
        """Reset current state — next operation will use original S3 image."""
        await self.redis.delete(f'image:{image_id}:current_state')
        await self.redis.clear_history(image_id)

    async def _get_image_authorized(self, image_id: int, user_id: int):
        image = await self.image_repo.get_by_id(image_id)
        if not image:
            raise ValueError(f'Image {image_id} not found')
        if image.user_id != user_id:
            raise ValueError('Unauthorized: image belongs to different user')
        return image   
    
    async def save_result(self, image_id: int, user_id: int) -> Image:
        image = await self._get_image_authorized(image_id, user_id)

        result_bytes = await self.redis.get_cache_image(image_id, suffix='current_state')
        if not result_bytes:
            raise ValueError('No processed result to save. Run remove/replace first.')

        result_path = f"saved/{user_id}/{image_id}/result_{int(datetime.utcnow().timestamp())}.jpg"

        result_s3_uri = await self.s3.upload_bytes(
            data=result_bytes,
            path=result_path,
            content_type='image/jpeg'
        )

        saved = await self.image_repo.create(
            filename=f"edited_{image.filename}",
            storage_path=result_s3_uri,
            user_id=user_id,
            cache_key=None
        )
        saved.status = 'processed'
        await self.image_repo.update(saved)
        return saved
    
    async def undo(self, image_id: int, user_id: int) -> Dict:
        """Undo last operation — pop from undo stack, push current to redo."""
        await self._get_image_authorized(image_id, user_id)

        current = await self.redis.get_cache_image(image_id, suffix='current_state')
        prev_state = await self.redis.pop_undo_state(image_id)

        if not prev_state:
            raise ValueError('Nothing to undo')

        if current:
            await self.redis.push_redo_state(image_id, current, label='redo')

        await self._save_current_state(image_id, prev_state['bytes'])

        presigned_url = await self._get_temp_url_from_bytes(image_id, user_id, prev_state['bytes'], 'undo')

        return {
            'presigned_url': presigned_url,
            'label': prev_state['label'],
            'history': await self.redis.get_history_labels(image_id)
        }

    async def redo(self, image_id: int, user_id: int) -> Dict:
        """Redo last undone operation."""
        await self._get_image_authorized(image_id, user_id)

        current = await self.redis.get_cache_image(image_id, suffix='current_state')
        next_state = await self.redis.pop_redo_state(image_id)

        if not next_state:
            raise ValueError('Nothing to redo')

        if current:
            await self.redis.push_undo_state(image_id, current, label='redo_checkpoint')

        await self._save_current_state(image_id, next_state['bytes'])

        presigned_url = await self._get_temp_url_from_bytes(image_id, user_id, next_state['bytes'], 'redo')

        return {
            'presigned_url': presigned_url,
            'label': next_state['label'],
            'history': await self.redis.get_history_labels(image_id)
        }

    async def get_history(self, image_id: int, user_id: int) -> Dict:
        """Get undo stack labels for UI."""
        await self._get_image_authorized(image_id, user_id)
        labels = await self.redis.get_history_labels(image_id)
        return {'history': labels}

    async def _get_temp_url_from_bytes(self, image_id: int, user_id: int, image_bytes: bytes, op: str) -> str:
        """Upload bytes to S3 temp path and return presigned URL."""
        path = f"temp/{user_id}/{image_id}/{op}_{int(datetime.utcnow().timestamp())}.jpg"
        await self.s3.upload_bytes(data=image_bytes, path=path, content_type='image/jpeg')
        return await self.s3.get_presigned_url(path=path, expiration=3600)
    
    
    async def detect_objects(
        self,
        image_id: int,
        user_id: int,
        conf_threshold: float = 0.5,
        classes: Optional[List[str]] = None
    ) -> Dict:
        
        """
        Detection objects in upload image
        
        Workflow:
            1. Get image from database
            2. Download image bytes from s3
            3. Run ML detection (YOLO)
            4. Save detections to database
            5. Cache detections in Redis
            6. Return detection results
            
        Args:
            1. image_id: ID of image to process
            2. user_id: ID of user (for authorization check)
            3. conf_threshold: Confidence threshold (0.0-1.0, default: 0.5)
            4. classes: Optional list of class names to filter
            
        Returns:
            Dict{
                - detections: List[Dict] - detected objects with bbox
                - image_size: Tuple[int, int] - (width, height)
                - metrics: Dict - detection metrics
                - timestamp: str - ISO timestamp
            }
            
        Raises:
            ValueError: If image not found or unauthorized
        """
        # Get image from db
        image = await self.image_repo.get_by_id(image_id)
        
        if not image:
            raise ValueError(f'Image {image_id} not found')
        
        if image.user_id != user_id:
            raise ValueError(f'Unauthorized: image belong to different user')
        
        # Download image bytes from S3/R2
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
        # Run ML pipeline
        result = await self.pipeline.detect_objects(
            image_bytes = image_bytes,
            conf_threshold = conf_threshold,
            classes = classes,
            track_metrics = True
        )
        
        # Save detections to database
        detections = result['detections']
        
        # Convert ML format to DB models
        db_detections = []
        
        for det in detections:
            db_detection = Detection(
                image_id=image_id,
                bbox_id=det['bbox_id'],
                detected_class=det['detected_class'],
                confidence=det['confidence'],
                x1=det['x1'],
                y1=det['y1'],
                x2=det['x2'],
                y2=det['y2']
            )
            db_detections.append(db_detection)
            
        await self.detection_repo.delete_by_image(image_id)
        await self.redis.delete(f'image:{image_id}:detections')
        await self.detection_repo.create_many(db_detections)
        
        # Cache detections in Redis
        await self.redis.cache_detections(
            image_id = image_id,
            detections = detections,
            ttl = 3600
        )
        
        return result
    
    async def remove_object(
        self,
        image_id: int,
        bbox_id: int,
        user_id: int,
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True,
        ldm_steps: int = 25, 
        ldm_sampler: str = 'plms', 
        hd_strategy: str = 'CROP'
    ) -> Dict:
        
        """
        Remove object from image
        
        Workflow:
            1. Get image and detection from database
            2. Download original image from S3
            3. Run ML removal (YOLO + LaMa + processors)
            4. Upload result to S3
            5. Update image record with result path
            6. Return result with download URL
            
        Args:
            1. image_id: ID of image to process
            2. bbox_id: ID of bbox to remove
            3. user_id: ID of user (for authorization)
            4. expand_mask_pixels: Pixels to expand mask (default: 5)
            5. use_edge_blending: Apply edge blending (default: True)
            
        Returns:
            Dict with{
                - result_url: str - S3 path to result image
                - presigned_url: str - Temporary download URL
                - metrics: Dict - processing metrics
                - timestamp: str - ISO timestamp
            }
        
        Raises:
            ValueError: If image/detection not found or unauthorized
        """
        
        image = await self._get_image_authorized(image_id, user_id)
 
        detections = await self.detection_repo.get_by_image(image_id)
        detection = next((d for d in detections if d.bbox_id == bbox_id), None)
        if not detection:
            raise ValueError(f'Detection with bbox_id {bbox_id} not found')
        
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
        await self.redis.push_undo_state(image_id, image_bytes, label=f'remove bbox_id={bbox_id}')
        
        # Run ML removal
        selected_bbox = {
            'x1': detection.x1,
            'y1': detection.y1,
            'x2': detection.x2,
            'y2': detection.y2
        }
        scene_bboxes = [{'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2} for d in detections]
        
        result = await self.pipeline.remove_object(
            image_bytes = image_bytes,
            selected_bbox = selected_bbox,
            expand_mask_pixels = expand_mask_pixels,
            use_edge_blending = use_edge_blending,
            scene_bboxes=scene_bboxes,
            track_metrics = True,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy
        )
        
        # Upload result to S3
        result_path = f"results/{user_id}/{image_id}/remove_{bbox_id}_{int(datetime.utcnow().timestamp())}.jpg"
        
        result_url = await self.s3.upload_bytes(
            data = result['result_bytes'],
            path = result_path,
            content_type = 'image/jpeg'
        )

        await self._save_current_state(image_id, result['result_bytes'])

        await self.detection_repo.delete_by_image(image_id)
        await self.redis.delete(f'image:{image_id}:detections')
        # Generate presigned URL for download
        presigned_url = await self.s3.get_presigned_url(
            path=result_path,
            expiration=3600
        )
        
        return {
            'result_url': result_url,
            'presigned_url': presigned_url,
            'metrics': result['metrics'],
            'timestamp': result['timestamp']
        }
        
    async def replace_object(
        self,
        image_id: int,
        bbox_id: int,
        replace_image_bytes: bytes,
        user_id: int,
        expand_mask_pixels: int = 25,
        use_color_matching: bool = True,
        use_edge_blending: bool = True,
        color_match_method: str = 'mean_std',
        ldm_steps: int = 25, 
        ldm_sampler: str = 'plms', 
        hd_strategy: str = 'CROP'
    ) -> Dict:
        
        """
        Replace object in image with replacement image.
        
        Workflow:
            1. Get image and detection from database
            2. Download original image from S3
            3. Run ML replacement (YOLO + LaMa + processors)
            4. Upload result to S3
            5. Return result with download URL
        
        Args:
            1. image_id: ID of image to process
            2. bbox_id: ID of bbox to replace
            3. replacement_image_bytes: Replacement object image bytes
            4. user_id: ID of user (for authorization)
            5. expand_mask_pixels: Pixels to expand mask (default: 0)
            6. use_color_matching: Apply color matching (default: True)
            7. use_edge_blending: Apply edge blending (default: True)
            8. color_match_method: Color matching method (default: 'mean_std')
        
        Returns:
            Dict {
                - result_url: str - S3 path to result image
                - presigned_url: str - Temporary download URL
                - metrics: Dict - processing metrics
                - timestamp: str - ISO timestamp
            }
        Raises:
            ValueError: If image/detection not found or unauthorized
        """
        
        image = await self._get_image_authorized(image_id, user_id)
 
        detections = await self.detection_repo.get_by_image(image_id)
        detection = next((d for d in detections if d.bbox_id == bbox_id), None)
        if not detection:
            raise ValueError(f'Detection with bbox_id {bbox_id} not found')
        
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
        await self.redis.push_undo_state(image_id, image_bytes, label=f'replace bbox_id={bbox_id}')
        
        # Run ML replacement
        selected_bbox = {
            'x1': detection.x1,
            'y1': detection.y1,
            'x2': detection.x2,
            'y2': detection.y2
        }
        scene_bboxes = [{'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2} for d in detections]

        result = await self.pipeline.replace_object(
            image_bytes=image_bytes,
            selected_bbox=selected_bbox,
            replacement_image_bytes=replace_image_bytes,
            expand_mask_pixels=expand_mask_pixels,
            use_color_matching=use_color_matching,
            use_edge_blending=use_edge_blending,
            color_match_method=color_match_method,
            scene_bboxes=scene_bboxes,
            track_metrics=True,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy
        )
 
        result_bytes = result['result_bytes']
 
        # Save as current state for next operation
        await self._save_current_state(image_id, result_bytes)

        await self.detection_repo.delete_by_image(image_id)
        await self.redis.delete(f'image:{image_id}:detections')
 
        # Upload to S3
        result_path = f"results/{user_id}/{image_id}/replace_{bbox_id}_{int(datetime.utcnow().timestamp())}.jpg"
        result_url = await self.s3.upload_bytes(
            data=result_bytes,
            path=result_path,
            content_type='image/jpeg'
        )
        presigned_url = await self.s3.get_presigned_url(path=result_path, expiration=3600)
 
        return {
            'result_url': result_url,
            'presigned_url': presigned_url,
            'metrics': result['metrics'],
            'timestamp': result['timestamp']
        }
        
    async def remove_multiple_objects(
        self,
        image_id: int,
        bbox_ids: List[int],
        user_id: int,
        expand_mask_pixels: int = 5,
        use_edge_blending: bool = True,
        ldm_steps: int = 25, 
        ldm_sampler: str = 'plms', 
        hd_strategy: str = 'CROP'
    ) -> Dict:
        """
        Remove multiple objects from image in one operation.
        
        Workflow:
            1. Get image and all detections
            2. Filter selected detections by bbox_ids
            3. Download original image
            4. Run ML removal (combined mask)
            5. Upload result to S3
            6. Return result URL
        
        Args:
            1. image_id: ID of image to process
            2. bbox_ids: List of bbox IDs to remove
            3. user_id: ID of user (for authorization)
            4. expand_mask_pixels: Pixels to expand mask (default: 5)
            5. use_edge_blending: Apply edge blending (default: True)
        
        Returns:
            Dict with result URL and metrics
        
        Raises:
            ValueError: If image not found, unauthorized, or no valid detections
        """
        
        image = await self._get_image_authorized(image_id, user_id)
 
        all_detections = await self.detection_repo.get_by_image(image_id)
        selected_detections = [d for d in all_detections if d.bbox_id in bbox_ids]
 
        if not selected_detections:
            raise ValueError(f'No valid detections found for bbox_ids: {bbox_ids}')
        
        image_bytes = await self._get_current_image_bytes(image_id, image.storage_path)
        await self.redis.push_undo_state(image_id, image_bytes, label=f'remove {len(bbox_ids)} objects')
 
        selected_bboxes = [
            {'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2}
            for d in selected_detections
        ]
        scene_bboxes = [
            {'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2}
            for d in all_detections
            if d.bbox_id not in bbox_ids
        ]   
        
        result = await self.pipeline.remove_multiple_objects(
            image_bytes=image_bytes,
            selected_bboxes=selected_bboxes,
            expand_mask_pixels=expand_mask_pixels,
            use_edge_blending=use_edge_blending,
            scene_bboxes=scene_bboxes or None,
            track_metrics=True,
            ldm_steps=ldm_steps,
            ldm_sampler=ldm_sampler,
            hd_strategy=hd_strategy
        )
 
        result_bytes = result['result_bytes']
 
        # Save as current state for next operation
        await self._save_current_state(image_id, result_bytes)
        
        for det in selected_detections:
            await self.db.delete(det)
        await self.db.commit()
        await self.redis.delete(f'image:{image_id}:detections')
        
        bbox_ids_str = '_'.join(map(str, bbox_ids))
        result_path = f"results/{user_id}/{image_id}/remove_multi_{bbox_ids_str}_{int(datetime.utcnow().timestamp())}.jpg"
        result_url = await self.s3.upload_bytes(
            data=result_bytes,
            path=result_path,
            content_type='image/jpeg'
        )
        presigned_url = await self.s3.get_presigned_url(path=result_path, expiration=3600)
 
        return {
            'result_url': result_url,
            'presigned_url': presigned_url,
            'metrics': result['metrics'],
            'timestamp': result['timestamp']
        }
        
    def get_supported_classes(self) -> List[str]:
        """
        Get list of supported YOLO detection classes.
        
        Returns:
            List of class names (80 COCO classes)
        """
        return self.pipeline.get_supported_classes()