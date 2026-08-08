from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from app.repository.mljob_repo import MLJobRepository
from app.repository.image_repo import ImageRepository
from app.repository.image_version_repo import ImageVersionRepository
from app.db.models.mljobs import MLJob
from app.db.enums.ml_task_status import MLTaskType
from app.core.logging import get_logger

logger = get_logger(__name__)


class MLJobService:
    """
    Job-lifecycle bookkeeping, deliberately separate from the ML services —
    those own the actual pipeline work, this only owns "did it run, how
    long did it take, did it fail and why".

    Every job carries two ids now:
      - content_id: WHAT was processed. This is the dedup key — a job
        against the same content_id + task_type that already succeeded
        means the work doesn't need to happen again (see find_completed).
      - image_version_id: WHICH version was current when the job started.
        Purely for tracing/UI ("why is this version's spinner running") —
        never used for dedup, since two different versions can share a
        content_id and both would show up here.

    For edit ops that fork a new version mid-task (remove/replace), the
    job is a record of "this task ran against version N" — the fact that
    it produced version N+1 is ImageEditHistory's job to record, not
    MLJob's.
    """

    def __init__(
        self,
        mljob_repo: MLJobRepository,
        image_repo: ImageRepository,
        image_version_repo: ImageVersionRepository,
    ):
        self.mljob_repo = mljob_repo
        self.image_repo = image_repo
        self.image_version_repo = image_version_repo

    async def _get_authorized_image(self, image_id: int, user_id: int):
        image = await self.image_repo.get_by_id(image_id)
        if not image:
            raise ValueError(f"Image {image_id} not found")
        if image.user_id != user_id:
            raise ValueError("Unauthorized: image belongs to different user")
        return image

    async def find_completed(
        self, image_id: int, user_id: int, task_type: MLTaskType
    ) -> Optional[MLJob]:
        """
        Dedup check for callers that want to skip a pipeline run entirely
        (e.g. DetectorService before calling YOLO). Resolves the image's
        current version, then asks whether its content_id already has a
        SUCCESS job for this task_type.
        """
        image = await self._get_authorized_image(image_id, user_id)
        version = await self.image_version_repo.get_current(image)
        if version is None:
            return None
        return await self.mljob_repo.get_successful(version.content_id, task_type)

    async def start(self, image_id: int, user_id: int, task_type: MLTaskType) -> MLJob:
        image = await self._get_authorized_image(image_id, user_id)
        version = await self.image_version_repo.get_current(image)
        if version is None:
            raise ValueError(f"Image {image_id} has no current version")
        job = await self.mljob_repo.create(version.content_id, version.id, task_type)
        job = await self.mljob_repo.mark_running(job.id)
        logger.info("ml_job_started", job_id=job.id, task_type=task_type)
        return job

    async def complete(self, job_id: int, processing_time_ms: int) -> MLJob:
        job = await self.mljob_repo.mark_success(job_id, processing_time_ms)
        logger.info("ml_job_completed", job_id=job.id, processing_time_ms=processing_time_ms)
        return job

    async def fail(self, job_id: int, error: Exception) -> MLJob:
        message = str(error)[:2000]
        job = await self.mljob_repo.mark_failed(job_id, message)
        logger.warning("ml_job_failed", job_id=job.id, error=message)
        return job

    @asynccontextmanager
    async def track(self, image_id: int, user_id: int, task_type: MLTaskType):
        job = await self.start(image_id, user_id, task_type)   
        started_at = datetime.now(timezone.utc)
        try:
            yield job                                            
        except Exception as e:
            await self.fail(job.id, e)                            
            raise                                                  
        else:
            elapsed_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
            await self.complete(job.id, elapsed_ms)