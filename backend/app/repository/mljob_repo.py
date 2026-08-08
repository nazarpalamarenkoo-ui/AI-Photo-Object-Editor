from typing import Optional, List
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.models.mljobs import MLJob
from app.db.enums.ml_task_status import MLTaskType
from app.db.enums.ml_job_status import JobStatus
from app.repository.base_repo import BaseRepository


class MLJobRepository(BaseRepository):

    async def create(self, content_id: int, image_version_id: int, task_type: MLTaskType) -> MLJob:
        async with self.session_factory() as db:
            job = MLJob(
                content_id=content_id,
                image_version_id=image_version_id,
                task_type=task_type,
                status=JobStatus.PENDING,
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)
            return job

    async def get_by_id(self, job_id: int) -> Optional[MLJob]:
        async with self.session_factory() as db:
            result = await db.execute(select(MLJob).where(MLJob.id == job_id))
            return result.scalar_one_or_none()

    async def get_by_content(self, content_id: int) -> List[MLJob]:
        async with self.session_factory() as db:
            result = await db.execute(
                select(MLJob)
                .where(MLJob.content_id == content_id)
                .order_by(MLJob.started_at.desc())
            )
            return result.scalars().all()  # type: ignore

    async def get_by_version(self, image_version_id: int) -> List[MLJob]:
        """Для UI/трасування — "які job'и стартували з цієї версії", незалежно
        від того, чи вміст співпав з чимось іншим."""
        async with self.session_factory() as db:
            result = await db.execute(
                select(MLJob)
                .where(MLJob.image_version_id == image_version_id)
                .order_by(MLJob.started_at.desc())
            )
            return result.scalars().all()  # type: ignore

    async def get_successful(self, content_id: int, task_type: MLTaskType) -> Optional[MLJob]:
        """
        Ключовий метод для дедуплікації: чи вже є завершений SUCCESS job
        для цього вмісту й типу задачі. Якщо є — викликач може одразу
        читати готові Detection/SegmentationMask замість запуску нового
        інференсу. Бере найсвіжіший, якщо їх раптом декілька.
        """
        async with self.session_factory() as db:
            result = await db.execute(
                select(MLJob)
                .where(
                    MLJob.content_id == content_id,
                    MLJob.task_type == task_type,
                    MLJob.status == JobStatus.SUCCESS,
                )
                .order_by(MLJob.finished_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_pending(self, task_type: Optional[MLTaskType] = None, limit: int = 20) -> List[MLJob]:
        """For a worker polling for work."""
        stmt = select(MLJob).where(MLJob.status == JobStatus.PENDING)
        if task_type is not None:
            stmt = stmt.where(MLJob.task_type == task_type)
        stmt = stmt.order_by(MLJob.started_at.asc()).limit(limit)
        async with self.session_factory() as db:
            result = await db.execute(stmt)
            return result.scalars().all()  # type: ignore

    async def mark_running(self, job_id: int) -> MLJob:
        async with self.session_factory() as db:
            job = await db.get(MLJob, job_id)
            job.status = JobStatus.RUNNING
            await db.commit()
            await db.refresh(job)
            return job

    async def mark_success(self, job_id: int, processing_time_ms: int) -> MLJob:
        async with self.session_factory() as db:
            job = await db.get(MLJob, job_id)
            job.status = JobStatus.SUCCESS
            job.finished_at = datetime.now(timezone.utc)
            job.processing_time_ms = processing_time_ms
            await db.commit()
            await db.refresh(job)
            return job

    async def mark_failed(self, job_id: int, error_message: str) -> MLJob:
        async with self.session_factory() as db:
            job = await db.get(MLJob, job_id)
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(timezone.utc)
            job.error_message = error_message
            await db.commit()
            await db.refresh(job)
            return job