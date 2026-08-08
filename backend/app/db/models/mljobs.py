from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, Integer, DateTime, Enum, Text
from app.db.db_connect import Base
from sqlalchemy.sql import func
from app.db.enums.ml_task_status import MLTaskType
from app.db.enums.ml_job_status import JobStatus

class MLJob(Base):
    __tablename__ = "ml_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    content_id: Mapped[int] = mapped_column(
        ForeignKey("image_contents.id"),
        nullable=False,
        index=True
    )
    
    image_version_id: Mapped[int] = mapped_column(
        ForeignKey("image_versions.id"),
        nullable=False,
        index=True
    )

    task_type: Mapped[MLTaskType] = mapped_column(
        Enum(MLTaskType, name="ml_task_type", values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        nullable=False
    )

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        nullable=False,
        default=JobStatus.PENDING
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    processing_time_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    content: Mapped["ImageContent"] = relationship(
        back_populates="jobs"
    )
    image_version: Mapped["ImageVersion"] = relationship(
        back_populates="jobs"
    )