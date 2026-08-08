from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, Float, Integer, DateTime, Boolean, UniqueConstraint
from app.db.db_connect import Base
from sqlalchemy.sql import func
class Detection(Base):

    __tablename__ = "detections"
    __table_args__ = (
        UniqueConstraint("content_id", "bbox_id", name="uq_detection_content_bbox"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    content_id: Mapped[int] = mapped_column(
        ForeignKey("image_contents.id"),
        nullable=False,
        index=True
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true"
    )
    bbox_id: Mapped[int] = mapped_column(
        Integer
    )
    x1: Mapped[int] = mapped_column(
        Integer
    )
    y1: Mapped[int] = mapped_column(
        Integer
    )
    x2: Mapped[int] = mapped_column(
        Integer
    )
    y2: Mapped[int] = mapped_column(
        Integer
    )
    detected_class: Mapped[str] = mapped_column(
        String(100)
    )
    confidence: Mapped[float] = mapped_column(
        Float
    )
    model_name: Mapped[str] = mapped_column(
        String(100)
    )
    model_version: Mapped[str] = mapped_column(
        String(30)
    )
    inference_time_ms: Mapped[float] = mapped_column(
        Float
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    content: Mapped["ImageContent"] = relationship(back_populates="detections")