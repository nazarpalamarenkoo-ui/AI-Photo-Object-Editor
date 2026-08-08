from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, Float, Integer, DateTime, Boolean, Enum, UniqueConstraint
from app.db.db_connect import Base
from sqlalchemy.sql import func
from app.db.enums.segmentation_mode import SegmentationMode

class SegmentationMask(Base):

    __tablename__ = "segmentation_masks"
    __table_args__ = (
        UniqueConstraint("content_id", "mask_id", name="uq_segmentation_content_mask"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    content_id: Mapped[int] = mapped_column(
        ForeignKey('image_contents.id'),
        nullable=False,
        index=True,
    )

    mask_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    mask_storage_path: Mapped[str] = mapped_column(
        String(500)
    )

    preview_storage_path: Mapped[str] = mapped_column(
        String(500)
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

    area: Mapped[float] = mapped_column(
        Float
    )

    score: Mapped[float] = mapped_column(
        Float
    )

    segmentation_mode: Mapped[SegmentationMode] = mapped_column(
        Enum(
            SegmentationMode,
            name="segmentation_mode",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False
    )
    model_name: Mapped[str] = mapped_column(
        String(100)
    )
    model_version: Mapped[str] = mapped_column(
        String(20)
    )

    inference_time_ms: Mapped[float] = mapped_column(
        Float
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    content: Mapped["ImageContent"] = relationship(back_populates="segmentation_masks")