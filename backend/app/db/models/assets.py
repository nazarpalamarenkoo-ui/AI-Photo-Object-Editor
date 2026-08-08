import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.db_connect import Base


class Asset(Base):

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    public_id: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        index=True,
        default=lambda: uuid.uuid4().hex,
        nullable=False,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_image_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("image_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_segmentation_mask_id: Mapped[int | None] = mapped_column(
        ForeignKey("segmentation_masks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    content_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="image/png",
        server_default="image/png",
    )

    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    area_pixels: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    label: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="assets")

    source_image_version: Mapped["ImageVersion | None"] = relationship()
    source_segmentation_mask: Mapped["SegmentationMask | None"] = relationship()