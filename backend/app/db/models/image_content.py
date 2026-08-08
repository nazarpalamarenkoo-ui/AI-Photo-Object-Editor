import hashlib
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.db_connect import Base


class ImageContent(Base):

    __tablename__ = "image_contents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    content_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)

    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    versions: Mapped[list["ImageVersion"]] = relationship(
        back_populates="content",
    )

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="content",
        cascade="all, delete-orphan",
    )
    segmentation_masks: Mapped[list["SegmentationMask"]] = relationship(
        back_populates="content",
        cascade="all, delete-orphan",
    )
    jobs: Mapped[list["MLJob"]] = relationship(
        back_populates="content",
        cascade="all, delete-orphan",
    )

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """sha256 hex digest вмісту файлу — використовується як content_hash."""
        return hashlib.sha256(data).hexdigest()