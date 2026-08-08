from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.db_connect import Base
from app.db.enums.image_status import ImageStatus
class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    storage_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False
    )

    mime_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    width: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    height: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    file_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False
    )

    cache_key: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    current_version_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "image_versions.id",
            use_alter=True,
            name="fk_images_current_version_id",
        ),
        nullable=True,
        index=True,
    )

    status: Mapped[ImageStatus] = mapped_column(
        Enum(
            ImageStatus,
            name="image_status",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=ImageStatus.UPLOADED,
        server_default=ImageStatus.UPLOADED.value
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    user: Mapped["User"] = relationship(
        back_populates="images"
    )

    versions: Mapped[list["ImageVersion"]] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        foreign_keys="ImageVersion.image_id"
    )

    current_version: Mapped["ImageVersion | None"] = relationship(
        foreign_keys=[current_version_id],
        post_update=True
    )