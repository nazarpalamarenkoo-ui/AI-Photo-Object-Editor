from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.db_connect import Base


class ImageVersion(Base):

    __tablename__ = "image_versions"
    __table_args__ = (
        UniqueConstraint("image_id", "version_number", name="uq_image_version_number"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id"),
        nullable=False,
        index=True,
    )

    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("image_versions.id"),
        nullable=True,
        index=True,
    )

    content_id: Mapped[int] = mapped_column(
        ForeignKey("image_contents.id"),
        nullable=False,
        index=True,
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    storage_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    image: Mapped["Image"] = relationship(
        back_populates="versions",
        foreign_keys=[image_id],
    )

    parent: Mapped["ImageVersion | None"] = relationship(
        remote_side=[id],
        back_populates="children",
    )
    children: Mapped[list["ImageVersion"]] = relationship(
        back_populates="parent",
    )

    content: Mapped["ImageContent"] = relationship(
        back_populates="versions",
    )

    jobs: Mapped[list["MLJob"]] = relationship(
        back_populates="image_version",
        cascade="all, delete-orphan",
    )
    edit_history: Mapped[list["ImageEditHistory"]] = relationship(
        back_populates="image_version",
        cascade="all, delete-orphan",
    )