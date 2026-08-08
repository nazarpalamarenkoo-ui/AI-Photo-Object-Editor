from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.db_connect import Base
from app.db.enums.edit_operation import EditOperation
from app.db.enums.engine_types import EngineType


class ImageEditHistory(Base):

    __tablename__ = "image_edit_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    image_version_id: Mapped[int] = mapped_column(
        ForeignKey("image_versions.id"),
        nullable=False,
        index=True,
    )

    operation: Mapped[EditOperation] = mapped_column(
        Enum(EditOperation, name="edit_operation", values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        nullable=False,
    )

    engine: Mapped[EngineType] = mapped_column(
        Enum(EngineType, name="engine_type", values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        nullable=False,
    )

    parameters: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    processing_time_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    image_version: Mapped["ImageVersion"] = relationship(back_populates="edit_history")