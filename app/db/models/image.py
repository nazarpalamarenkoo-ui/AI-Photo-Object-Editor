from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, ForeignKey
from datetime import datetime
from typing import List
from app.db.db_connect import Base
from app.db.models.user import User
from app.db.models.detection import Detection
class Image(Base):
    """Image model"""
    __tablename__ = "images"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(String(500))
    cache_key: Mapped[str] = mapped_column(String(100), nullable=True)  # Redis cache key
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(20), default="uploaded")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="images")
    detections: Mapped[List["Detection"]] = relationship(back_populates="image", cascade="all, delete-orphan")