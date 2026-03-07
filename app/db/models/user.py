from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Boolean, DateTime
from datetime import datetime
from typing import List
from app.db.db_connect import Base
from app.db.models.image import Image

class User(Base):
    
    __tablename__ = 'user'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now())
    
    # Relationships
    images: Mapped[List["Image"]] = relationship(back_populates="user", cascade="all, delete-orphan")