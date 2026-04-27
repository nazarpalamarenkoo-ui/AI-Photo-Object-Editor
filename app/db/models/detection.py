from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, Float, Integer
from app.db.db_connect import Base

class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    bbox_id: Mapped[int] = mapped_column(Integer)
    x1: Mapped[int] = mapped_column()
    y1: Mapped[int] = mapped_column()
    x2: Mapped[int] = mapped_column()
    y2: Mapped[int] = mapped_column()
    detected_class: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)

    image: Mapped["Image"] = relationship(back_populates="detections")