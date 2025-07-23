import datetime as dt
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, ForeignKey
)
from sqlalchemy.orm import relationship
from db import Base

class ClassificationLog(Base):
    __tablename__ = "classification_logs"

    id              = Column(Integer, primary_key=True, index=True)
    # user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    predicted_class = Column(String(32), nullable=False)
    confidence      = Column(Float, nullable=False)
    raw_output      = Column(JSON)
    image_url       = Column(String(256))
    rebate          = Column(Float)
    created_at      = Column(DateTime, default=dt.datetime.utcnow)

    # user = relationship("User", back_populates="logs")