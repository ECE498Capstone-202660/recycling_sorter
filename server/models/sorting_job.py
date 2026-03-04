import datetime as dt
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from db import Base


class SortingJob(Base):
    __tablename__ = "sorting_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    status = Column(String(16), nullable=False, default="queued")
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
