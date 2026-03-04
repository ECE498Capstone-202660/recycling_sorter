import datetime as dt
from sqlalchemy import Column, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db import Base


class KioskLock(Base):
    __tablename__ = "kiosk_lock"

    id = Column(Integer, primary_key=True, index=True)
    is_locked = Column(Boolean, nullable=False, default=False)
    locked_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    locked_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    user = relationship("User")
