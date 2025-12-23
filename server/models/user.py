import datetime as dt
from sqlalchemy import Column, Integer, String, DateTime
from db import Base

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(128), nullable=False)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    first_name      = Column(String(64), nullable=False)
    last_name       = Column(String(64), nullable=False)
    created_at      = Column(DateTime, default=dt.datetime.utcnow)
