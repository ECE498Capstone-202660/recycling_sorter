from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

url = os.getenv("MYSQL_URL")
if not url:
    raise RuntimeError("Missing database environment variable: MYSQL_URL")

# Ensure SQLAlchemy uses PyMySQL driver
if url.startswith("mysql://"):
    url = url.replace("mysql://", "mysql+pymysql://", 1)

DB_URL = url

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
