from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

def _getenv(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None

DB_URL = (
    f"mysql+pymysql://{_getenv('MYSQL_USER', 'MYSQLUSER')}:{_getenv('MYSQL_PASSWORD', 'MYSQLPASSWORD')}"
    f"@{_getenv('MYSQL_HOST', 'MYSQLHOST')}:{_getenv('MYSQL_PORT', 'MYSQLPORT')}/"
    f"{_getenv('MYSQL_DATABASE', 'MYSQLDATABASE')}"
)

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
