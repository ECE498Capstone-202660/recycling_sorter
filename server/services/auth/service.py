from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt
from models import User
import os

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM  = os.getenv("ALGORITHM", "HS256") 
pwd_ctx    = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_pw(pwd: str) -> str:
    return pwd_ctx.hash(pwd)

def verify_pw(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def sign_token(data: dict) -> str:
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# ─── CRUD wrappers ────────────────────────────────────────
def user_exists(username: str, db: Session) -> bool:
    return db.query(User).filter_by(username=username).first() is not None

def add_user(
    username: str,
    password: str,
    db: Session,
    email: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
) -> User:
    user = User(
        username=username,
        hashed_password=hash_pw(password),
        email=email,
        first_name=first_name,
        last_name=last_name,
    )
    db.add(user); db.commit(); db.refresh(user)
    return user

def get_hashed(username: str, db: Session) -> str | None:
    u = db.query(User).filter_by(username=username).first()
    return u.hashed_password if u else None
