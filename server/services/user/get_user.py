from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
from db.deps import get_db
from services.auth.service import (
    user_exists,
    SECRET_KEY,
    ALGORITHM,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> str:
    try:
        username: str = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])["sub"]
    except (JWTError, KeyError):
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )
    if not user_exists(username, db):
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )
    return username