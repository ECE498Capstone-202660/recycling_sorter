from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from db.deps import get_db
from services.auth.service import (
    user_exists,
    add_user,
    verify_pw,
    sign_token,
    get_hashed,
    SECRET_KEY,
    ALGORITHM,
)
from schemas.auth import Token

router = APIRouter()
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


@router.post("/register", response_model=Token)
def register(
    username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)
):
    if user_exists(username, db):
        raise HTTPException(status_code=400, detail="Username already exists")
    add_user(username, password, db)
    return {"token": sign_token({"sub": username})}


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    hashed = get_hashed(form_data.username, db)
    if not hashed or not verify_pw(form_data.password, hashed):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": sign_token({"sub": form_data.username})}


@router.post("/logout")
def logout():
    return {"message": "Logged out successfully"}
