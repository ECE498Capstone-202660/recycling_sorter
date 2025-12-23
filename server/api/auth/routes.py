from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from db.deps import get_db
from models import User
from services.auth.service import (
    user_exists,
    add_user,
    verify_pw,
    sign_token,
    get_hashed,
)
from schemas.auth import Token

router = APIRouter()

@router.post("/register", response_model=Token)
def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str | None = Form(None),
    first_name: str | None = Form(None),
    last_name: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if user_exists(username, db):
        raise HTTPException(status_code=400, detail="Username already exists")
    if email:
        existing_email = db.query(User).filter_by(email=email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already exists")
    add_user(username, password, db, email=email, first_name=first_name, last_name=last_name)
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
