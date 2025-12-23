from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.deps import get_db
from models import User
from schemas.user import UserOut, UserUpdate
from services.user.get_user import get_current_user

router = APIRouter()


@router.get("/me", response_model=UserOut)
def read_me(
    current_username: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter_by(username=current_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/{user_id}", response_model=UserOut)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/me", response_model=UserOut)
def update_me(
    payload: UserUpdate,
    current_username: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter_by(username=current_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    updates = payload.dict(exclude_unset=True)
    if "email" in updates and updates["email"]:
        exists = (
            db.query(User)
            .filter(User.email == updates["email"], User.id != user.id)
            .first()
        )
        if exists:
            raise HTTPException(status_code=409, detail="Email already in use")

    for field, value in updates.items():
        setattr(user, field, value)

    db.add(user)
    db.commit()
    db.refresh(user)
    return user
