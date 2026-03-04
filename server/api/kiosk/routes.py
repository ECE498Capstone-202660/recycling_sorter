import datetime as dt
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.deps import get_db
from models import KioskLock, User
from services.user.get_user import get_current_user

router = APIRouter()

LOCK_TTL_SECONDS = 600


def _get_lock(db: Session) -> KioskLock:
    lock = db.query(KioskLock).first()
    if lock:
        return lock
    lock = KioskLock(is_locked=False)
    db.add(lock)
    db.commit()
    db.refresh(lock)
    return lock


def _clear_lock(lock: KioskLock) -> None:
    lock.is_locked = False
    lock.locked_by_user_id = None
    lock.locked_at = None
    lock.expires_at = None


def _is_expired(lock: KioskLock, now: dt.datetime) -> bool:
    return bool(lock.is_locked and lock.expires_at and lock.expires_at <= now)


@router.get("/kiosk/status")
def kiosk_status(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    now = dt.datetime.utcnow()
    lock = _get_lock(db)
    if _is_expired(lock, now):
        _clear_lock(lock)
        db.commit()

    if not lock.is_locked:
        return {"status": "idle"}
    if lock.locked_by_user_id == user.id:
        return {"status": "active", "locked_at": lock.locked_at}
    return {"status": "busy"}


@router.post("/kiosk/start")
def kiosk_start(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    now = dt.datetime.utcnow()
    lock = _get_lock(db)
    if _is_expired(lock, now):
        _clear_lock(lock)

    if lock.is_locked and lock.locked_by_user_id != user.id:
        raise HTTPException(status_code=423, detail="Kiosk is in use")

    lock.is_locked = True
    lock.locked_by_user_id = user.id
    lock.locked_at = now
    lock.expires_at = now + dt.timedelta(seconds=LOCK_TTL_SECONDS)
    db.commit()
    return {"status": "active", "locked_at": lock.locked_at}


@router.post("/kiosk/stop")
def kiosk_stop(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    lock = _get_lock(db)
    if not lock.is_locked:
        return {"status": "idle"}
    if lock.locked_by_user_id != user.id:
        raise HTTPException(status_code=403, detail="Not lock owner")

    _clear_lock(lock)
    db.commit()
    return {"status": "idle"}
