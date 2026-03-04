import os
import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile, Header
from sqlalchemy.orm import Session
from db.deps import get_db
from models import ClassificationLog, User, KioskLock
from schemas.classification import PredictionResponse, HistoryItem
from services.user.get_user import get_current_user
from services.classification.image_upload import save_uploaded_image
from services.classification.inference import preprocess_image, run_inference_model

router = APIRouter()
MACHINE_TOKEN = os.getenv("MACHINE_TOKEN")

def to_dict(row: ClassificationLog) -> dict:
    return {
        "predicted_class": row.predicted_class,
        "confidence": f"{row.confidence*100:.2f}%",
        "raw_output": row.raw_output,
        "image_url": row.image_url,
        "rebate": row.rebate,
        "created_at": row.created_at.isoformat()
    }


def _require_machine_token(x_machine_token: str | None) -> None:
    if not MACHINE_TOKEN or not x_machine_token or x_machine_token != MACHINE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid machine token")


def _get_lock(db: Session) -> KioskLock:
    lock = db.query(KioskLock).first()
    if lock:
        return lock
    lock = KioskLock(is_locked=False)
    db.add(lock)
    db.commit()
    db.refresh(lock)
    return lock


@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    weight: float = Form(...),
    x_machine_token: str | None = Header(default=None, alias="X-Machine-Token"),
    db: Session = Depends(get_db)
):
    _require_machine_token(x_machine_token)

    lock = _get_lock(db)
    now = dt.datetime.utcnow()
    if lock.is_locked and lock.expires_at and lock.expires_at <= now:
        lock.is_locked = False
        lock.locked_by_user_id = None
        lock.locked_at = None
        lock.expires_at = None
        db.commit()

    weight = weight * 0.054
    img_bytes = await image.read()
    tensor = preprocess_image(img_bytes)
    res = run_inference_model(tensor, weight_grams=weight)

    rebate_by_category = {
        "Glass": 0.10,
        "Metal": 0.10,
        "Paper": 0.10,
        "Plastic": 0.10,
        "Trash": 0.00,
    }
    rebate = rebate_by_category.get(res["predicted_class"], 0.0)

    if lock.is_locked and lock.locked_by_user_id:
        url = save_uploaded_image(img_bytes, image.filename)
        row = ClassificationLog(
            user_id=lock.locked_by_user_id,
            predicted_class=res["predicted_class"],
            confidence=res["confidence"],
            raw_output=res["raw_output"],
            image_url=url,
            rebate=rebate,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return to_dict(row)

    return {
        "predicted_class": res["predicted_class"],
        "confidence": f"{res['confidence']*100:.2f}%",
        "raw_output": res["raw_output"],
        "image_url": None,
        "rebate": rebate,
        "created_at": dt.datetime.utcnow().isoformat(),
    }

@router.get("/latest", response_model=list[PredictionResponse])
def latest(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    rows = (
        db.query(ClassificationLog)
          .filter(ClassificationLog.user_id == user.id)
          .order_by(ClassificationLog.created_at.desc())
          .limit(3)
          .all()
    )
    return [to_dict(r) for r in rows]

@router.get("/history", response_model=list[HistoryItem])
def history(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    user = db.query(User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    rows = (
        db.query(ClassificationLog)
          .filter(ClassificationLog.user_id == user.id)
          .order_by(ClassificationLog.created_at.desc())
          .all()
    )
    return [
        {
            "id": r.id,
            "category": r.predicted_class,
            "date": r.created_at.strftime("%Y-%m-%d"),
            "rebate": round(r.rebate, 2),
            "image_url": r.image_url,
        }
        for r in rows
    ]
